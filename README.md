# SatQuery AI

An agentic vision-language assistant for multimodal remote-sensing imagery.
Upload one satellite image or a co-registered pair, ask a question in plain
English, and the controller works out which task you asked for, which specialist
tool answers it, and what evidence supports the answer.

SIH 2026 · problem statement **SIH26167** (ISRO / Department of Space).

## Status: Phase A - the controller spine

The orchestration layer is complete and runs end to end. The analysis tools are
declared in the registry with their real parameter schemas, and the ones that
are not implemented yet say so rather than returning a fabricated result.

| Component | State |
| --- | --- |
| Sensor Passport - sensor, resolution, CRS, extent, acquisition date | done |
| Input compatibility checking and pair rejection | done |
| Query interpretation and task routing (offline, no model needed) | done |
| Predefined tool registry with permitted-parameter enforcement | done |
| Execution trace, JSON and PDF reports | done |
| Synthetic fixtures with known ground truth | done |
| HTTP API, CLI, minimal web console | done |
| Spectral / SAR indices, change detection, reliability map, verifier | Phase B |
| Neural VQA, grounding, change and fusion models | Phase D |
| BigEarthNet adaptation of the dual-branch encoder | Phase E |

## Quick start

On NixOS, the dev shell puts the shared libraries that the PyPI wheels expect on
`LD_LIBRARY_PATH`:

```sh
nix develop            # flake.nix - if the directory is a git repo, git add flake.nix first
./scripts/setup.sh     # creates .venv, installs requirements, generates fixtures
```

On any other Linux or macOS machine, skip `nix develop` and run `scripts/setup.sh`
directly.

Then:

```sh
./scripts/demo.sh      # every demo scenario, over the synthetic fixtures
./scripts/test.sh      # the test suite
./scripts/serve.sh     # API and console on http://127.0.0.1:8000
```

Set `SATQUERY_FORCE_CPU=1` to force the no-GPU path, which is how the classical
fallback is exercised on a machine that has a card.

## Command line

```sh
python -m satquery.cli tools -v                      # the registry and its parameters
python -m satquery.cli fixtures                      # regenerate the fixture set
python -m satquery.cli ask "What changed?" T1.tif T2.tif --pdf report.pdf
python -m satquery.cli demo                          # all scenarios
```

## How a query is handled

```
upload -> sensor passport (per image)
       -> compatibility check  --- incompatible? reject here, before any model
       -> query interpretation -> task
       -> tool selection from the registry, constrained by the input configuration
       -> permitted parameters bound; anything else rejected and recorded
       -> tools executed in sequence
       -> outputs integrated: answer, confidence, artifacts
       -> execution trace -> JSON and PDF report
```

The problem statement says internal planning text is not evaluated, only the
observable trace: the task, the tools, the permitted parameters and the outputs.
So the trace is the primary output, and every stage writes to it while it runs.

## Input configurations

The three legal configurations are decided from the passports, not from what the
user claims:

| Configuration | Detected from |
| --- | --- |
| Single | one image |
| Cross-modal pair | one optical and one SAR image over the same footprint |
| Bi-temporal pair | two images of the same modality, different dates |

A pair that fails a check - footprints that do not overlap, resolutions more than
four times apart, one image georeferenced and the other not - is rejected with
the reason recorded. No tool runs.

GeoTIFF is the expected format. PNG and JPEG are accepted for the benchmark
datasets (VRSBench, RSVQA, CDVQA), which carry no georeferencing; for those, the
passport sets `georeferenced: false` and every spatial answer stays in pixel
coordinates rather than km².

## Layout

```
satquery/
  passport.py       Sensor Passport: geometry, CRS, extent, dates
  sensors.py        modality and sensor inference, with evidence
  compat.py         pair compatibility and configuration classification
  router.py         query interpretation and task routing
  registry.py       tool specs, permitted parameters, selection
  tools/            the registry contents; geometry.py is implemented
  controller.py     the agentic pipeline
  trace.py          the execution trace
  report.py         JSON and PDF export
  api.py            FastAPI application
  cli.py            command line
  fixtures/synth.py synthetic imagery with known ground truth
tests/              55 tests over the above
```

## Synthetic fixtures

Dataset download is the long pole of this project, and the controller should not
wait for it. `satquery.fixtures.synth` builds Sentinel-2-like and
Sentinel-1-like rasters from a hand-drawn land-cover map, with plausible
reflectance, dB backscatter and speckle. Because the label map is known, the
fixture knows its own answer: the bi-temporal pair contains a planted amount of
new built-up area and a planted amount of lost water, so a change tool can be
tested against a number rather than against a plausible-looking picture.

The set also includes a scene 200 km away, to exercise rejection, and an
un-georeferenced PNG, to exercise the benchmark path.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | version, GPU state, how many tools are implemented |
| `GET /api/tools` | the registry, with each tool's permitted parameters |
| `POST /api/upload` | one or two images; returns passports and the compatibility verdict |
| `POST /api/query` | run a query against an uploaded session |
| `GET /api/runs/{id}` | the stored run |
| `GET /api/runs/{id}/report.json` | machine-readable report |
| `GET /api/runs/{id}/report.pdf` | audit report |

Upload is separate from query on purpose: the compatibility verdict is visible
before a query runs, so a rejected pair is explained rather than appearing as a
failed query.

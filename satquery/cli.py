"""Command-line access to the controller.

Useful on its own, and the fastest way to check that a change to routing or the
registry did what was intended without starting the server.

    python -m satquery.cli fixtures
    python -m satquery.cli tools
    python -m satquery.cli ask "what changed?" T1.tif T2.tif
    python -m satquery.cli demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .controller import Controller
from .registry import REGISTRY
from .report import write_json_report, write_pdf_report
from .runtime import detect_gpu

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, RED, BLUE = "\033[32m", "\033[33m", "\033[31m", "\033[36m"

_STATUS_COLOUR = {
    "ok": GREEN, "pass": GREEN,
    "warn": YELLOW, "not_implemented": YELLOW, "no_answer": YELLOW, "no_tool": YELLOW,
    "fail": RED, "failed": RED, "rejected": RED,
}


def _colour(text: str, code: str, enabled: bool) -> str:
    return f"{code}{text}{RESET}" if enabled else text


def print_result(result, *, colour: bool = True, verbose: bool = False) -> None:
    trace = result.trace
    c = lambda t, code: _colour(t, code, colour)  # noqa: E731

    print(c(f"\nrun {trace.run_id}", BOLD), c(f"({trace.total_ms:.0f} ms)", DIM))
    print(c("query:", DIM), trace.query)
    print(c("inputs:", DIM), ", ".join(trace.input_files))

    for passport in result.passports:
        print(c("  passport:", DIM), passport.filename, "->", passport.summary_line())

    if result.compatibility:
        print(c("configuration:", DIM),
              c(result.compatibility.configuration.value,
                GREEN if result.compatibility.compatible else RED))
        for check in result.compatibility.checks:
            mark = _STATUS_COLOUR.get(check.status.value, "")
            label = f"{check.status.value.upper():>4}"
            print(f"  {c(label, mark)}  {check.name}: {check.detail}")

    if trace.rejected:
        print(c("\nREJECTED", RED))
        for reason in trace.rejection_reasons:
            print("  -", reason)
        return

    if result.interpretation:
        interpretation = result.interpretation
        print(c("task:", DIM), c(interpretation.task.value, BLUE),
              c(f"(confidence {interpretation.confidence:.2f}, {interpretation.method})", DIM))
        if interpretation.note:
            print(c(f"  note: {interpretation.note}", DIM))
        if interpretation.entities.get("classes"):
            print(c("  classes:", DIM), ", ".join(interpretation.entities["classes"]))
        if interpretation.entities.get("target_phrase"):
            print(c("  target:", DIM), interpretation.entities["target_phrase"])

    print(c("tools:", DIM), ", ".join(trace.tools) or "none")
    for name, tool_result in result.executed:
        mark = _STATUS_COLOUR.get(tool_result.status.value, "")
        print(f"  {c(tool_result.status.value, mark)}  {name}"
              + (f"  {c(tool_result.detail, DIM)}" if tool_result.detail else ""))
        if verbose and tool_result.metrics:
            print(c(f"      {json.dumps(tool_result.metrics)[:400]}", DIM))

    if trace.answer:
        print(c("\nanswer:", BOLD), trace.answer)
    if trace.confidence is not None:
        print(c("confidence:", DIM), f"{trace.confidence:.2f}")
    if trace.warnings:
        print(c(f"\n{len(trace.warnings)} warning(s):", YELLOW))
        for warning in trace.warnings:
            print("  -", warning)


def cmd_fixtures(args: argparse.Namespace) -> int:
    from .fixtures.synth import make_demo_set

    manifest = make_demo_set(args.outdir, size=args.size)
    print(f"wrote {len(manifest['files'])} fixtures to {args.outdir}")
    for key, entry in manifest["files"].items():
        print(f"  {key:<20} {os.path.basename(entry['path'])}")
    print("\nbi-temporal ground truth (km2 change, T1 -> T2):")
    for name, delta in manifest["bitemporal_truth_km2"].items():
        print(f"  {name:<12} {delta:+.4f}")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    gpu = detect_gpu()
    print(f"GPU: {gpu.name or 'none'} ({gpu.reason or 'available'})\n")
    print(f"{'tool':<22} {'backend':<10} {'impl':<6} {'vram':<7} tasks")
    print("-" * 100)
    for spec in REGISTRY.all():
        print(f"{spec.name:<22} {spec.backend:<10} "
              f"{'yes' if spec.implemented else 'no':<6} "
              f"{(str(spec.vram_mb) + 'MB') if spec.vram_mb else '-':<7} "
              f"{', '.join(t.value for t in spec.tasks)}")
        if args.verbose and spec.params:
            for param in spec.params:
                bounds = ""
                if param.choices:
                    bounds = f" in {list(param.choices)}"
                elif param.minimum is not None or param.maximum is not None:
                    bounds = f" [{param.minimum}, {param.maximum}]"
                print(f"    - {param.name}: {param.type} = {param.default}{bounds}")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    controller = Controller(workdir=args.workdir)
    result = controller.run(args.query, args.images)
    print_result(result, colour=not args.no_colour, verbose=args.verbose)

    if args.json:
        write_json_report(result, args.json)
        print(f"\nwrote {args.json}")
    if args.pdf:
        write_pdf_report(result, args.pdf)
        print(f"wrote {args.pdf}")
    return 0 if not result.trace.rejected else 2


DEMO_SCENARIOS = [
    ("Single-image VQA",
     "How many water bodies are visible in this image?", ["single_optical"]),
    ("Single-image captioning",
     "Describe the land cover and major objects visible in this image.", ["single_optical"]),
    ("Text-guided grounding",
     "Highlight the water body referred to in the query.", ["single_optical"]),
    ("Change description",
     "What changed between these two dates, and where did the change occur?",
     ["bitemporal_t1", "bitemporal_t2"]),
    ("Change VQA",
     "Has the built-up area increased, decreased, or remained unchanged?",
     ["bitemporal_t1", "bitemporal_t2"]),
    ("Optical-SAR analysis",
     "Use the optical and SAR images together to identify built-up and water-covered regions.",
     ["cross_optical", "cross_sar"]),
    ("Rejection of an incompatible pair",
     "What changed between these two dates?",
     ["bitemporal_t1", "incompatible_far"]),
]


def cmd_demo(args: argparse.Namespace) -> int:
    from .fixtures.synth import make_demo_set

    manifest_path = os.path.join(args.fixtures, "manifest.json")
    if not os.path.isfile(manifest_path):
        print(f"generating fixtures in {args.fixtures} ...")
        manifest = make_demo_set(args.fixtures)
    else:
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)

    controller = Controller(workdir=args.workdir)
    failures = 0
    for title, query, keys in DEMO_SCENARIOS:
        print("\n" + "=" * 78)
        print(_colour(title, BOLD, not args.no_colour))
        print("=" * 78)
        try:
            paths = [manifest["files"][k]["path"] for k in keys]
        except KeyError as exc:
            print(f"missing fixture {exc}; re-run 'fixtures'")
            failures += 1
            continue
        result = controller.run(query, paths)
        print_result(result, colour=not args.no_colour, verbose=args.verbose)
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    # --no-colour is shared, so it works before or after the subcommand.
    colour = argparse.ArgumentParser(add_help=False)
    colour.add_argument("--no-colour", action="store_true", help="disable ANSI colour")

    parser = argparse.ArgumentParser(prog="satquery", description=__doc__, parents=[colour])
    subparsers = parser.add_subparsers(dest="command", required=True)

    fixtures = subparsers.add_parser("fixtures", parents=[colour], help="generate the synthetic fixture set")
    fixtures.add_argument("--outdir", default="data/fixtures")
    fixtures.add_argument("--size", type=int, default=512)
    fixtures.set_defaults(func=cmd_fixtures)

    tools = subparsers.add_parser("tools", parents=[colour], help="list the predefined tool registry")
    tools.add_argument("-v", "--verbose", action="store_true", help="show permitted parameters")
    tools.set_defaults(func=cmd_tools)

    ask = subparsers.add_parser("ask", parents=[colour], help="run one query over one or two images")
    ask.add_argument("query")
    ask.add_argument("images", nargs="+")
    ask.add_argument("--workdir", default="data/runs")
    ask.add_argument("--json", help="write the JSON report to this path")
    ask.add_argument("--pdf", help="write the PDF report to this path")
    ask.add_argument("-v", "--verbose", action="store_true")
    ask.set_defaults(func=cmd_ask)

    demo = subparsers.add_parser("demo", parents=[colour], help="run every demo scenario over the fixtures")
    demo.add_argument("--fixtures", default="data/fixtures")
    demo.add_argument("--workdir", default="data/runs")
    demo.add_argument("-v", "--verbose", action="store_true")
    demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

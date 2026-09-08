"""Population of the predefined tool registry.

Every tool named on the technical-approach slide is declared here with its real
task coverage, its legal input configurations and its permitted parameters. The
ones whose implementation is still pending are registered as
``implemented=False``: the controller will still select and report them, and the
trace will say the implementation is pending rather than pretending a result.

Phase B fills in the classical analysis tools, Phase D the neural ones.
"""

from __future__ import annotations

from ..enums import InputConfiguration, Modality, Task
from ..registry import REGISTRY, ToolParam, ToolResult, ToolSpec
from .change import run_change_index_diff
from .fusion import run_modality_reliability
from .geometry import PAIR_GEOMETRY, SCENE_STATISTICS
from .optical import run_spectral_index
from .sar import run_sar_backscatter
from .verify import run_physics_verifier

SINGLE = InputConfiguration.SINGLE
CROSS = InputConfiguration.CROSS_MODAL_PAIR
BITEMP = InputConfiguration.BI_TEMPORAL_PAIR


def _pending(name: str, phase: str):
    def run(_ctx) -> ToolResult:
        return ToolResult.pending(name, phase)

    return run


# --- Phase B: classical remote-sensing analysis, CPU only ------------------
# These are implemented and run without a GPU. They are also the fallback that
# keeps every query answerable when the neural tools cannot be loaded.

SPECTRAL_INDEX = ToolSpec(
    name="spectral_index",
    version="0.1",
    summary="Computes NDVI, NDWI, MNDWI, NDBI and BSI, and reports class areas in km2.",
    backend="classical",
    tasks=(Task.VQA, Task.CAPTION, Task.CROSS_MODAL_ANALYSIS),
    configurations=(SINGLE, CROSS),
    params=(
        ToolParam("index", "str", "auto",
                  "Which index to compute; 'auto' picks from the query.",
                  choices=("auto", "ndvi", "ndwi", "mndwi", "ndbi", "bsi")),
        ToolParam("threshold", "float", 0.0,
                  "Decision threshold, applied when 'index' names a specific index. "
                  "Ignored in 'auto' mode, which uses the per-class thresholds.",
                  minimum=-1.0, maximum=1.0),
        ToolParam("min_region_px", "int", 25, "Discard connected regions smaller than this.",
                  minimum=0, maximum=100_000),
    ),
    required_modalities=(Modality.OPTICAL,),
    required_roles=("red", "nir"),
    priority=60,
    implemented=True,
    run=run_spectral_index,
)

SAR_BACKSCATTER = ToolSpec(
    name="sar_backscatter",
    version="0.1",
    summary="Thresholds VV/VH backscatter to separate water, built-up and vegetated surfaces.",
    backend="classical",
    tasks=(Task.VQA, Task.CAPTION, Task.CROSS_MODAL_ANALYSIS),
    configurations=(SINGLE, CROSS),
    params=(
        ToolParam("water_vv_db", "float", -18.0, "VV backscatter below this is called water.",
                  minimum=-40.0, maximum=0.0),
        ToolParam("builtup_vv_db", "float", -5.0, "VV backscatter above this is called built-up.",
                  minimum=-30.0, maximum=15.0),
        ToolParam("speckle_filter", "str", "lee", "Speckle filter applied before thresholding.",
                  choices=("none", "lee", "median")),
    ),
    required_modalities=(Modality.SAR,),
    required_roles=("vv",),
    priority=60,
    implemented=True,
    run=run_sar_backscatter,
)

CHANGE_INDEX_DIFF = ToolSpec(
    name="change_index_diff",
    version="0.1",
    summary="Bi-temporal index differencing with Otsu thresholding; yields a change map and "
            "a signed area delta per class.",
    backend="classical",
    tasks=(Task.CHANGE_MAP, Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA),
    configurations=(BITEMP,),
    params=(
        ToolParam("index", "str", "auto", "Index differenced between the two dates.",
                  choices=("auto", "ndvi", "ndwi", "mndwi", "ndbi", "cva")),
        ToolParam("change_threshold", "float", 0.5,
                  "Normalised magnitude above which a pixel is called changed.",
                  minimum=0.0, maximum=1.0),
        ToolParam("thresholding", "str", "otsu", "How the threshold is chosen.",
                  choices=("otsu", "fixed")),
        ToolParam("min_region_px", "int", 25, "Discard change regions smaller than this.",
                  minimum=0, maximum=100_000),
    ),
    priority=60,
    implemented=True,
    run=run_change_index_diff,
)

MODALITY_RELIABILITY = ToolSpec(
    name="modality_reliability",
    version="0.1",
    summary="Builds the per-region map of which sensor answered, flagging where cloud blocks "
            "optical and SAR carries the answer.",
    backend="classical",
    tasks=(Task.CROSS_MODAL_ANALYSIS,),
    configurations=(CROSS,),
    params=(
        ToolParam("cloud_threshold", "float", 0.28,
                  "Top-of-atmosphere brightness above which optical is called cloud-obscured.",
                  minimum=0.0, maximum=1.0),
        ToolParam("agreement_tolerance", "float", 0.15,
                  "Index margin within which an optical pixel is called borderline and "
                  "excluded from the agreement statistic.",
                  minimum=0.0, maximum=1.0),
    ),
    required_modalities=(Modality.OPTICAL, Modality.SAR),
    priority=65,
    implemented=True,
    run=run_modality_reliability,
)

PHYSICS_VERIFIER = ToolSpec(
    name="physics_verifier",
    version="0.1",
    summary="Cross-checks a model's claim against spectral and radar indices, and converts "
            "agreement into a measured confidence.",
    backend="classical",
    tasks=(Task.VQA, Task.CAPTION, Task.CHANGE_VQA, Task.CHANGE_DESCRIPTION,
           Task.CHANGE_MAP, Task.CROSS_MODAL_ANALYSIS),
    configurations=(SINGLE, CROSS, BITEMP),
    params=(
        ToolParam("strictness", "str", "balanced", "How much index disagreement is tolerated.",
                  choices=("lenient", "balanced", "strict")),
    ),
    priority=10,  # runs last, over another tool's answer
    implemented=True,
    run=run_physics_verifier,
)

# --- Phase D: neural specialist models -------------------------------------

RS_VLM_VQA = ToolSpec(
    name="rs_vlm_vqa",
    version="0.1",
    summary="Remote-sensing adapted vision-language model for visual question answering "
            "and scene captioning (Qwen2.5-VL-3B, 4-bit, with a LoRA adapter).",
    backend="neural",
    tasks=(Task.VQA, Task.CAPTION),
    configurations=(SINGLE,),
    params=(
        ToolParam("max_new_tokens", "int", 128, "Maximum length of the generated answer.",
                  minimum=8, maximum=512),
        ToolParam("temperature", "float", 0.2, "Sampling temperature.",
                  minimum=0.0, maximum=1.5),
        ToolParam("image_max_side", "int", 896, "Longest side the image is resized to.",
                  minimum=224, maximum=1344),
        ToolParam("use_adapter", "bool", True, "Load the BigEarthNet LoRA adapter."),
    ),
    requires_gpu=True,
    vram_mb=2200,
    priority=90,
    implemented=False,
    run=_pending("rs_vlm_vqa", "Phase D"),
)

TEXT_GROUNDING = ToolSpec(
    name="text_grounding",
    version="0.1",
    summary="Text-guided region grounding: returns boxes or masks for the phrase in the query "
            "(GroundingDINO-SwinT).",
    backend="neural",
    tasks=(Task.GROUNDING,),
    configurations=(SINGLE,),
    params=(
        ToolParam("box_threshold", "float", 0.30, "Minimum box confidence.",
                  minimum=0.0, maximum=1.0),
        ToolParam("text_threshold", "float", 0.25, "Minimum phrase-match confidence.",
                  minimum=0.0, maximum=1.0),
        ToolParam("max_detections", "int", 20, "Maximum number of regions returned.",
                  minimum=1, maximum=300),
    ),
    requires_gpu=True,
    vram_mb=700,
    priority=90,
    implemented=False,
    run=_pending("text_grounding", "Phase D"),
)

CHANGE_ENCODER = ToolSpec(
    name="change_encoder",
    version="0.1",
    summary="Learned bi-temporal change segmentation producing a spatial change map "
            "(BIT / ChangeFormer).",
    backend="neural",
    tasks=(Task.CHANGE_MAP,),
    configurations=(BITEMP,),
    params=(
        ToolParam("change_threshold", "float", 0.5, "Probability above which a pixel is changed.",
                  minimum=0.0, maximum=1.0),
        ToolParam("tile_size", "int", 256, "Tile size used for inference.",
                  choices=(128, 256, 512)),
        ToolParam("min_region_px", "int", 25, "Discard change regions smaller than this.",
                  minimum=0, maximum=100_000),
    ),
    requires_gpu=True,
    vram_mb=300,
    priority=90,
    implemented=False,
    run=_pending("change_encoder", "Phase D"),
)

VLM_CHANGE_QA = ToolSpec(
    name="vlm_change_qa",
    version="0.1",
    summary="Describes and answers questions about what changed between two dates, grounded "
            "in the change map.",
    backend="neural",
    tasks=(Task.CHANGE_DESCRIPTION, Task.CHANGE_VQA),
    configurations=(BITEMP,),
    params=(
        ToolParam("max_new_tokens", "int", 160, "Maximum length of the generated answer.",
                  minimum=8, maximum=512),
        ToolParam("use_change_map", "bool", True,
                  "Condition the answer on the change map from change_encoder."),
    ),
    requires_gpu=True,
    vram_mb=2200,
    priority=90,
    implemented=False,
    run=_pending("vlm_change_qa", "Phase D"),
)

OPTICAL_SAR_FUSION = ToolSpec(
    name="optical_sar_fusion",
    version="0.1",
    summary="Dual-branch Sentinel-1 + Sentinel-2 encoder adapted on BigEarthNet; extracts "
            "complementary land-cover information from a co-registered pair.",
    backend="neural",
    tasks=(Task.CROSS_MODAL_ANALYSIS,),
    configurations=(CROSS,),
    params=(
        ToolParam("top_k", "int", 5, "Number of land-cover classes reported.",
                  minimum=1, maximum=19),
        ToolParam("min_probability", "float", 0.4, "Minimum class probability to report.",
                  minimum=0.0, maximum=1.0),
        ToolParam("tile_size", "int", 120, "Tile size the encoder was trained at.",
                  choices=(120, 224, 256)),
    ),
    required_modalities=(Modality.OPTICAL, Modality.SAR),
    requires_gpu=True,
    vram_mb=400,
    priority=95,
    implemented=False,
    run=_pending("optical_sar_fusion", "Phase E"),
)


BUILTIN_TOOLS: tuple[ToolSpec, ...] = (
    SCENE_STATISTICS,
    PAIR_GEOMETRY,
    SPECTRAL_INDEX,
    SAR_BACKSCATTER,
    CHANGE_INDEX_DIFF,
    MODALITY_RELIABILITY,
    PHYSICS_VERIFIER,
    RS_VLM_VQA,
    TEXT_GROUNDING,
    CHANGE_ENCODER,
    VLM_CHANGE_QA,
    OPTICAL_SAR_FUSION,
)


def register_builtin_tools(registry=REGISTRY) -> None:
    """Register every built-in tool, ignoring ones already present."""
    for spec in BUILTIN_TOOLS:
        if registry.get(spec.name) is None:
            registry.register(spec)


register_builtin_tools()

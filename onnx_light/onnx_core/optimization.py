"""Graph-pattern optimization with standard and Python-defined patterns."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from ..onnx_lib import GraphProto, ModelProto
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]
from ..onnx_py import _onnxpypatterns as _patterns  # type: ignore[attr-defined]
from .graph_builder import GraphBuilder, SchemaLookup, _default_schema_lookup

PatternOptimization: TypeAlias = _C.PatternOptimization
MatchResult: TypeAlias = _C.MatchResult
LocalRewriting: TypeAlias = _C.LocalRewriting
OptimizationReport: TypeAlias = _C.OptimizationReport

CastPattern: TypeAlias = _patterns.CastPattern
CastCastPattern: TypeAlias = _patterns.CastCastPattern
CastCastBinaryPattern: TypeAlias = _patterns.CastCastBinaryPattern
CastOpCastPattern: TypeAlias = _patterns.CastOpCastPattern
ClipClipPattern: TypeAlias = _patterns.ClipClipPattern
ConstantToInitializerPattern: TypeAlias = _patterns.ConstantToInitializerPattern
ConvBiasNullPattern: TypeAlias = _patterns.ConvBiasNullPattern
DropoutPattern: TypeAlias = _patterns.DropoutPattern
IdentityPattern: TypeAlias = _patterns.IdentityPattern
NotNotPattern: TypeAlias = _patterns.NotNotPattern
PadConvPattern: TypeAlias = _patterns.PadConvPattern
SplitConcatPattern: TypeAlias = _patterns.SplitConcatPattern
GathersSplitPattern: TypeAlias = _patterns.GathersSplitPattern
SlicesSplitPattern: TypeAlias = _patterns.SlicesSplitPattern
ConcatEmptyPattern: TypeAlias = _patterns.ConcatEmptyPattern
ConcatGatherPattern: TypeAlias = _patterns.ConcatGatherPattern
ConcatTwiceUnaryPattern: TypeAlias = _patterns.ConcatTwiceUnaryPattern
GatherConcatPattern: TypeAlias = _patterns.GatherConcatPattern
GatherGatherPattern: TypeAlias = _patterns.GatherGatherPattern
GatherShapePattern: TypeAlias = _patterns.GatherShapePattern
SliceSlicePattern: TypeAlias = _patterns.SliceSlicePattern
SequenceConstructAtPattern: TypeAlias = _patterns.SequenceConstructAtPattern
SplitToSequenceSequenceAtPattern: TypeAlias = _patterns.SplitToSequenceSequenceAtPattern
NotWherePattern: TypeAlias = _patterns.NotWherePattern
UnsqueezeEqualPattern: TypeAlias = _patterns.UnsqueezeEqualPattern
WhereAddPattern: TypeAlias = _patterns.WhereAddPattern
ConcatReshapePattern: TypeAlias = _patterns.ConcatReshapePattern
ReshapePattern: TypeAlias = _patterns.ReshapePattern
ReduceReshapePattern: TypeAlias = _patterns.ReduceReshapePattern
Reshape2Of3Pattern: TypeAlias = _patterns.Reshape2Of3Pattern
ReshapeReshapeBinaryPattern: TypeAlias = _patterns.ReshapeReshapeBinaryPattern
ReshapeReshapePattern: TypeAlias = _patterns.ReshapeReshapePattern
ReshapeSqueezePattern: TypeAlias = _patterns.ReshapeSqueezePattern
ShapeBasedEditDistanceReshapePattern: TypeAlias = _patterns.ShapeBasedEditDistanceReshapePattern
ShapeBasedReshapeIsSqueezePattern: TypeAlias = _patterns.ShapeBasedReshapeIsSqueezePattern
ShapedBasedReshapePattern: TypeAlias = _patterns.ShapedBasedReshapePattern
StaticConcatReshapePattern: TypeAlias = _patterns.StaticConcatReshapePattern
UnsqueezeOrSqueezeReshapePattern: TypeAlias = _patterns.UnsqueezeOrSqueezeReshapePattern
UnsqueezeReshapePattern: TypeAlias = _patterns.UnsqueezeReshapePattern
MulUnsqueezeUnsqueezePattern: TypeAlias = _patterns.MulUnsqueezeUnsqueezePattern
SqueezeAddPattern: TypeAlias = _patterns.SqueezeAddPattern
SqueezeBinaryUnsqueezePattern: TypeAlias = _patterns.SqueezeBinaryUnsqueezePattern
SwapUnsqueezeTransposePattern: TypeAlias = _patterns.SwapUnsqueezeTransposePattern
TransposeEqualReshapePattern: TypeAlias = _patterns.TransposeEqualReshapePattern
TransposeReshapeTransposePattern: TypeAlias = _patterns.TransposeReshapeTransposePattern
MulMulMulScalarPattern: TypeAlias = _patterns.MulMulMulScalarPattern
SwitchOrderBinaryPattern: TypeAlias = _patterns.SwitchOrderBinaryPattern
SwapRangeAddScalarPattern: TypeAlias = _patterns.SwapRangeAddScalarPattern
ReduceArgTopKPattern: TypeAlias = _patterns.ReduceArgTopKPattern
ReduceSumNormalizePattern: TypeAlias = _patterns.ReduceSumNormalizePattern
Sub1MulPattern: TypeAlias = _patterns.Sub1MulPattern
SwapUnaryPattern: TypeAlias = _patterns.SwapUnaryPattern
SameChildrenPattern: TypeAlias = _patterns.SameChildrenPattern
SameChildrenFromInputPattern: TypeAlias = _patterns.SameChildrenFromInputPattern
ShapeBasedIdentityPattern: TypeAlias = _patterns.ShapeBasedIdentityPattern
ShapeBasedSameChildrenPattern: TypeAlias = _patterns.ShapeBasedSameChildrenPattern
ShapeBasedShapeShapeAddPattern: TypeAlias = _patterns.ShapeBasedShapeShapeAddPattern
RotaryEmbeddingPattern: TypeAlias = _patterns.RotaryEmbeddingPattern
RotaryConcatPartPattern: TypeAlias = _patterns.RotaryConcatPartPattern
FunctionCausalMaskPattern: TypeAlias = _patterns.FunctionCausalMaskPattern
FunctionCausalMaskMulAddPattern: TypeAlias = _patterns.FunctionCausalMaskMulAddPattern
FunctionCosSinCachePattern: TypeAlias = _patterns.FunctionCosSinCachePattern
FunctionHalfRotaryEmbeddingPattern: TypeAlias = _patterns.FunctionHalfRotaryEmbeddingPattern
FunctionAttentionPattern: TypeAlias = _patterns.FunctionAttentionPattern
FunctionAttentionGQAPattern: TypeAlias = _patterns.FunctionAttentionGQAPattern
AttentionGQAPattern: TypeAlias = _patterns.AttentionGQAPattern
GemmTransposePattern: TypeAlias = _patterns.GemmTransposePattern
MatMulAddPattern: TypeAlias = _patterns.MatMulAddPattern
MatMulReshape2Of3Pattern: TypeAlias = _patterns.MatMulReshape2Of3Pattern
MulMulMatMulPattern: TypeAlias = _patterns.MulMulMatMulPattern
ReshapeMatMulReshapePattern: TypeAlias = _patterns.ReshapeMatMulReshapePattern
ShapeBasedMatMulToMulPattern: TypeAlias = _patterns.ShapeBasedMatMulToMulPattern
SwitchReshapeActivationPattern: TypeAlias = _patterns.SwitchReshapeActivationPattern
TransposeMatMulPattern: TypeAlias = _patterns.TransposeMatMulPattern
TransposeReshapeMatMulPattern: TypeAlias = _patterns.TransposeReshapeMatMulPattern
BatchNormalizationPattern: TypeAlias = _patterns.BatchNormalizationPattern
BatchNormalizationTrainingPattern: TypeAlias = _patterns.BatchNormalizationTrainingPattern
CastLayerNormalizationCastPattern: TypeAlias = _patterns.CastLayerNormalizationCastPattern
LayerNormalizationPattern: TypeAlias = _patterns.LayerNormalizationPattern
LayerNormalizationScalePattern: TypeAlias = _patterns.LayerNormalizationScalePattern
RMSNormalizationPattern: TypeAlias = _patterns.RMSNormalizationPattern
RMSNormalizationMulPattern: TypeAlias = _patterns.RMSNormalizationMulPattern
GeluPattern: TypeAlias = _patterns.GeluPattern
LeakyReluPattern: TypeAlias = _patterns.LeakyReluPattern
MaxReluPattern: TypeAlias = _patterns.MaxReluPattern
SoftmaxCrossEntropyLossCastPattern: TypeAlias = _patterns.SoftmaxCrossEntropyLossCastPattern


def standard_pattern_names() -> list[str]:
    """Returns the standard ONNX pattern names."""
    return _patterns.registered_pattern_names()


def standard_patterns(names: Iterable[str] | None = None) -> list[PatternOptimization]:
    """Creates the selected standard ONNX patterns."""
    selected = standard_pattern_names() if names is None else list(names)
    return [_patterns.create_pattern(name) for name in selected]


_GLOBAL_PATTERNS: dict[str, PatternOptimization] = {
    pattern.name: pattern for pattern in standard_patterns()
}


def register_pattern(pattern: PatternOptimization) -> None:
    """Registers or replaces a process-global pattern."""
    name = str(pattern.name)
    if not name:
        raise ValueError("A registered pattern must have a non-empty name.")
    _GLOBAL_PATTERNS[name] = pattern


def unregister_pattern(name: str) -> bool:
    """Removes a global pattern and returns whether it existed."""
    return _GLOBAL_PATTERNS.pop(name, None) is not None


def clear_registered_patterns() -> None:
    """Removes every globally registered pattern, including standard patterns."""
    _GLOBAL_PATTERNS.clear()


def reset_registered_patterns() -> None:
    """Restores the global registry to the standard ONNX patterns."""
    _GLOBAL_PATTERNS.clear()
    _GLOBAL_PATTERNS.update((pattern.name, pattern) for pattern in standard_patterns())


def registered_patterns() -> tuple[PatternOptimization, ...]:
    """Returns global patterns in registration order."""
    return tuple(_GLOBAL_PATTERNS.values())


def registered_pattern_names() -> tuple[str, ...]:
    """Returns global pattern names in registration order."""
    return tuple(_GLOBAL_PATTERNS)


def render_rst_standard_patterns_table() -> str:
    """Renders the standard ONNX patterns as a reST ``list-table``.

    The table is generated from the patterns returned by
    :func:`standard_patterns`, so it stays in sync with the registered
    patterns without any manual maintenance.

    Returns:
        The ``list-table`` directive as a reST string.
    """
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 30 10 30 40",
        "",
        "    * - Class / registered name",
        "      - Priority",
        "      - Candidate roots",
        "      - Transformation",
    ]
    for pattern in sorted(standard_patterns(), key=lambda p: str(p.name)):
        class_name = type(pattern).__name__
        roots = ", ".join(f"``{op}``" for op in sorted(pattern.fast_op_type()))
        doc = (type(pattern).__doc__ or "").strip()
        summary = " ".join(doc.split("\n\n", 1)[0].split()) if doc else ""
        lines.extend(
            [
                f"    * - :class:`{class_name}` / ``{pattern.name}``",
                f"      - {pattern.priority}",
                f"      - {roots}",
                f"      - {summary}",
            ]
        )
    return "\n".join(lines) + "\n"


class GraphGraph(_C.GraphGraph):
    """Indexes a builder and runs globally or locally registered patterns.

    Global patterns are applied first. Patterns registered on ``builder`` then
    replace global patterns sharing their name, and ``patterns`` passed here
    have the highest precedence. Set ``use_global_patterns=False`` to start
    from an empty registry.
    """

    def __init__(
        self,
        builder: GraphBuilder,
        patterns: Iterable[str | PatternOptimization] | None = None,
        *,
        use_global_patterns: bool = True,
    ) -> None:
        selected = (
            {pattern.name: pattern for pattern in registered_patterns()}
            if use_global_patterns
            else {}
        )
        if hasattr(builder, "registered_patterns"):
            selected.update((pattern.name, pattern) for pattern in builder.registered_patterns())
        if patterns is not None:
            for pattern in patterns:
                resolved = (
                    _patterns.create_pattern(pattern) if isinstance(pattern, str) else pattern
                )
                selected[resolved.name] = resolved
        super().__init__(builder, list(selected.values()))


def replay(
    model: ModelProto,
    rewrites: Iterable[LocalRewriting],
    schema_lookup: SchemaLookup | None = _default_schema_lookup,
) -> GraphProto:
    """Replays captured rewrites and returns the resulting graph."""
    return _C.replay(model, list(rewrites), schema_lookup)


__all__ = [
    "AttentionGQAPattern",
    "BatchNormalizationPattern",
    "BatchNormalizationTrainingPattern",
    "CastCastBinaryPattern",
    "CastCastPattern",
    "CastLayerNormalizationCastPattern",
    "CastOpCastPattern",
    "CastPattern",
    "ClipClipPattern",
    "ConcatEmptyPattern",
    "ConcatGatherPattern",
    "ConcatReshapePattern",
    "ConcatTwiceUnaryPattern",
    "ConstantToInitializerPattern",
    "ConvBiasNullPattern",
    "DropoutPattern",
    "FunctionAttentionGQAPattern",
    "FunctionAttentionPattern",
    "FunctionCausalMaskMulAddPattern",
    "FunctionCausalMaskPattern",
    "FunctionCosSinCachePattern",
    "FunctionHalfRotaryEmbeddingPattern",
    "GatherConcatPattern",
    "GatherGatherPattern",
    "GatherShapePattern",
    "GathersSplitPattern",
    "GeluPattern",
    "GemmTransposePattern",
    "GraphBuilder",
    "GraphGraph",
    "IdentityPattern",
    "LayerNormalizationPattern",
    "LayerNormalizationScalePattern",
    "LeakyReluPattern",
    "LocalRewriting",
    "MatMulAddPattern",
    "MatMulReshape2Of3Pattern",
    "MatchResult",
    "MaxReluPattern",
    "MulMulMatMulPattern",
    "MulMulMulScalarPattern",
    "MulUnsqueezeUnsqueezePattern",
    "NotNotPattern",
    "NotWherePattern",
    "OptimizationReport",
    "PadConvPattern",
    "PatternOptimization",
    "RMSNormalizationMulPattern",
    "RMSNormalizationPattern",
    "ReduceArgTopKPattern",
    "ReduceReshapePattern",
    "ReduceSumNormalizePattern",
    "Reshape2Of3Pattern",
    "ReshapeMatMulReshapePattern",
    "ReshapePattern",
    "ReshapeReshapeBinaryPattern",
    "ReshapeReshapePattern",
    "ReshapeSqueezePattern",
    "RotaryConcatPartPattern",
    "RotaryEmbeddingPattern",
    "SameChildrenFromInputPattern",
    "SameChildrenPattern",
    "SequenceConstructAtPattern",
    "ShapeBasedEditDistanceReshapePattern",
    "ShapeBasedIdentityPattern",
    "ShapeBasedMatMulToMulPattern",
    "ShapeBasedReshapeIsSqueezePattern",
    "ShapeBasedSameChildrenPattern",
    "ShapeBasedShapeShapeAddPattern",
    "ShapedBasedReshapePattern",
    "SliceSlicePattern",
    "SlicesSplitPattern",
    "SoftmaxCrossEntropyLossCastPattern",
    "SplitConcatPattern",
    "SplitToSequenceSequenceAtPattern",
    "SqueezeAddPattern",
    "SqueezeBinaryUnsqueezePattern",
    "StaticConcatReshapePattern",
    "Sub1MulPattern",
    "SwapRangeAddScalarPattern",
    "SwapUnaryPattern",
    "SwapUnsqueezeTransposePattern",
    "SwitchOrderBinaryPattern",
    "SwitchReshapeActivationPattern",
    "TransposeEqualReshapePattern",
    "TransposeMatMulPattern",
    "TransposeReshapeMatMulPattern",
    "TransposeReshapeTransposePattern",
    "UnsqueezeEqualPattern",
    "UnsqueezeOrSqueezeReshapePattern",
    "UnsqueezeReshapePattern",
    "WhereAddPattern",
    "clear_registered_patterns",
    "register_pattern",
    "registered_pattern_names",
    "registered_patterns",
    "render_rst_standard_patterns_table",
    "replay",
    "reset_registered_patterns",
    "standard_pattern_names",
    "standard_patterns",
    "unregister_pattern",
]

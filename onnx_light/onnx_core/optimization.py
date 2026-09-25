"""Graph-pattern optimization with standard and Python-defined patterns."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Literal, TypeAlias

from ..onnx_lib import GraphProto, ModelProto
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]
from ..onnx_py import _onnxpypatterns as _patterns  # type: ignore[attr-defined]
from .graph_builder import GraphBuilder, SchemaLookup, _default_schema_lookup
from .shape_inference import Device

PatternOptimization: TypeAlias = _C.PatternOptimization
MatchResult: TypeAlias = _C.MatchResult
LocalRewriting: TypeAlias = _C.LocalRewriting
OptimizationReport: TypeAlias = _C.OptimizationReport

CastPattern: TypeAlias = _patterns.CastPattern
CastCastPattern: TypeAlias = _patterns.CastCastPattern
CastCastBinaryPattern: TypeAlias = _patterns.CastCastBinaryPattern
CastOpCastPattern: TypeAlias = _patterns.CastOpCastPattern
ClipClipPattern: TypeAlias = _patterns.ClipClipPattern
ReluClipFusionPattern: TypeAlias = _patterns.ReluClipFusionPattern
ConstantToInitializerPattern: TypeAlias = _patterns.ConstantToInitializerPattern
ConvBiasNullPattern: TypeAlias = _patterns.ConvBiasNullPattern
ConvAddFusionPattern: TypeAlias = _patterns.ConvAddFusionPattern
ConvMulFusionPattern: TypeAlias = _patterns.ConvMulFusionPattern
ConvBatchNormalizationFusionPattern: TypeAlias = _patterns.ConvBatchNormalizationFusionPattern
DropoutPattern: TypeAlias = _patterns.DropoutPattern
IdentityPattern: TypeAlias = _patterns.IdentityPattern
NotNotPattern: TypeAlias = _patterns.NotNotPattern
PadConvPattern: TypeAlias = _patterns.PadConvPattern
PadPadFusionPattern: TypeAlias = _patterns.PadPadFusionPattern
SplitConcatPattern: TypeAlias = _patterns.SplitConcatPattern
GathersSplitPattern: TypeAlias = _patterns.GathersSplitPattern
SlicesSplitPattern: TypeAlias = _patterns.SlicesSplitPattern
ConcatEmptyPattern: TypeAlias = _patterns.ConcatEmptyPattern
ConcatSliceEliminationPattern: TypeAlias = _patterns.ConcatSliceEliminationPattern
ConcatGatherPattern: TypeAlias = _patterns.ConcatGatherPattern
ConcatTwiceUnaryPattern: TypeAlias = _patterns.ConcatTwiceUnaryPattern
GatherConcatPattern: TypeAlias = _patterns.GatherConcatPattern
GatherGatherPattern: TypeAlias = _patterns.GatherGatherPattern
GatherSliceToSplitPattern: TypeAlias = _patterns.GatherSliceToSplitPattern
GatherToSlicePattern: TypeAlias = _patterns.GatherToSlicePattern
GatherShapePattern: TypeAlias = _patterns.GatherShapePattern
GatherUpstreamPropagationPattern: TypeAlias = _patterns.GatherUpstreamPropagationPattern
PreShapeNodeEliminationPattern: TypeAlias = _patterns.PreShapeNodeEliminationPattern
SliceSlicePattern: TypeAlias = _patterns.SliceSlicePattern
SliceEliminationPattern: TypeAlias = _patterns.SliceEliminationPattern
SliceConcatToSpaceToDepthPattern: TypeAlias = _patterns.SliceConcatToSpaceToDepthPattern
SequenceConstructAtPattern: TypeAlias = _patterns.SequenceConstructAtPattern
SplitToSequenceSequenceAtPattern: TypeAlias = _patterns.SplitToSequenceSequenceAtPattern
NotWherePattern: TypeAlias = _patterns.NotWherePattern
UnsqueezeEqualPattern: TypeAlias = _patterns.UnsqueezeEqualPattern
WhereAddPattern: TypeAlias = _patterns.WhereAddPattern
ExpandPattern: TypeAlias = _patterns.ExpandPattern
ExpandBroadcastPattern: TypeAlias = _patterns.ExpandBroadcastPattern
ShapeBasedConcatExpandPattern: TypeAlias = _patterns.ShapeBasedConcatExpandPattern
ShapeBasedExpandBroadcastPattern: TypeAlias = _patterns.ShapeBasedExpandBroadcastPattern
ShapeBasedExpandBroadcastMatMulPattern: TypeAlias = (
    _patterns.ShapeBasedExpandBroadcastMatMulPattern
)
ShapeBasedStaticExpandPattern: TypeAlias = _patterns.ShapeBasedStaticExpandPattern
ShapeBasedExpandSwapPattern: TypeAlias = _patterns.ShapeBasedExpandSwapPattern
ShapeBasedExpandCastWhereSwapPattern: TypeAlias = _patterns.ShapeBasedExpandCastWhereSwapPattern
ExpandSwapPattern: TypeAlias = _patterns.ExpandSwapPattern
SwapExpandUnsqueezePattern: TypeAlias = _patterns.SwapExpandUnsqueezePattern
ExpandUnsqueezeExpandPattern: TypeAlias = _patterns.ExpandUnsqueezeExpandPattern
SwapExpandReshapePattern: TypeAlias = _patterns.SwapExpandReshapePattern
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
DivMulPattern: TypeAlias = _patterns.DivMulPattern
STFTFusionPattern: TypeAlias = _patterns.STFTFusionPattern
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
LinearAttentionPattern: TypeAlias = _patterns.LinearAttentionPattern
FunctionAttentionGQAPattern: TypeAlias = _patterns.FunctionAttentionGQAPattern
AttentionGQAPattern: TypeAlias = _patterns.AttentionGQAPattern
GemmTransposePattern: TypeAlias = _patterns.GemmTransposePattern
GemmSumFusionPattern: TypeAlias = _patterns.GemmSumFusionPattern
MatMulAddPattern: TypeAlias = _patterns.MatMulAddPattern
MatMulBatchNormalizationFusionPattern: TypeAlias = _patterns.MatMulBatchNormalizationFusionPattern
MatMulReshape2Of3Pattern: TypeAlias = _patterns.MatMulReshape2Of3Pattern
MatMulScaleFusionPattern: TypeAlias = _patterns.MatMulScaleFusionPattern
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
TreeEnsemblePattern: TypeAlias = _patterns.TreeEnsemblePattern
LabelEncoderFusionPattern: TypeAlias = _patterns.LabelEncoderFusionPattern
TransposeTransposePattern: TypeAlias = _patterns.TransposeTransposePattern
TransposeGatherPattern: TypeAlias = _patterns.TransposeGatherPattern
UnsqueezeUnsqueezePattern: TypeAlias = _patterns.UnsqueezeUnsqueezePattern
SqueezeUnsqueezePattern: TypeAlias = _patterns.SqueezeUnsqueezePattern
ShapeTransposePattern: TypeAlias = _patterns.ShapeTransposePattern
UnsqueezeShapePattern: TypeAlias = _patterns.UnsqueezeShapePattern


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
    """Indexes a builder and selects the patterns used for rewriting.

    ``patterns=None`` selects all registered device-independent patterns.
    ``False`` selects none. A concrete ``Device`` includes independent patterns
    and patterns targeting that exact device, and sets ``builder.device``;
    a conflicting builder device raises ``ValueError``.

    A string or compiled regex selects registered names using ``fullmatch``.
    An iterable selects only its exact names and pattern instances. These
    explicit selections can include device-specific patterns without changing
    the builder device. Builder registrations override global registrations
    before selection; repeated explicit names keep the last instance.
    Disabling patterns does not disable the optimizer's cleanup passes.
    """

    def __init__(
        self,
        builder: GraphBuilder,
        patterns: (
            Iterable[str | PatternOptimization]
            | str
            | re.Pattern[str]
            | Device
            | Literal[False]
            | None
        ) = None,
    ) -> None:
        available = {pattern.name: pattern for pattern in registered_patterns()}
        if hasattr(builder, "registered_patterns"):
            available.update((pattern.name, pattern) for pattern in builder.registered_patterns())
        target_device = None
        if patterns is None:
            selected = {
                name: pattern
                for name, pattern in available.items()
                if pattern.device == Device.kUndefined
            }
        elif patterns is False:
            selected = {}
        elif isinstance(patterns, Device):
            target_device = patterns
            selected = {
                name: pattern
                for name, pattern in available.items()
                if pattern.device in (Device.kUndefined, patterns)
            }
        elif isinstance(patterns, (str, re.Pattern)):
            expression = re.compile(patterns)
            if not isinstance(expression.pattern, str):
                raise TypeError("The pattern selection regex must match strings, not bytes.")
            selected = {
                name: pattern
                for name, pattern in available.items()
                if expression.fullmatch(name) is not None
            }
        elif isinstance(patterns, Iterable):
            selected = {}
            for pattern in patterns:
                if isinstance(pattern, str):
                    resolved = (
                        available[pattern]
                        if pattern in available
                        else _patterns.create_pattern(pattern)
                    )
                elif isinstance(pattern, _C.PatternOptimization):
                    resolved = pattern
                else:
                    raise TypeError(
                        "A pattern list must contain names or PatternOptimization instances."
                    )
                selected[resolved.name] = resolved
        else:
            raise TypeError(
                "patterns must be None, False, a Device, a regex, or an iterable of patterns."
            )
        super().__init__(builder, list(selected.values()))
        if target_device is not None:
            self.set_target_device(target_device)


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
    "ConcatSliceEliminationPattern",
    "ConcatTwiceUnaryPattern",
    "ConstantToInitializerPattern",
    "ConvAddFusionPattern",
    "ConvBatchNormalizationFusionPattern",
    "ConvBiasNullPattern",
    "ConvMulFusionPattern",
    "DivMulPattern",
    "DropoutPattern",
    "ExpandBroadcastPattern",
    "ExpandPattern",
    "ExpandSwapPattern",
    "ExpandUnsqueezeExpandPattern",
    "FunctionAttentionGQAPattern",
    "FunctionAttentionPattern",
    "FunctionCausalMaskMulAddPattern",
    "FunctionCausalMaskPattern",
    "FunctionCosSinCachePattern",
    "FunctionHalfRotaryEmbeddingPattern",
    "GatherConcatPattern",
    "GatherGatherPattern",
    "GatherShapePattern",
    "GatherSliceToSplitPattern",
    "GatherToSlicePattern",
    "GatherUpstreamPropagationPattern",
    "GathersSplitPattern",
    "GeluPattern",
    "GemmSumFusionPattern",
    "GemmTransposePattern",
    "GraphBuilder",
    "GraphGraph",
    "IdentityPattern",
    "LabelEncoderFusionPattern",
    "LayerNormalizationPattern",
    "LayerNormalizationScalePattern",
    "LeakyReluPattern",
    "LinearAttentionPattern",
    "LocalRewriting",
    "MatMulAddPattern",
    "MatMulBatchNormalizationFusionPattern",
    "MatMulReshape2Of3Pattern",
    "MatMulScaleFusionPattern",
    "MatchResult",
    "MaxReluPattern",
    "MulMulMatMulPattern",
    "MulMulMulScalarPattern",
    "MulUnsqueezeUnsqueezePattern",
    "NotNotPattern",
    "NotWherePattern",
    "OptimizationReport",
    "PadConvPattern",
    "PadPadFusionPattern",
    "PatternOptimization",
    "PreShapeNodeEliminationPattern",
    "RMSNormalizationMulPattern",
    "RMSNormalizationPattern",
    "ReduceArgTopKPattern",
    "ReduceReshapePattern",
    "ReduceSumNormalizePattern",
    "ReluClipFusionPattern",
    "Reshape2Of3Pattern",
    "ReshapeMatMulReshapePattern",
    "ReshapePattern",
    "ReshapeReshapeBinaryPattern",
    "ReshapeReshapePattern",
    "ReshapeSqueezePattern",
    "RotaryConcatPartPattern",
    "RotaryEmbeddingPattern",
    "STFTFusionPattern",
    "SameChildrenFromInputPattern",
    "SameChildrenPattern",
    "SequenceConstructAtPattern",
    "ShapeBasedConcatExpandPattern",
    "ShapeBasedEditDistanceReshapePattern",
    "ShapeBasedExpandBroadcastMatMulPattern",
    "ShapeBasedExpandBroadcastPattern",
    "ShapeBasedExpandCastWhereSwapPattern",
    "ShapeBasedExpandSwapPattern",
    "ShapeBasedIdentityPattern",
    "ShapeBasedMatMulToMulPattern",
    "ShapeBasedReshapeIsSqueezePattern",
    "ShapeBasedSameChildrenPattern",
    "ShapeBasedShapeShapeAddPattern",
    "ShapeBasedStaticExpandPattern",
    "ShapeTransposePattern",
    "ShapedBasedReshapePattern",
    "SliceConcatToSpaceToDepthPattern",
    "SliceEliminationPattern",
    "SliceSlicePattern",
    "SlicesSplitPattern",
    "SoftmaxCrossEntropyLossCastPattern",
    "SplitConcatPattern",
    "SplitToSequenceSequenceAtPattern",
    "SqueezeAddPattern",
    "SqueezeBinaryUnsqueezePattern",
    "SqueezeUnsqueezePattern",
    "StaticConcatReshapePattern",
    "Sub1MulPattern",
    "SwapExpandReshapePattern",
    "SwapExpandUnsqueezePattern",
    "SwapRangeAddScalarPattern",
    "SwapUnaryPattern",
    "SwapUnsqueezeTransposePattern",
    "SwitchOrderBinaryPattern",
    "SwitchReshapeActivationPattern",
    "TransposeEqualReshapePattern",
    "TransposeGatherPattern",
    "TransposeMatMulPattern",
    "TransposeReshapeMatMulPattern",
    "TransposeReshapeTransposePattern",
    "TransposeTransposePattern",
    "TreeEnsemblePattern",
    "UnsqueezeEqualPattern",
    "UnsqueezeOrSqueezeReshapePattern",
    "UnsqueezeReshapePattern",
    "UnsqueezeShapePattern",
    "UnsqueezeUnsqueezePattern",
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

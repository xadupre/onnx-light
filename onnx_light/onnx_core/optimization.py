"""Graph-pattern optimization with standard and Python-defined patterns."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import re
from typing import Literal, TypeAlias, overload

from ..onnx_lib import GraphProto, LocalRewritingProto, ModelProto, RewritingProto
from ..onnx_py._onnxpycore import builder as _C  # type: ignore[attr-defined]
from ..onnx_py import _onnxpypatterns as _patterns  # type: ignore[attr-defined]
from .graph_builder import (
    ConstantFoldingOptions,
    GraphBuilder,
    SchemaLookup,
    _default_schema_lookup,
)
from .shape_inference import Device

PatternOptimization: TypeAlias = _C.PatternOptimization
RepeatedLayerPattern: TypeAlias = _patterns.RepeatedLayerPattern
MatchResult: TypeAlias = _C.MatchResult
OptimizationReport: TypeAlias = _C.OptimizationReport

CastPattern: TypeAlias = _patterns.CastPattern
CastCastPattern: TypeAlias = _patterns.CastCastPattern
CastCastBinaryPattern: TypeAlias = _patterns.CastCastBinaryPattern
CastOpCastPattern: TypeAlias = _patterns.CastOpCastPattern
ClipClipPattern: TypeAlias = _patterns.ClipClipPattern
ReluClipFusionPattern: TypeAlias = _patterns.ReluClipFusionPattern
ConstantToInitializerPattern: TypeAlias = _patterns.ConstantToInitializerPattern
InitializerUnsqueezeCastPattern: TypeAlias = _patterns.InitializerUnsqueezeCastPattern
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
GroupedMatMulPattern: TypeAlias = _patterns.GroupedMatMulPattern
GroupedMatMulConcatPattern: TypeAlias = _patterns.GroupedMatMulConcatPattern
GroupedMatMulLoopPattern: TypeAlias = _patterns.GroupedMatMulLoopPattern
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


def standard_pattern_names(*, include_opt_in: bool = False) -> list[str]:
    """Returns standard pattern names, excluding opt-in patterns unless requested."""
    return _patterns.registered_pattern_names(include_opt_in=include_opt_in)


def standard_pattern_set_names() -> list[str]:
    """Returns the names of registered standard pattern sets."""
    return _patterns.registered_pattern_set_names()


def standard_pattern_sets() -> dict[str, tuple[str, ...]]:
    """Returns named, overlapping C++-registered groups of pattern names."""
    return {
        name: tuple(pattern_names)
        for name, pattern_names in _patterns.registered_pattern_sets().items()
    }


def register_pattern_set(name: str, pattern_names: Iterable[str]) -> None:
    """Creates a named pattern set in the C++ pattern registry."""
    if isinstance(pattern_names, str):
        raise TypeError("Pattern names must be an iterable of names, not a string.")
    _patterns.register_pattern_set(name, list(pattern_names))


def extend_pattern_set(name: str, pattern_names: Iterable[str]) -> None:
    """Adds names to an existing pattern set in the C++ registry."""
    if isinstance(pattern_names, str):
        raise TypeError("Pattern names must be an iterable of names, not a string.")
    _patterns.extend_pattern_set(name, list(pattern_names))


def unregister_pattern_set(name: str) -> bool:
    """Removes a custom pattern set from the C++ registry."""
    return _patterns.unregister_pattern_set(name)


def standard_patterns(names: Iterable[str] | None = None) -> list[PatternOptimization]:
    """Creates the selected standard ONNX patterns."""
    selected = standard_pattern_names() if names is None else list(names)
    global_patterns = globals().get("_GLOBAL_PATTERNS", {})
    patterns = []
    for name in selected:
        if name not in _STANDARD_PATTERN_NAMES and name in global_patterns:
            patterns.append(global_patterns[name])
        else:
            patterns.append(_patterns.create_pattern(name))
    return patterns


_STANDARD_PATTERN_NAMES = frozenset(standard_pattern_names(include_opt_in=True))
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
    Pass ``remove_unused=False`` to :meth:`optimize` to disable only the unused-node
    cleanup pass; other cleanup passes may still remove nodes or initializers.
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

    @overload
    def optimize(
        self,
        max_iter: int = -1,
        *,
        report: Literal[False] = False,
        folding_options: ConstantFoldingOptions | None = None,
        remove_unused: bool = True,
        callback: Callable[[GraphGraph, MatchResult], bool | None] | None = None,
        parallel: bool = False,
    ) -> RewritingProto: ...

    @overload
    def optimize(
        self,
        max_iter: int = -1,
        *,
        report: Literal[True],
        folding_options: ConstantFoldingOptions | None = None,
        remove_unused: bool = True,
        callback: Callable[[GraphGraph, MatchResult], bool | None] | None = None,
        parallel: bool = False,
    ) -> tuple[RewritingProto, OptimizationReport]: ...

    @overload
    def optimize(
        self,
        max_iter: int = -1,
        *,
        report: bool,
        folding_options: ConstantFoldingOptions | None = None,
        remove_unused: bool = True,
        callback: Callable[[GraphGraph, MatchResult], bool | None] | None = None,
        parallel: bool = False,
    ) -> RewritingProto | tuple[RewritingProto, OptimizationReport]: ...

    def optimize(
        self,
        max_iter: int = -1,
        *,
        report: bool = False,
        folding_options: ConstantFoldingOptions | None = None,
        remove_unused: bool = True,
        callback: Callable[[GraphGraph, MatchResult], bool | None] | None = None,
        parallel: bool = False,
    ) -> RewritingProto | tuple[RewritingProto, OptimizationReport]:
        """Returns sequential proto edits and optionally their optimization report.

        Calls ``callback(graph, match)`` for each eligible, disjoint pattern match,
        including matches in nested subgraphs, before reserving its nodes.
        Returning ``False`` rejects that match without blocking other patterns;
        ``True`` or ``None`` accepts it, so reporting-only callbacks need no return.
        Cleanup passes are not intercepted. Matches may be reconsidered in later
        iterations and accepted matches are revalidated before application.

        The callback must not mutate the graph, patterns, or match, or retain
        references to the graph or matched nodes beyond the call. Exceptions
        propagate to the caller.

        ``parallel=True`` matches explicitly opted-in native patterns concurrently.
        Other patterns, including Python-defined patterns requiring the GIL,
        remain sequential.
        The optimization callback always runs sequentially on the calling thread.
        """
        return super().optimize(
            max_iter,
            report=report,
            folding_options=(
                ConstantFoldingOptions() if folding_options is None else folding_options
            ),
            remove_unused=remove_unused,
            callback=callback,
            parallel=parallel,
        )


def apply_rewritings(
    model: ModelProto,
    rewrites: RewritingProto,
    schema_lookup: SchemaLookup | None = _default_schema_lookup,
) -> GraphProto:
    """Applies captured rewrites and returns the resulting graph."""
    return _C.apply_rewritings(model, rewrites, schema_lookup)


def apply_rewritings_to_model(
    model: ModelProto,
    rewrites: RewritingProto,
    schema_lookup: SchemaLookup | None = _default_schema_lookup,
) -> ModelProto:
    """Applies captured rewrites to a model, preserving functions and imports."""
    return _C.apply_rewritings_to_model(model, rewrites, schema_lookup)


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
    "GroupedMatMulConcatPattern",
    "GroupedMatMulLoopPattern",
    "GroupedMatMulPattern",
    "IdentityPattern",
    "InitializerUnsqueezeCastPattern",
    "LabelEncoderFusionPattern",
    "LayerNormalizationPattern",
    "LayerNormalizationScalePattern",
    "LeakyReluPattern",
    "LinearAttentionPattern",
    "LocalRewritingProto",
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
    "RepeatedLayerPattern",
    "Reshape2Of3Pattern",
    "ReshapeMatMulReshapePattern",
    "ReshapePattern",
    "ReshapeReshapeBinaryPattern",
    "ReshapeReshapePattern",
    "ReshapeSqueezePattern",
    "RewritingProto",
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
    "apply_rewritings",
    "apply_rewritings_to_model",
    "clear_registered_patterns",
    "extend_pattern_set",
    "register_pattern",
    "register_pattern_set",
    "registered_pattern_names",
    "registered_patterns",
    "render_rst_standard_patterns_table",
    "reset_registered_patterns",
    "standard_pattern_names",
    "standard_pattern_set_names",
    "standard_pattern_sets",
    "standard_patterns",
    "unregister_pattern",
    "unregister_pattern_set",
]

"""Comparison of ONNX :class:`OpSchema` and ``onnx_light`` ``LightOpSchema``.

This module produces a structured comparison between the operator schemas
exposed by the reference :epkg:`onnx` package (``onnx.defs.OpSchema``) and the
lightweight schemas exposed by ``onnx_light`` (``LightOpSchema`` from the C++
``onnx_op`` extension).

For every operator known to either side it reports:

* whether the operator is described by an ``OpSchema`` (``onnx``);
* whether the operator is described by a ``LightOpSchema`` (``onnx_light``);
* whether a shape-inference function is registered on the ``onnx`` side
  (``OpSchema.has_type_and_shape_inference_function``);
* whether a shape-inference function is registered on the ``onnx_light`` side
  (in the ``onnx_optim`` C++ library, see
  :cpp:func:`ComputeShapeNode`);
* how many backend test cases exercise the operator in each package (a
  test case is attributed to the operator whose lowercased or
  ``snake_case`` name matches its ``test_<op>(_<variant>)*`` data-folder
  name — the convention used by ``onnx/backend/test/data/node`` and
  mirrored by ``onnx_light``'s ``test_cc_<op>`` registry).

The :func:`compute_schema_comparison` function returns a
:class:`SchemaComparison` describing the rows. The :func:`render_rst_table`
helper turns the comparison into a Sphinx ``list-table`` ready to be embedded
in a documentation page (see ``docs/design/schema_comparison.rst``).
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable

# Operators whose output shapes can be inferred by ``onnx_optim`` (see
# ``onnx_light/onnx_optim/shapes/dispatch_table.cc``). The list is small on
# purpose: ``onnx_optim`` only implements a handful of operators today. Keep
# this set in sync with the dispatch table in ``dispatch_table.cc``.
ONNX_OPTIM_SHAPE_INFERENCE_OPS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ai.onnx", "Abs"),
        ("ai.onnx", "Acos"),
        ("ai.onnx", "Acosh"),
        ("ai.onnx", "Add"),
        ("ai.onnx", "AffineGrid"),
        ("ai.onnx", "And"),
        ("ai.onnx", "ArgMax"),
        ("ai.onnx", "ArgMin"),
        ("ai.onnx", "Asin"),
        ("ai.onnx", "Asinh"),
        ("ai.onnx", "Atan"),
        ("ai.onnx", "Atanh"),
        ("ai.onnx", "Attention"),
        ("ai.onnx", "AveragePool"),
        ("ai.onnx", "BatchNormalization"),
        ("ai.onnx", "Bernoulli"),
        ("ai.onnx", "BitCast"),
        ("ai.onnx", "BitShift"),
        ("ai.onnx", "BitwiseAnd"),
        ("ai.onnx", "BitwiseNot"),
        ("ai.onnx", "BitwiseOr"),
        ("ai.onnx", "BitwiseXor"),
        ("ai.onnx", "BlackmanWindow"),
        ("ai.onnx", "Cast"),
        ("ai.onnx", "CastLike"),
        ("ai.onnx", "Ceil"),
        ("ai.onnx", "Celu"),
        ("ai.onnx", "CenterCropPad"),
        ("ai.onnx", "Clip"),
        ("ai.onnx", "Col2Im"),
        ("ai.onnx", "Compress"),
        ("ai.onnx", "Concat"),
        ("ai.onnx", "ConcatFromSequence"),
        ("ai.onnx", "Constant"),
        ("ai.onnx", "ConstantOfShape"),
        ("ai.onnx", "Cos"),
        ("ai.onnx", "Cosh"),
        ("ai.onnx", "CumProd"),
        ("ai.onnx", "CumSum"),
        ("ai.onnx", "DFT"),
        ("ai.onnx", "DepthToSpace"),
        ("ai.onnx", "DeformConv"),
        ("ai.onnx", "Det"),
        ("ai.onnx", "Conv"),
        ("ai.onnx", "ConvInteger"),
        ("ai.onnx", "ConvTranspose"),
        ("ai.onnx", "Sin"),
        ("ai.onnx", "Sinh"),
        ("ai.onnx", "DequantizeLinear"),
        ("ai.onnx", "Div"),
        ("ai.onnx", "Dropout"),
        ("ai.onnx", "DynamicQuantizeLinear"),
        ("ai.onnx", "Einsum"),
        ("ai.onnx", "Elu"),
        ("ai.onnx", "Equal"),
        ("ai.onnx", "Erf"),
        ("ai.onnx", "Exp"),
        ("ai.onnx", "Expand"),
        ("ai.onnx", "EyeLike"),
        ("ai.onnx", "Flatten"),
        ("ai.onnx", "Floor"),
        ("ai.onnx", "Gather"),
        ("ai.onnx", "GatherElements"),
        ("ai.onnx", "GatherND"),
        ("ai.onnx", "Gelu"),
        ("ai.onnx", "Greater"),
        ("ai.onnx", "GreaterOrEqual"),
        ("ai.onnx", "GridSample"),
        ("ai.onnx", "GlobalAveragePool"),
        ("ai.onnx", "GlobalLpPool"),
        ("ai.onnx", "GlobalMaxPool"),
        ("ai.onnx", "GroupNormalization"),
        ("ai.onnx", "GRU"),
        ("ai.onnx", "Gemm"),
        ("ai.onnx", "HammingWindow"),
        ("ai.onnx", "HannWindow"),
        ("ai.onnx", "HardSigmoid"),
        ("ai.onnx", "HardSwish"),
        ("ai.onnx", "Hardmax"),
        ("ai.onnx", "If"),
        ("ai.onnx", "Identity"),
        ("ai.onnx", "ImageDecoder"),
        ("ai.onnx", "InstanceNormalization"),
        ("ai.onnx", "IsInf"),
        ("ai.onnx", "IsNaN"),
        ("ai.onnx", "LayerNormalization"),
        ("ai.onnx", "LeakyRelu"),
        ("ai.onnx", "Less"),
        ("ai.onnx", "Loop"),
        ("ai.onnx", "Log"),
        ("ai.onnx", "LogSoftmax"),
        ("ai.onnx", "LRN"),
        ("ai.onnx", "LpNormalization"),
        ("ai.onnx", "LpPool"),
        ("ai.onnx", "LSTM"),
        ("ai.onnx", "MatMul"),
        ("ai.onnx", "MatMulInteger"),
        ("ai.onnx", "Max"),
        ("ai.onnx", "MaxPool"),
        ("ai.onnx", "MaxRoiPool"),
        ("ai.onnx", "MaxUnpool"),
        ("ai.onnx", "Mean"),
        ("ai.onnx", "MeanVarianceNormalization"),
        ("ai.onnx", "MelWeightMatrix"),
        ("ai.onnx", "Min"),
        ("ai.onnx", "Mish"),
        ("ai.onnx", "NegativeLogLikelihoodLoss"),
        ("ai.onnx", "Mod"),
        ("ai.onnx", "Mul"),
        ("ai.onnx", "Multinomial"),
        ("ai.onnx", "Neg"),
        ("ai.onnx", "NonMaxSuppression"),
        ("ai.onnx", "NonZero"),
        ("ai.onnx", "Not"),
        ("ai.onnx", "Optional"),
        ("ai.onnx", "OptionalGetElement"),
        ("ai.onnx", "OptionalHasElement"),
        ("ai.onnx", "Or"),
        ("ai.onnx", "OneHot"),
        ("ai.onnx", "PRelu"),
        ("ai.onnx", "Pad"),
        ("ai.onnx", "Pow"),
        ("ai.onnx", "QLinearConv"),
        ("ai.onnx", "QLinearMatMul"),
        ("ai.onnx", "QuantizeLinear"),
        ("ai.onnx", "RandomNormal"),
        ("ai.onnx", "RandomNormalLike"),
        ("ai.onnx", "RandomUniform"),
        ("ai.onnx", "RandomUniformLike"),
        ("ai.onnx", "Range"),
        ("ai.onnx", "ReduceL1"),
        ("ai.onnx", "ReduceL2"),
        ("ai.onnx", "ReduceLogSum"),
        ("ai.onnx", "ReduceLogSumExp"),
        ("ai.onnx", "ReduceMax"),
        ("ai.onnx", "ReduceMean"),
        ("ai.onnx", "ReduceMin"),
        ("ai.onnx", "ReduceProd"),
        ("ai.onnx", "ReduceSum"),
        ("ai.onnx", "ReduceSumSquare"),
        ("ai.onnx", "Relu"),
        ("ai.onnx", "Reciprocal"),
        ("ai.onnx", "Reshape"),
        ("ai.onnx", "ReverseSequence"),
        ("ai.onnx", "Resize"),
        ("ai.onnx", "Round"),
        ("ai.onnx", "Scatter"),
        ("ai.onnx", "ScatterElements"),
        ("ai.onnx", "ScatterND"),
        ("ai.onnx", "Slice"),
        ("ai.onnx", "Split"),
        ("ai.onnx", "Squeeze"),
        ("ai.onnx", "Sub"),
        ("ai.onnx", "Tile"),
        ("ai.onnx", "Transpose"),
        ("ai.onnx", "Trilu"),
        ("ai.onnx", "Unsqueeze"),
        ("ai.onnx", "Unique"),
        ("ai.onnx", "Upsample"),
        ("ai.onnx", "RNN"),
        ("ai.onnx", "RoiAlign"),
        ("ai.onnx", "RMSNormalization"),
        ("ai.onnx", "RotaryEmbedding"),
        ("ai.onnx", "Scan"),
        ("ai.onnx", "Selu"),
        ("ai.onnx", "SequenceConstruct"),
        ("ai.onnx", "SequenceErase"),
        ("ai.onnx", "SequenceAt"),
        ("ai.onnx", "SequenceEmpty"),
        ("ai.onnx", "SequenceInsert"),
        ("ai.onnx", "SequenceLength"),
        ("ai.onnx", "SequenceMap"),
        ("ai.onnx", "SplitToSequence"),
        ("ai.onnx", "Shape"),
        ("ai.onnx", "Shrink"),
        ("ai.onnx", "Size"),
        ("ai.onnx", "Sigmoid"),
        ("ai.onnx", "Softmax"),
        ("ai.onnx", "SoftmaxCrossEntropyLoss"),
        ("ai.onnx", "Softplus"),
        ("ai.onnx", "Softsign"),
        ("ai.onnx", "SpaceToDepth"),
        ("ai.onnx", "Sqrt"),
        ("ai.onnx", "STFT"),
        ("ai.onnx", "Sum"),
        ("ai.onnx", "Swish"),
        ("ai.onnx", "StringConcat"),
        ("ai.onnx", "StringSplit"),
        ("ai.onnx", "StringNormalizer"),
        ("ai.onnx", "RegexFullMatch"),
        ("ai.onnx", "TfIdfVectorizer"),
        ("ai.onnx", "Tan"),
        ("ai.onnx", "Tanh"),
        ("ai.onnx", "TensorScatter"),
        ("ai.onnx", "ThresholdedRelu"),
        ("ai.onnx", "TopK"),
        ("ai.onnx", "Where"),
        ("ai.onnx", "Xor"),
        ("ai.onnx.ml", "ArrayFeatureExtractor"),
        ("ai.onnx.ml", "Binarizer"),
        ("ai.onnx.ml", "CastMap"),
        ("ai.onnx.ml", "CategoryMapper"),
        ("ai.onnx.ml", "DictVectorizer"),
        ("ai.onnx.ml", "FeatureVectorizer"),
        ("ai.onnx.ml", "Imputer"),
        ("ai.onnx.ml", "LabelEncoder"),
        ("ai.onnx.ml", "LinearClassifier"),
        ("ai.onnx.ml", "LinearRegressor"),
        ("ai.onnx.ml", "Normalizer"),
        ("ai.onnx.ml", "OneHotEncoder"),
        ("ai.onnx.ml", "SVMClassifier"),
        ("ai.onnx.ml", "SVMRegressor"),
        ("ai.onnx.ml", "Scaler"),
        ("ai.onnx.ml", "TreeEnsemble"),
        ("ai.onnx.ml", "TreeEnsembleClassifier"),
        ("ai.onnx.ml", "TreeEnsembleRegressor"),
        ("ai.onnx.ml", "ZipMap"),
        ("ai.onnx.preview", "FlexAttention"),
        ("ai.onnx.preview.training", "Adagrad"),
        ("ai.onnx.preview.training", "Adam"),
        ("ai.onnx.preview.training", "Momentum"),
    }
)


@dataclass
class SchemaComparisonRow:
    """One row of :class:`SchemaComparison`, describing a single operator.

    :param domain: Operator domain (``ai.onnx`` for the default ONNX domain).
    :param name: Operator name.
    :param in_onnx: ``True`` when the operator has an ``OpSchema`` in
        :mod:`onnx.defs`.
    :param in_onnx_light: ``True`` when the operator has a ``LightOpSchema``
        in the ``onnx_op`` C++ extension.
    :param onnx_shape_inference: ``True`` when ``onnx`` registers a type and
        shape inference function for the operator.
    :param onnx_light_shape_inference: ``True`` when ``onnx_optim`` registers
        a shape-inference dispatch entry for the operator.
    :param onnx_backend_tests: Number of node backend tests in :mod:`onnx`
        attributed to the operator (one count per
        ``onnx/backend/test/data/node/test_<op>(_<variant>)*`` subfolder
        whose name starts with the operator's lowercased or ``snake_case``
        form).
    :param onnx_light_backend_tests: Number of node backend tests collected by
        :func:`onnx_light.backend.test.case.base.collect_test_case` attributed
        to the operator by the same name-prefix convention (with
        ``test_cc_`` also stripped for cases registered by
        ``lib_onnx_backend_test``).
    """

    domain: str
    name: str
    in_onnx: bool = False
    in_onnx_light: bool = False
    onnx_shape_inference: bool = False
    onnx_light_shape_inference: bool = False
    onnx_backend_tests: int = 0
    onnx_light_backend_tests: int = 0


@dataclass
class SchemaComparison:
    """Aggregated comparison between ``onnx`` and ``onnx_light`` operators.

    :param rows: One :class:`SchemaComparisonRow` per ``(domain, op_name)``
        known to either side, sorted by ``(domain, name)``.
    :param onnx_available: ``True`` when the reference :mod:`onnx` package is
        importable and was used to populate the ``onnx``-side columns.
    """

    rows: list[SchemaComparisonRow] = field(default_factory=list)
    onnx_available: bool = True

    @property
    def total_onnx(self) -> int:
        """Number of operators with an ``OpSchema``."""
        return sum(1 for r in self.rows if r.in_onnx)

    @property
    def total_onnx_light(self) -> int:
        """Number of operators with a ``LightOpSchema``."""
        return sum(1 for r in self.rows if r.in_onnx_light)

    @property
    def total_onnx_shape_inference(self) -> int:
        """Operators with shape inference on the ``onnx`` side."""
        return sum(1 for r in self.rows if r.onnx_shape_inference)

    @property
    def total_onnx_light_shape_inference(self) -> int:
        """Operators with shape inference on the ``onnx_light`` (``onnx_optim``) side."""
        return sum(1 for r in self.rows if r.onnx_light_shape_inference)

    @property
    def total_onnx_backend_tests(self) -> int:
        """Total number of ``onnx`` node backend tests counted."""
        return sum(r.onnx_backend_tests for r in self.rows)

    @property
    def total_onnx_light_backend_tests(self) -> int:
        """Total number of ``onnx_light`` node backend tests counted."""
        return sum(r.onnx_light_backend_tests for r in self.rows)


def _light_schemas_latest() -> dict[tuple[str, str], Any]:
    """Returns ``{(domain, name): schema}`` keeping only the latest version."""
    from .onnx_py._onnxpy import onnx_op as _op  # type: ignore[attr-defined]

    latest: dict[tuple[str, str], Any] = {}
    for s in _op.GetAllOnnxOpSchemasWithHistory():
        key = (s.domain, s.name)
        if key not in latest or s.since_version > latest[key].since_version:
            latest[key] = s
    return latest


_TEST_NAME_PREFIXES: tuple[str, ...] = ("test_cc_", "test_")


def _op_name_forms(op_name: str) -> tuple[str, ...]:
    """Returns the lowercase forms of *op_name* used in backend-test names.

    Two forms are produced because the upstream
    ``onnx/backend/test/data/node`` directory uses both styles:

    * ``lower`` — the operator name lowercased with separators removed
      (matches names like ``test_qlinearconv``, ``test_abs``).
    * ``snake`` — the operator name converted to ``snake_case``
      (matches names like ``test_reduce_l1_*``, ``test_argmax_*``).
    """
    lower = op_name.lower()
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", op_name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    snake = s.lower()
    if snake == lower:
        return (lower,)
    return (lower, snake)


def _build_op_form_index(op_keys: Iterable[tuple[str, str]]) -> list[tuple[str, tuple[str, str]]]:
    """Builds a longest-first list of ``(form, key)`` for test-name matching.

    Each operator contributes one or two entries (see :func:`_op_name_forms`),
    plus a domain-prefixed variant for non-default domains (the upstream
    ``ai.onnx.ml`` test data uses names like ``test_ai_onnx_ml_binarizer``).
    Sorting by descending form length ensures the longest specific match is
    selected first (so ``reduce_l1`` beats ``reduce`` for
    ``test_reduce_l1_*``).
    """
    entries: list[tuple[str, tuple[str, str]]] = []
    for domain, name in op_keys:
        forms = _op_name_forms(name)
        for form in forms:
            entries.append((form, (domain, name)))
            if domain and domain != "ai.onnx":
                domain_prefix = domain.replace(".", "_") + "_"
                entries.append((domain_prefix + form, (domain, name)))
    # Sort by descending form length, with operator name as tiebreaker for
    # stability.
    entries.sort(key=lambda e: (-len(e[0]), e[1][1]))
    return entries


def _attribute_test_name(
    name: str, sorted_op_forms: list[tuple[str, tuple[str, str]]]
) -> tuple[str, str] | None:
    """Attributes a backend-test *name* to an operator key by prefix match.

    Strips ``test_cc_`` (onnx_light convention) or ``test_`` (upstream onnx
    convention) and looks for the longest operator-name form (see
    :func:`_op_name_forms`) that is either the whole remainder or a
    ``form_<variant>`` prefix. Returns ``None`` when no match is found.
    """
    rest: str | None = None
    for prefix in _TEST_NAME_PREFIXES:
        if name.startswith(prefix):
            rest = name[len(prefix) :]
            break
    if not rest:
        return None
    for form, key in sorted_op_forms:
        if rest == form or rest.startswith(form + "_"):
            return key
    return None


def _count_onnx_light_backend_tests(
    op_keys: Iterable[tuple[str, str]] | None = None,
) -> Counter[tuple[str, str]]:
    """Counts ``onnx_light`` node backend tests, attributing by test-case name.

    Each test case is attributed to the operator whose lowercased or
    ``snake_case`` name matches the test-case name as the longest prefix
    after stripping ``test_cc_`` or ``test_`` (mirroring the upstream
    ``onnx/backend/test/data/node`` subfolder naming convention). When a
    name does not match any known operator, the count falls back to the
    ``op_type`` of the first node of the model so unusual cases still
    contribute somewhere.
    """
    from .backend.test.case.base import collect_test_case

    if op_keys is None:
        op_keys = set(_light_schemas_latest())
    sorted_forms = _build_op_form_index(op_keys)

    counts: Counter[tuple[str, str]] = Counter()
    for name, tc in collect_test_case().items():
        key = _attribute_test_name(name, sorted_forms)
        if key is None:
            model = tc.model
            if model is None:
                continue
            nodes = list(model.graph.node)
            if not nodes:
                continue
            n = nodes[0]
            key = (n.domain or "ai.onnx", n.op_type)
        counts[key] += 1
    return counts


def _count_onnx_backend_tests(
    op_keys: Iterable[tuple[str, str]] | None = None,
) -> Counter[tuple[str, str]]:
    """Counts upstream ``onnx`` node backend tests by data-folder name.

    Each ``onnx/backend/test/data/node/test_*`` subfolder is attributed to
    the operator whose name (see :func:`_op_name_forms`) matches the
    folder name as the longest prefix after stripping ``test_``. This is
    the convention used by the upstream ONNX project (one folder per
    operator variant, named ``test_<op>(_<variant>)*``).

    Returns an empty counter when the upstream ``onnx`` package (or its
    backend test data) is not available.
    """
    try:
        import onnx
        from onnx.backend.test.loader import load_model_tests
    except ImportError:
        return Counter()

    try:
        tests = load_model_tests(kind="node")
    except Exception:  # pragma: no cover - defensive
        return Counter()

    if op_keys is None:
        # Fall back to the schemas exposed by ``onnx.defs``; this matches the
        # set of operators a maintainer would consider when reading the
        # comparison page.
        try:
            from onnx import defs

            op_keys = {(s.domain or "ai.onnx", s.name) for s in defs.get_all_schemas()}
        except ImportError:  # pragma: no cover - defensive
            op_keys = set()
    sorted_forms = _build_op_form_index(op_keys)

    counts: Counter[tuple[str, str]] = Counter()
    for t in tests:
        # The backend-test attribution mirrors the data-folder name: each
        # ``test_<op>(_<variant>)*`` subfolder counts as one case for ``<op>``.
        key = _attribute_test_name(t.name, sorted_forms)
        if key is None:
            # Fall back to the first node of the model when the folder name
            # does not match any known operator (defensive: keeps the count
            # consistent with prior behaviour for custom or third-party
            # operators that ship test data but no schema).
            model_path = os.path.join(t.model_dir, "model.onnx") if t.model_dir else None
            if not model_path or not os.path.exists(model_path):
                continue
            try:
                m = onnx.load(model_path, load_external_data=False)
            except Exception:  # pragma: no cover - defensive
                continue
            if not m.graph.node:
                continue
            n = m.graph.node[0]
            key = (n.domain or "ai.onnx", n.op_type)
        counts[key] += 1
    return counts


def _onnx_schemas_with_shape_inference() -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Returns ``(all_op_keys, shape_inference_op_keys)`` from :mod:`onnx.defs`.

    Both sets are empty when the upstream ``onnx`` package is not importable.
    """
    try:
        from onnx import defs
    except ImportError:
        return set(), set()

    all_ops: set[tuple[str, str]] = set()
    with_shape: set[tuple[str, str]] = set()
    for s in defs.get_all_schemas():
        key = (s.domain or "ai.onnx", s.name)
        all_ops.add(key)
        if s.has_type_and_shape_inference_function:
            with_shape.add(key)
    return all_ops, with_shape


def compute_schema_comparison() -> SchemaComparison:
    """Computes the full :class:`SchemaComparison`.

    The function silently degrades when the upstream :mod:`onnx` package is
    not importable: the ``onnx``-side columns are then left empty and
    :attr:`SchemaComparison.onnx_available` is set to ``False``.
    """
    light_schemas = _light_schemas_latest()
    onnx_all, onnx_with_shape = _onnx_schemas_with_shape_inference()
    # Build the union of operator keys known to either side, so test-name
    # attribution can match every operator the comparison will display.
    all_op_keys: set[tuple[str, str]] = set(light_schemas) | onnx_all
    light_tests = _count_onnx_light_backend_tests(all_op_keys)
    onnx_tests = _count_onnx_backend_tests(all_op_keys)
    onnx_available = bool(onnx_all)

    all_keys: set[tuple[str, str]] = (
        set(light_schemas) | onnx_all | set(light_tests) | set(onnx_tests)
    )

    rows: list[SchemaComparisonRow] = []
    for key in sorted(all_keys):
        domain, name = key
        rows.append(
            SchemaComparisonRow(
                domain=domain,
                name=name,
                in_onnx=key in onnx_all,
                in_onnx_light=key in light_schemas,
                onnx_shape_inference=key in onnx_with_shape,
                onnx_light_shape_inference=key in ONNX_OPTIM_SHAPE_INFERENCE_OPS,
                onnx_backend_tests=int(onnx_tests.get(key, 0)),
                onnx_light_backend_tests=int(light_tests.get(key, 0)),
            )
        )
    return SchemaComparison(rows=rows, onnx_available=onnx_available)


def _yn(value: bool) -> str:
    """Returns the checkbox for true and an empty string for false."""
    return "☑" if value else ""


def render_rst_table(
    comparison: SchemaComparison, only_in_either: bool = True, css_class: str | None = None
) -> str:
    """Renders *comparison* as one reST ``list-table`` directive per domain.

    :param comparison: The comparison to render.
    :param only_in_either: When ``True`` (the default), operators that are
        absent from both ``onnx`` and ``onnx_light`` (this can happen for
        custom-domain test fixtures) are filtered out.
    :param css_class: When provided, a ``:class:`` option is emitted on the
        ``list-table`` directive. This is used by the documentation to opt
        the rendered ``<table>`` element into the ``sphinx-datatables``
        extension (``css_class="sphinx-datatable"``), which turns the table
        into an interactive DataTables widget (search box, column sorting,
        pagination).
    :returns: A multi-line string containing domain rubrics and table
        directives, ready to be
        emitted in a ``runpython`` block.
    """
    by_domain: dict[str, list[SchemaComparisonRow]] = {}
    for r in comparison.rows:
        if only_in_either and not r.in_onnx and not r.in_onnx_light:
            continue
        by_domain.setdefault(r.domain, []).append(r)

    parts: list[str] = []
    for domain, rows in sorted(by_domain.items()):
        parts.append(f".. rubric:: {domain}\n\n")
        header_lines = [
            ".. list-table::",
            "    :header-rows: 1",
            "    :widths: 18 8 8 14 14 13 13",
        ]
        if css_class:
            header_lines.append(f"    :class: {css_class}")
        header = "\n".join(header_lines) + "\n\n"
        header += (
            "    * - Operator\n"
            "      - ``onnx``\n"
            "      - ``onnx_light``\n"
            "      - ``onnx`` shape inference\n"
            "      - ``onnx_light`` shape inference (``onnx_optim``)\n"
            "      - ``onnx`` backend tests\n"
            "      - ``onnx_light`` backend tests\n"
        )
        body_parts: list[str] = []
        for r in rows:
            body_parts.append(
                "    * - "
                + r.name
                + "\n      - "
                + _yn(r.in_onnx)
                + "\n      - "
                + _yn(r.in_onnx_light)
                + "\n      - "
                + _yn(r.onnx_shape_inference)
                + "\n      - "
                + _yn(r.onnx_light_shape_inference)
                + "\n      - "
                + str(r.onnx_backend_tests)
                + "\n      - "
                + str(r.onnx_light_backend_tests)
                + "\n"
            )
        parts.append(header + "".join(body_parts))
    return "".join(parts)


def render_rst_summary(comparison: SchemaComparison) -> str:
    """Renders a short reST summary (totals) for *comparison*."""
    lines = [
        ".. list-table::",
        "    :header-rows: 1",
        "    :widths: 50 25 25",
        "",
        "    * - Metric",
        "      - ``onnx``",
        "      - ``onnx_light``",
        "    * - Operators with a schema",
        f"      - {comparison.total_onnx}",
        f"      - {comparison.total_onnx_light}",
        "    * - Operators with shape inference",
        f"      - {comparison.total_onnx_shape_inference}",
        f"      - {comparison.total_onnx_light_shape_inference}",
        "    * - Node backend tests (counted)",
        f"      - {comparison.total_onnx_backend_tests}",
        f"      - {comparison.total_onnx_light_backend_tests}",
    ]
    return "\n".join(lines) + "\n"


def iter_rows(comparison: SchemaComparison) -> Iterable[SchemaComparisonRow]:
    """Iterates over ``comparison.rows`` (convenience helper)."""
    yield from comparison.rows

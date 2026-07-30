import re
from typing import Any, Callable, Sequence, TypeAlias
import numpy as np
from ..... import onnx
from .....onnx import helper as onnx_helper

try:
    from .....onnx_py._onnxpybackend import backend_test as _backend_test_cc  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised only in reduced builds
    raise ImportError(
        "onnx-light was built without the backend-test extensions "
        "(ONNX_LIGHT_BUILD_KERNELS=OFF); install the full build to use the "
        "backend test cases."
    ) from exc
from .....onnx_py._onnxpyprotoop import onnx_op as _onnx_op  # type: ignore
from .....ext_test_case import ExtTestCase

_LIGHT_SINCE_VERSION_CACHE: dict[tuple[str, str], int] = {}
# Backend test inputs/outputs are usually ndarrays, but ONNX sequence cases use
# recursively nested Python lists of ndarrays, and map-typed inputs are Python dicts.
# ``None`` marks a graph input with no associated data (model-only validation cases).
BackendTestValue: TypeAlias = np.ndarray | list["BackendTestValue"] | dict[Any, Any] | None
BackendTestDataSets: TypeAlias = Sequence[
    tuple[Sequence[BackendTestValue], Sequence[BackendTestValue]]
]


def _latest_since_version(op_type: str, domain: str) -> int:
    """Returns the largest ``since_version`` declared by a ``LightOpSchema``
    for ``(domain, op_type)``.

    Uses ``onnx_op.GetAllOnnxOpSchemasWithHistory()`` (the C++-side
    ``LightOpSchema`` registry) so that ``expect`` does not depend on the
    full ONNX schema registry being initialised.
    """
    if not _LIGHT_SINCE_VERSION_CACHE:
        from ....onnx_py._onnxpy import onnx_op as _op  # type: ignore

        for sch in _op.GetAllOnnxOpSchemasWithHistory():
            key = (sch.domain, sch.name)
            prev = _LIGHT_SINCE_VERSION_CACHE.get(key, -1)
            if int(sch.since_version) > prev:
                _LIGHT_SINCE_VERSION_CACHE[key] = int(sch.since_version)
    # ai.onnx is exposed as the empty string in NodeProto/OpsetIdProto.
    lookup_domain = "ai.onnx" if domain == "" else domain
    assert (
        lookup_domain,
        op_type,
    ) in _LIGHT_SINCE_VERSION_CACHE, f"Missing information for {(lookup_domain, op_type)}"
    return _LIGHT_SINCE_VERSION_CACHE[(lookup_domain, op_type)]


class TestCase(_backend_test_cc.TestCase):
    """
    Defines a test case.

    Inherits the immutable metadata fields (``name``, ``model_name``, ``kind``,
    ``tag``) and the numeric tolerances (``rtol``, ``atol``) from the C++
    :class:`onnx_light.onnx_py._onnxpy.backend_test.TestCase` binding. The
    Python-only fields (``url``, ``model_dir``) and the Python-side
    ``model``/``data_sets`` overlay (which store an :class:`onnx.ModelProto`
    and Python sequences of numpy arrays, distinct from the C++
    ``vector<DataSet>`` of raw-byte ``Tensor`` instances) are added by this
    subclass.
    """

    # Tell PyTest this isn't a real test.
    __test__ = False

    def __init__(
        self,
        name: str,
        model_name: str,
        url: str | None,
        model_dir: str | None,
        model: onnx.ModelProto | None,
        data_sets: BackendTestDataSets | None,
        kind: str,
        atol: float,
        rtol: float,
        tag: str = "",
    ) -> None:
        super().__init__(
            name=name, model_name=model_name, kind=kind, tag=tag, atol=atol, rtol=rtol
        )
        self.url = url
        self.model_dir = model_dir
        # The C++ ``model`` / ``data_sets`` members hold C++-side proto / Tensor
        # instances. Python callers populate them with the Python ``ModelProto``
        # and lists of numpy arrays, so we shadow them with Python-side storage
        # accessed through the ``model`` / ``data_sets`` properties below.
        self._py_model = model
        self._py_data_sets = data_sets

    @property
    def model(self) -> onnx.ModelProto | None:
        return self._py_model

    @model.setter
    def model(self, value: onnx.ModelProto | None) -> None:
        self._py_model = value

    @property
    def data_sets(self) -> BackendTestDataSets | None:
        return self._py_data_sets

    @data_sets.setter
    def data_sets(self, value: BackendTestDataSets | None) -> None:
        self._py_data_sets = value

    def __repr__(self) -> str:
        "usual"
        return f"{self.__class__.__name__}(name={self.name!r}, kind={self.kind!r})"

    def assert_allclose(self, rt: Callable, atol: float | None = None, rtol: float | None = None):
        """
        Checks that the outputs match the expected outputs.
        Uses atol, rtol from the class or overwritten values.
        """
        if not self.data_sets:
            return
        use_atol = atol if atol is not None else self.atol
        use_rtol = rtol if rtol is not None else self.rtol
        for i, (inputs, expected) in enumerate(self.data_sets):
            outputs = rt(self.model, *inputs)
            if outputs is None:
                # The test only validates the model and the inputs.
                continue
            assert len(outputs) == len(expected), (
                f"Number of outputs ({len(outputs)}) != expected ({len(expected)}) "
                f"in test {self!r}"
            )

            def _assert_value(out, exp, location):
                if isinstance(exp, list):
                    assert isinstance(out, (list, tuple)), (
                        f"{location} is not a sequence in test {self.name!r}: "
                        f"{type(out)} != {type(exp)}"
                    )
                    assert len(out) == len(exp), (
                        f"{location} length mismatch for test {self.name!r}: "
                        f"{len(out)} != {len(exp)}"
                    )
                    for k, (sub_out, sub_exp) in enumerate(zip(out, exp)):
                        _assert_value(sub_out, sub_exp, f"{location}[{k}]")
                    return

                exp_arr = np.asarray(exp)
                if exp_arr.dtype.kind in ("U", "S", "O"):
                    np.testing.assert_array_equal(
                        np.asarray(out),
                        exp_arr,
                        err_msg=f"{location} mismatch for test {self.name!r}",
                    )
                else:
                    np.testing.assert_allclose(
                        out,
                        exp,
                        rtol=use_rtol,
                        atol=use_atol,
                        err_msg=f"{location} mismatch for test {self.name!r}",
                    )

            for j, (out, exp) in enumerate(zip(outputs, expected)):
                _assert_value(out, exp, f"Output {i}/{j}")


class Base:
    """Base class for all tests."""


ALL_TESTS: dict[str, TestCase] = {}


def _light_op_since_version(op_type: str, domain: str) -> int:
    """Returns the latest ``since_version`` of an operator from ``LightOpSchema``.

    Uses the C++ ``onnx_op`` extension (``GetAllOnnxOpSchemasWithHistory``)
    rather than the full ``onnx.defs`` registry, to keep this module aligned
    with the lightweight operator schema source of truth used across
    ``onnx-light``.
    """
    # ``LightOpSchema`` records use ``ai.onnx`` for the standard domain while
    # ``NodeProto.domain`` uses the empty string as the equivalent shorthand.
    lookup_domain = _onnx_op.kOnnxDomain if domain == "" else domain

    best = -1
    for schema in _onnx_op.GetAllOnnxOpSchemasWithHistory():
        if schema.name == op_type and schema.domain == lookup_domain:
            if schema.since_version > best:
                best = schema.since_version
    if best < 0:
        raise ValueError(f"No LightOpSchema found for op_type={op_type!r} domain={domain!r}.")
    return best


def _transform_value(arr) -> BackendTestValue:
    if isinstance(arr, list):
        return [_transform_value(x) for x in arr]
    if isinstance(arr, (int, float, str, np.integer, np.floating, np.str_)):
        arr = np.array(arr)
    assert isinstance(arr, np.ndarray), f"Not implemented when arr is {type(arr)}."
    return arr


# build value infos using onnx_light helper
def _extract_vi(arr, arr_name):
    if isinstance(arr, onnx.TensorProto):
        elem_type = arr.data_type
        shape = tuple(arr.dims)
        return onnx_helper.make_tensor_value_info(arr_name, elem_type, shape)
    if isinstance(arr, list):
        elem_type = onnx_helper.np_dtype_to_tensor_dtype(arr[0].dtype)
        return onnx_helper.make_tensor_sequence_value_info(arr_name, elem_type, None)
    if isinstance(arr, (int, float, str, np.integer, np.floating, np.str_)):
        arr = np.array(arr)
    elem_type = onnx_helper.np_dtype_to_tensor_dtype(arr.dtype)
    return onnx_helper.make_tensor_value_info(arr_name, elem_type, list(arr.shape))


def expect(
    node_op: onnx.NodeProto,
    inputs: Sequence[np.ndarray | onnx.TensorProto | float | int],
    outputs: Sequence[np.ndarray | onnx.TensorProto | float | int],
    name: str,
    **kwargs: Any,
) -> None:
    """
    In the case of ops with optional inputs and outputs, node_op.input and node_op.output indicate
    which inputs/outputs are present and which are omitted. However, the parameter inputs
    and outputs of this function include values only for inputs/outputs that are present.
    E.g., for an op with 3 inputs, if the second parameter is optional and we wish to omit it,
    node_op.inputs would look like ["Param1", "", "Param3"], while inputs would look like
    [input-1-value, input-3-value]
    Instead of creating model with latest version, it now generates models for since_version
    by default. Thus it can make every model uses the same opset version after every opset
    change. Besides, user can specify "use_max_opset_version" to generate models for
    the latest opset version that supports before targeted opset version.

    float or int are converted into numpy arrays with an empty shape.
    """
    # retrieve the specifications for this node
    op_type = node_op.op_type
    domain = node_op.domain

    schema_since_version = _light_op_since_version(op_type, domain)

    present_inputs = [x for x in node_op.input if x != ""]
    present_outputs = [x for x in node_op.output if x != ""]

    inputs_vi = [_extract_vi(arr, n) for arr, n in zip(inputs, present_inputs)]
    outputs_vi = [_extract_vi(arr, n) for arr, n in zip(outputs, present_outputs)]

    # create a model based on that specification
    if "opset_imports" not in kwargs:
        opset_imports = [onnx_helper.make_opsetid(domain, schema_since_version)]
    else:
        opset_imports = kwargs.pop("opset_imports")

    graph = onnx_helper.make_graph(
        nodes=[node_op], name=name, inputs=inputs_vi, outputs=outputs_vi
    )
    model = onnx_helper.make_model(
        graph,
        opset_imports=opset_imports,
        producer_name=kwargs.pop("producer_name", "backend-test"),
        **kwargs,
    )

    inputs_dict = dict(zip(present_inputs, map(_transform_value, inputs)))
    outputs_dict = dict(zip(present_outputs, map(_transform_value, outputs)))

    # add this information into a dictionary
    ALL_TESTS[name] = TestCase(
        name=name,
        model_name=name,
        url=None,
        model_dir=None,
        model=model,
        data_sets=[(list(inputs_dict.values()), list(outputs_dict.values()))],
        kind="node",
        atol=1e-7,
        rtol=1e-3,
    )


def _init_dtype_maps():
    """Initializes and returns the dtype mapping tables (lazily, on first call).

    Returns:
        A tuple ``(DTYPE_TO_NP, SUB_BYTE_DTYPES)`` used by the tensor
        conversion helpers.
    """
    import ml_dtypes as _ml_dtypes

    dtype_to_np = {
        int(onnx.TensorProto.FLOAT): np.float32,
        int(onnx.TensorProto.DOUBLE): np.float64,
        int(onnx.TensorProto.INT32): np.int32,
        int(onnx.TensorProto.INT64): np.int64,
        int(onnx.TensorProto.UINT8): np.uint8,
        int(onnx.TensorProto.INT8): np.int8,
        int(onnx.TensorProto.BOOL): np.bool_,
        int(onnx.TensorProto.UINT16): np.uint16,
        int(onnx.TensorProto.INT16): np.int16,
        int(onnx.TensorProto.UINT32): np.uint32,
        int(onnx.TensorProto.UINT64): np.uint64,
        int(onnx.TensorProto.FLOAT16): np.float16,
        int(onnx.TensorProto.BFLOAT16): _ml_dtypes.bfloat16,
        int(onnx.TensorProto.FLOAT8E4M3FN): _ml_dtypes.float8_e4m3fn,
        int(onnx.TensorProto.FLOAT8E4M3FNUZ): _ml_dtypes.float8_e4m3fnuz,
        int(onnx.TensorProto.FLOAT8E5M2): _ml_dtypes.float8_e5m2,
        int(onnx.TensorProto.FLOAT8E5M2FNUZ): _ml_dtypes.float8_e5m2fnuz,
        int(onnx.TensorProto.FLOAT8E8M0): _ml_dtypes.float8_e8m0fnu,
    }
    sub_byte_dtypes = {
        int(onnx.TensorProto.INT4): (_ml_dtypes.int4, 4, True),
        int(onnx.TensorProto.UINT4): (_ml_dtypes.uint4, 4, False),
        int(onnx.TensorProto.INT2): (_ml_dtypes.int2, 2, True),
        int(onnx.TensorProto.UINT2): (_ml_dtypes.uint2, 2, False),
    }
    return dtype_to_np, sub_byte_dtypes


_DTYPE_MAPS: tuple[dict, dict] | None = None


def _get_dtype_maps():
    """Returns the cached ``(DTYPE_TO_NP, SUB_BYTE_DTYPES)`` tuple."""
    global _DTYPE_MAPS
    if _DTYPE_MAPS is None:
        _DTYPE_MAPS = _init_dtype_maps()
    return _DTYPE_MAPS


def _unpack_sub_byte(raw: bytes, shape, dtype, bits: int, signed: bool):
    """Unpacks sub-byte packed integers from raw bytes into a numpy array."""
    n = 1
    for d in shape:
        n *= int(d)
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    buf = np.frombuffer(raw, dtype=np.uint8)
    out = np.empty(n, dtype=np.int64 if signed else np.uint64)
    for i in range(n):
        byte = int(buf[i // per_byte])
        v = (byte >> (bits * (i % per_byte))) & mask
        if signed and (v & sign_bit):
            v -= 1 << bits
        out[i] = v
    return out.astype(dtype).reshape(tuple(int(d) for d in shape))


def _unpack_float4_e2m1(raw: bytes, shape):
    """Unpacks FLOAT4E2M1 values from raw bytes into a numpy array."""
    import ml_dtypes as _ml_dtypes

    n = 1
    for d in shape:
        n *= int(d)
    buf = np.frombuffer(raw, dtype=np.uint8)
    nibbles = np.empty(n, dtype=np.uint8)
    for i in range(n):
        byte = int(buf[i // 2])
        nibbles[i] = (byte >> (4 * (i % 2))) & 0x0F
    return nibbles.view(_ml_dtypes.float4_e2m1fn).reshape(tuple(int(d) for d in shape))


def _tensor_to_np(t):
    """Converts a C++ backend-test ``Tensor`` to a numpy array."""
    if int(t.data_type) == int(onnx.TensorProto.STRING):
        values = t.string_data()
        arr = np.array(values, dtype=object)
        return arr.reshape(tuple(int(d) for d in t.shape))
    dtype_to_np, sub_byte_dtypes = _get_dtype_maps()
    sub = sub_byte_dtypes.get(int(t.data_type))
    if sub is not None:
        dtype, bits, signed = sub
        return _unpack_sub_byte(t.raw_data(), t.shape, dtype, bits, signed)
    if int(t.data_type) == int(onnx.TensorProto.FLOAT4E2M1):
        return _unpack_float4_e2m1(t.raw_data(), t.shape)
    dtype = dtype_to_np.get(int(t.data_type))
    if dtype is None:
        raise NotImplementedError(
            f"Cannot convert C++ Tensor with data_type={t.data_type} to numpy."
        )
    arr = np.frombuffer(t.raw_data(), dtype=dtype)
    return arr.reshape(tuple(int(d) for d in t.shape))


def _ds_inputs_to_python(tc: Any) -> list[list[np.ndarray | dict[Any, Any] | None]]:
    """Returns per-DataSet positional inputs for ``tc``.

    For graph inputs declared with ``map(K, V)`` type (used by
    ``ai.onnx.ml::DictVectorizer`` and ``ai.onnx.ml::CastMap``), the
    DataSet stores a Map object in ``ds.maps``. The keys and values
    tensors are combined into a single Python ``dict`` so the backend
    harness (which zips them against ``sess.input_names``) feeds the
    map under its original graph-input name.
    """
    graph_inputs = list(tc.model.graph.input)
    data_sets: list[list[np.ndarray | dict[Any, Any] | None]] = []
    for ds in tc.data_sets:
        by_name = {t.name: _tensor_to_np(t) for t in ds.inputs}
        maps_by_name = {m.name: m for m in ds.maps} if ds.maps else {}
        inputs: list[np.ndarray | dict[Any, Any] | None] = []
        for gi in graph_inputs:
            if gi.type.has_map_type():
                m = maps_by_name.get(gi.name)
                if m is not None:
                    keys_np = _tensor_to_np(m.keys)
                    values_np = _tensor_to_np(m.values)
                    inputs.append(dict(zip(keys_np.tolist(), values_np.tolist())))
                else:
                    inputs.append(by_name.get(gi.name))
            else:
                inputs.append(by_name.get(gi.name))
        data_sets.append(inputs)
    return data_sets


def _expected_output_to_python(t, sequence_outputs):
    """Converts a DataSet output ``Tensor`` to its Python expected value.

    Sequence-typed graph outputs are materialized by the C++ test cases as a
    single stacked tensor whose outer (axis 0) dimension is the sequence
    length. Splits such a tensor back into a list of per-element arrays so it
    matches the sequence value (a list of arrays) produced by the runtime,
    instead of a single stacked array that would mismatch as
    "sequence vs non-sequence".

    Returns:
        A list of per-element ``numpy.ndarray`` when ``t`` names a
        sequence-typed graph output, otherwise the single ``numpy.ndarray``.
    """
    arr = _tensor_to_np(t)
    if t.name in sequence_outputs:
        return [arr[i] for i in range(arr.shape[0])]
    return arr


def _cc_to_python_test_case(cc_tc: Any) -> TestCase:
    """Converts a single C++ ``TestCase`` to the Python :class:`TestCase`.

    Wraps the C++-side model and data sets into the Python ``TestCase``
    subclass with numpy arrays (instead of raw-byte ``Tensor`` instances).
    """
    sequence_outputs = {o.name for o in cc_tc.model.graph.output if o.type.has_sequence_type()}
    py_inputs = _ds_inputs_to_python(cc_tc)
    data_sets = [
        (py_inputs[i], [_expected_output_to_python(y, sequence_outputs) for y in ds.outputs])
        for i, ds in enumerate(cc_tc.data_sets)
    ]
    return TestCase(
        name=cc_tc.name,
        model_name=cc_tc.model_name,
        url=None,
        model_dir=None,
        model=cc_tc.model,
        data_sets=data_sets,
        kind=cc_tc.kind,
        atol=cc_tc.atol,
        rtol=cc_tc.rtol,
        tag=cc_tc.tag,
    )


def _collect_cc_test_cases(
    include_big: bool = False, mode: "_backend_test_cc.TestMode | None" = None
) -> dict[str, TestCase]:
    """Collects backend test cases produced by the C++ ``lib_onnx_backend_test``.

    The C++ library implements the same data model as the Python infrastructure
    in this module (``TestCase`` + ``DataSet``). Each C++ ``TestCase`` exposes a
    serialized ``ModelProto`` and per-dataset ``Tensor`` objects (whose
    ``raw_data`` bytes are in row-major little-endian layout); we wrap them
    back into the Python ``TestCase`` dataclass so they integrate seamlessly
    with ``make_test_class``.

    Args:
        include_big: When ``True``, includes backend test cases whose name
            contains ``"_big_"``. Defaults to ``False``, which keeps these
            big cases excluded.
        mode: The generation mode (a ``TestMode`` value). When ``None``
            (default), defaults to ``TestMode.TEST`` which yields the standard
            correctness cases. ``TestMode.BENCHMARK`` yields large benchmark-sized
            cases where supported.

    Returns:
        A dictionary mapping test case names to TestCase instances.
    """
    from .....onnx_py._onnxpybackend import backend_test as _backend_test_cc  # type: ignore[attr-defined]

    result: dict[str, TestCase] = {}
    if mode is None:
        mode = _backend_test_cc.TestMode.TEST
    for tc in _backend_test_cc.collect_test_cases(include_big=include_big, mode=mode):
        if tc.name.startswith("test_cc_zipmap_"):
            continue
        result[tc.name] = _cc_to_python_test_case(tc)
    return result


def collect_test_case(
    include_big: bool = False, mode: "_backend_test_cc.TestMode | None" = None
) -> dict[str, TestCase]:
    """
    Collects all backend test cases.

    The canonical node test cases are produced by the C++
    ``lib_onnx_backend_test`` library and exposed through the
    ``onnx_light.onnx_py._onnxpy.backend_test`` Python bindings. In
    addition, any user-defined :class:`Base` subclass with ``export*``
    class methods is executed so that downstream code can still register
    extra Python-defined cases through the :func:`expect` helper.
    Python-defined cases take precedence over C++ cases of the same name.

    Args:
        include_big: When ``True``, includes backend test cases whose name
            contains ``"_big_"``. Defaults to ``False``, which keeps these
            big cases excluded.
        mode: The generation mode (a ``TestMode`` value). When ``None``
            (default), defaults to ``TestMode.TEST`` which yields the standard
            correctness cases. ``TestMode.BENCHMARK`` yields large benchmark-sized
            cases where supported.

    Returns:
        A dictionary mapping test case names to TestCase instances.
    """
    global ALL_TESTS

    # empty ALL_TESTS before collecting
    ALL_TESTS.clear()

    # call all export methods on user-defined Base subclasses so they can
    # register additional Python-only test cases through ``expect``.
    for subclass in Base.__subclasses__():
        for attr_name in dir(subclass):
            if attr_name.startswith("export"):
                method = getattr(subclass, attr_name)
                if callable(method):
                    method()

    # merge in C++-generated backend test node cases (Python-defined cases win
    # on name collision to preserve backwards compatibility)
    cc_cases = _collect_cc_test_cases(include_big=include_big, mode=mode)
    for name, tc in cc_cases.items():
        ALL_TESTS.setdefault(name, tc)

    # copy ALL_TESTS and reset it
    result = dict(ALL_TESTS)
    ALL_TESTS.clear()
    return result


def get_test_case(name: str, mode: "_backend_test_cc.TestMode | None" = None) -> TestCase | None:
    """Returns a single backend test case by exact name, or ``None``.

    Unlike :func:`collect_test_case`, which collects *all* C++ test cases
    and converts every one to Python, this function uses the C++ exact-name
    lookup (:func:`get_test_case_by_name`) to retrieve only the requested
    case without regex overhead. This is significantly faster when only one
    case is needed.

    Args:
        name: The exact test case name (e.g.
            ``"test_cc_loop_zero_trip_count"``).
        mode: The generation mode (a ``TestMode`` value). When ``None``
            (default), defaults to ``TestMode.TEST``.

    Returns:
        The :class:`TestCase` instance, or ``None`` if no case with that
        name exists.
    """
    from .....onnx_py._onnxpybackend import backend_test as _bt  # type: ignore[attr-defined]

    if mode is None:
        mode = _bt.TestMode.TEST
    cases = _bt.get_test_case_by_name(name, include_big=True, mode=mode)
    if not cases:
        return None
    return _cc_to_python_test_case(cases[0])


def get_test_cases_for_op(
    op_type: str,
    opset_version: int | None = None,
    domain: str = "",
    test_cases: dict[str, TestCase] | None = None,
) -> dict[str, TestCase]:
    """
    Retrieves backend test cases involving a specific operator and opset.

    A test case matches if its underlying ``ModelProto`` contains at least one
    node whose ``op_type`` and ``domain`` equal the requested values. When
    ``opset_version`` is provided, the test case must additionally import the
    given ``domain`` at exactly that version (``opset_import`` entry matching
    both ``domain`` and ``version``).

    Args:
        op_type: Operator type to look up (e.g. ``"Abs"``).
        opset_version: If not ``None``, only return test cases whose model
            imports the given ``domain`` at exactly this version.
        domain: Operator domain. Defaults to the standard ``ai.onnx`` domain
            (``""``).
        test_cases: Optional precomputed mapping returned by
            :func:`collect_test_case`. When ``None``, :func:`collect_test_case`
            is called.

    Returns:
        A new ``dict`` mapping test case names to :class:`TestCase` instances
        that match the request.
    """
    if test_cases is None:
        test_cases = collect_test_case()

    result: dict[str, TestCase] = {}
    for name, tc in test_cases.items():
        if tc.model is None:
            continue
        # Look for at least one node matching (op_type, domain).
        has_node = any(
            node.op_type == op_type and node.domain == domain for node in tc.model.graph.node
        )
        if not has_node:
            continue
        if opset_version is not None:
            matches_opset = any(
                opset.domain == domain and opset.version == opset_version
                for opset in tc.model.opset_import
            )
            if not matches_opset:
                continue
        result[name] = tc
    return result


def make_test_class(
    rt: Callable,
    include_regex: Sequence[str] | None = None,
    exclude_regex: Sequence[str] | None = None,
    atols: dict[str, float] | None = None,
    rtols: dict[str, float] | None = None,
    include_big: bool = False,
):
    """
    Collects all test cases with collect_test_case.
    Keeps or removes tests based on include_regex and exclude_regex.
    Creates a test class which has a test method per test, like ``test_{name}``.
    Compares outputs.

    If ``rt`` declares a single positional parameter (i.e. ``rt(model)``), it
    is treated as a model-level validator: it is invoked once per test case
    as ``rt(tc.model)`` and no output comparison is performed. This is the
    path used by :mod:`onnx_light.onnx.checker` (``check_model``).
    """
    # Collect all test cases
    all_tests = collect_test_case(include_big=include_big)

    # Filter tests based on include_regex and exclude_regex
    filtered_tests = {}
    for name, test_case in all_tests.items():
        # Check exclude patterns first
        if exclude_regex:
            excluded = False
            for pattern in exclude_regex:
                if re.search(pattern, name):
                    excluded = True
                    break
            if excluded:
                continue

        # Check include patterns
        if include_regex:
            included = False
            for pattern in include_regex:
                if re.search(pattern, name):
                    included = True
                    break
            if not included:
                continue

        filtered_tests[name] = test_case

    # Create test class dynamically
    class BackendTest(ExtTestCase):
        """Represents a dynamically generated test class for backend tests."""

    # Add test methods to the class
    for name, test_case in filtered_tests.items():
        # Get custom tolerances if provided
        atol = atols.get(name) if atols else None
        rtol = rtols.get(name) if rtols else None

        # Create test method using default arguments to capture loop variables
        def test_func(self, tc=test_case, custom_atol=atol, custom_rtol=rtol):
            tc.assert_allclose(rt, atol=custom_atol, rtol=custom_rtol)

        # Add the test method to the class
        test_func.__name__ = f"test_{name}"
        test_func.__doc__ = f"Test case: {name}"
        setattr(BackendTest, f"test_{name}", test_func)

    return BackendTest

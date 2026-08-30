import importlib.util
import os
import shutil
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from io import StringIO
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

HAS_OPTIM_EXT: bool = importlib.util.find_spec("onnx_light.onnx_py._onnxpycore") is not None


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


def has_onnxruntime() -> bool:
    "Tells if onnxruntime is installed."
    try:
        import onnxruntime

        return hasattr(onnxruntime, "__version__")
    except ImportError:
        return False


def has_ir_py() -> bool:
    "Tells if ir-py is installed."
    try:
        import onnx_ir  # type: ignore

        return hasattr(onnx_ir, "__version__")
    except ImportError:
        return False


def import_or_skip(module_name: str, attribute: Optional[str] = None) -> Any:
    """Imports a module (or one of its attributes) or skips the test otherwise.

    onnx-light can be built without the operator-kernel runtime and the
    backend-test registries (``ONNX_LIGHT_BUILD_KERNELS=OFF``). In that reduced
    build the kernel/backend Python modules raise :class:`ImportError`. Tests
    that depend on them call this helper, at module level or inside a test, to
    skip themselves cleanly instead of failing.

    Args:
        module_name: The fully-qualified name of the module to import.
        attribute: An optional attribute to return from the imported module.

    Returns:
        The imported module, or ``getattr(module, attribute)`` when *attribute*
        is provided.

    Raises:
        unittest.SkipTest: When the module (or attribute) cannot be imported.
    """
    try:
        module = import_module(module_name)
        if attribute is not None:
            return getattr(module, attribute)
        return module
    except (ImportError, AttributeError) as exc:
        raise unittest.SkipTest(
            f"{module_name!r} is unavailable in this build "
            f"(reduced onnx-light build without kernels/backend): {exc}"
        ) from exc


def skipif_ci_windows(msg) -> Callable:
    """Skips a unit test if it runs on Windows."""
    if is_windows():
        msg = f"Test does not work on Windows. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_ci_apple(msg) -> Callable:
    """Skips a unit test if it runs on Macosx."""
    if is_apple():
        msg = f"Test does not work on Apple. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def skipif_unstable(msg) -> Callable:
    """Skips a unit test if the environment variable `SKIP_UNSTABLE` is set to 1."""
    value = os.environ.get("SKIP_UNSTABLE", "0")
    if value in ("1", 1):
        msg = f"Test is unstable. Disabling it. {msg}"
        return unittest.skip(msg)
    return lambda x: x


def unit_test_going():
    """
    Enables a flag telling the script is running while testing it.
    Avois unit tests to be very long.
    """
    going = int(os.environ.get("UNITTEST_GOING", 0))
    return going == 1


def ignore_warnings(warns: Sequence[type[Warning]]) -> Callable:
    """
    Catches warnings.

    :param warns:   warnings to ignore
    """

    def wrapper(fct):
        assert warns is not None, f"warns cannot be None for '{fct}'."

        def call_f(self):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", warns)  # type: ignore
                return fct(self)

        return call_f

    return wrapper


def ignore_errors(errors: Union[Exception, Tuple[Exception]]) -> Callable:
    """
    Catches exception, skip the test if the error is expected sometimes.

    :param errors: errors to ignore
    """

    def wrapper(fct):
        if errors is None:
            raise AssertionError(f"errors cannot be None for '{fct}'.")

        def call_f(self):
            try:
                return fct(self)
            except errors as e:  # type: ignore
                raise unittest.SkipTest(  # noqa: B904
                    f"expecting error {e.__class__.__name__}: {e}"
                )

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:  # pragma: no cover
            pass
        return call_f

    return wrapper


def hide_stdout(f: Optional[Callable] = None) -> Callable:
    """
    Catches warnings, hides standard output.
    The function may be disabled by setting ``UNHIDE=1``
    before running the unit test.

    :param f: the function is called with the stdout as an argument
    """

    def wrapper(fct):
        def call_f(self):
            if os.environ.get("UNHIDE", ""):
                fct(self)
                return
            st = StringIO()
            with redirect_stdout(st), warnings.catch_warnings():
                warnings.simplefilter("ignore", (UserWarning, DeprecationWarning))  # type: ignore
                try:
                    fct(self)
                except AssertionError as e:
                    if "torch is not recent enough, file" in str(e):
                        raise unittest.SkipTest(str(e))  # noqa: B904
                    raise
            if f is not None:
                f(st.getvalue())
            return None

        try:  # noqa: SIM105
            call_f.__name__ = fct.__name__
        except AttributeError:
            pass
        return call_f

    return wrapper


class ExtTestCase(unittest.TestCase):
    _warns: List[Tuple[str, int, Warning]] = []

    def shortDescription(self):
        # To remove annoying display on the screen every time verbosity is enabled.
        return None

    def assertExists(self, name):
        assert os.path.exists(name), f"File or folder {name!r} does not exists."

    def assertIns(self, sub: Tuple[Any, ...], s: str):
        """
        Checks that one of the substrings in sub is part of s.
        """
        for t in sub:
            if t in s:
                return
        raise AssertionError(f"None of the substring in {sub} is part of {s!r}.")

    def assertIn(self, member, container, msg: Optional[Union[Callable, str]] = None):
        if not msg:
            super().assertIn(member, container)
        if member not in container:
            if callable(msg):
                super().assertIn(member, container, msg())
            else:
                super().assertIn(member, container, msg)

    def assertEqualArray(
        self,
        expected: np.ndarray,
        value: np.ndarray,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[str] = None,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        if msg:
            try:
                np.testing.assert_allclose(expected, value, atol=atol, rtol=rtol)
            except AssertionError as e:
                raise AssertionError(msg) from e
        else:
            np.testing.assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertAlmostEqual(
        self,
        expected: Any,
        value: Any,
        places: int = 7,
        msg: Optional[str] = None,
        delta: Optional[float] = None,
        *,
        atol: Optional[float] = None,
        rtol: float = 0,
    ):
        if not isinstance(expected, np.ndarray) and not isinstance(value, np.ndarray):
            if atol is None and rtol == 0:
                return super().assertAlmostEqual(expected, value, places, msg, delta)
        if not isinstance(expected, np.ndarray):
            expected = np.array(expected)
        if not isinstance(value, np.ndarray):
            value = np.array(value).astype(expected.dtype)
        if delta is not None:
            if atol is not None:
                raise TypeError("specify delta or atol, not both")
            atol = delta
        if atol is None:
            atol = 0.5 * 10 ** (-places)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol, msg=msg)

    def assertNotAlmostEqual(
        self,
        expected: Any,
        value: Any,
        places: int = 7,
        msg: Optional[str] = None,
        delta: Optional[float] = None,
        *,
        atol: Optional[float] = None,
        rtol: float = 0,
    ):
        if not isinstance(expected, np.ndarray) and not isinstance(value, np.ndarray):
            if atol is None and rtol == 0:
                return super().assertNotAlmostEqual(expected, value, places, msg, delta)
        try:
            self.assertAlmostEqual(expected, value, places, delta=delta, atol=atol, rtol=rtol)
        except AssertionError:
            return
        self.fail(msg or f"{expected!r} and {value!r} are unexpectedly almost equal.")

    def assertRaise(self, fct: Callable, exc_type: type[Exception]):
        try:
            fct()
        except exc_type as e:
            if not isinstance(e, exc_type):
                raise AssertionError(f"Unexpected exception {type(e)!r}.")  # noqa: B904
            return
        raise AssertionError("No exception was raised.")

    def assertEmpty(self, value: Any):
        if value is None:
            return
        if not value:
            return
        raise AssertionError(f"value is not empty: {value!r}.")

    @classmethod
    def _msg(cls, msg):
        if callable(msg):
            return msg()
        return msg

    def assertNotEmpty(self, value: Any, msg: Optional[Union[Callable, str]] = None):
        if value is None:
            raise AssertionError(self._msg(msg or f"value is empty: {value!r}."))
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(self._msg(msg or f"value is empty: {value!r}."))

    def assertStartsWith(self, prefix: str, full: str):
        if not full.startswith(prefix):
            raise AssertionError(f"prefix={prefix!r} does not start string  {full!r}.")

    @classmethod
    def tearDownClass(cls):
        for name, line, w in cls._warns:
            warnings.warn(f"\n{name}:{line}: {type(w)}\n  {str(w)}", stacklevel=0)

    def capture(self, fct: Callable) -> Tuple[Any, str, str]:
        """
        Runs a function and capture standard output and error.

        :param fct: function to run
        :return: result of *fct*, output, error
        """
        sout = StringIO()
        serr = StringIO()
        with redirect_stdout(sout), redirect_stderr(serr):
            try:
                res = fct()
            except Exception as e:
                raise AssertionError(
                    f"function {fct} failed, stdout="
                    f"\n{sout.getvalue()}\n---\nstderr=\n{serr.getvalue()}"
                ) from e
        return res, sout.getvalue(), serr.getvalue()

    def tryCall(
        self, fct: Callable, msg: Optional[str] = None, none_if: Optional[str] = None
    ) -> Optional[Any]:
        """
        Calls the function, catch any error.

        :param fct: function to call
        :param msg: error message to display if failing
        :param none_if: returns None if this substring is found in the error message
        :return: output of *fct*
        """
        try:
            return fct()
        except Exception as e:
            if none_if is not None and none_if in str(e):
                return None
            if msg is None:
                raise e
            raise AssertionError(msg) from e

    @classmethod
    def to_str(cls, onx: "ModelProto") -> str:  # type: ignore # noqa: F821
        if hasattr(onx, "SerializeToString"):
            return str(onx)
        raise RuntimeError(f"Unable to print type {type(onx)}")

    def get_dump_file(self, name: str, folder: Optional[str] = None, clean: bool = False) -> str:
        """Returns a filename to dump a model."""
        if folder is None:
            folder = "dump_test"
        if folder and not os.path.exists(folder):
            os.mkdir(folder)
        res = os.path.join(folder, name)
        if clean and os.path.exists(res):
            os.remove(res)
        return res

    @classmethod
    def get_dump_folder(cls, name: str, folder: Optional[str] = None, clean: bool = False) -> str:
        if folder is None:
            folder = "dump_test"
        if folder and not os.path.exists(folder):
            os.mkdir(folder)
        res = os.path.join(folder, name)
        if clean and os.path.exists(res):
            shutil.rmtree(res)
        if not os.path.exists(res):
            os.mkdir(res)
        return res

    def dump_onnx(self, name: str, proto: Any, folder: Optional[str] = None) -> str:
        """Dumps an onnx file."""
        fullname = self.get_dump_file(name, folder=folder)
        with open(fullname, "wb") as f:
            f.write(proto.SerializeToString())
        return fullname


class InferenceSessionAllTypes:
    """
    Wrapper around onnxruntime.InferenceSession that supports all ONNX dtypes.

    Uses IOBinding with raw memory buffers to support dtypes that ORT doesn't
    natively handle through NumPy conversion (FLOAT8, BFLOAT16, INT2, INT4, etc.).
    Creates an inference session.

    Args:
        model_bytes: ONNX model
        providers: List of execution providers (defaults to ["CPUExecutionProvider"])
    """

    @classmethod
    def mapping_numpy_dtype_to_onnx(cls):
        """Returns mapping from NumPy dtype to ONNX TensorProto data type."""
        import ml_dtypes  # noqa: F401
        from onnx_light.onnx import TensorProto

        return {
            np.dtype("float64"): TensorProto.DOUBLE,
            np.dtype("float32"): TensorProto.FLOAT,
            np.dtype("float16"): TensorProto.FLOAT16,
            np.dtype("uint8"): TensorProto.UINT8,
            np.dtype("int8"): TensorProto.INT8,
            np.dtype("uint16"): TensorProto.UINT16,
            np.dtype("int16"): TensorProto.INT16,
            np.dtype("int32"): TensorProto.INT32,
            np.dtype("int64"): TensorProto.INT64,
            np.dtype("bool"): TensorProto.BOOL,
            np.dtype("uint32"): TensorProto.UINT32,
            np.dtype("uint64"): TensorProto.UINT64,
            np.dtype("O"): TensorProto.STRING,
            #
            np.dtype("uint4"): TensorProto.UINT4,
            np.dtype("int4"): TensorProto.INT4,
            np.dtype("uint2"): TensorProto.UINT2,
            np.dtype("int2"): TensorProto.INT2,
            np.dtype("float8_e4m3fn"): TensorProto.FLOAT8E4M3FN,
            np.dtype("float8_e4m3fnuz"): TensorProto.FLOAT8E4M3FNUZ,
            np.dtype("float8_e5m2"): TensorProto.FLOAT8E5M2,
            np.dtype("float8_e5m2fnuz"): TensorProto.FLOAT8E5M2FNUZ,
            np.dtype("bfloat16"): TensorProto.BFLOAT16,
            # np.dtype("float4e2m1"): TensorProto.FLOAT4E2M1,
        }

    @classmethod
    def mapping_ort_type_name_to_numpy_dtype(self):
        """
        Returns mapping for special dtypes that need IOBinding.

        Maps ONNX dtype to (numpy view dtype, onnx tensor element type).
        """
        return {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.double,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int16)": np.int16,
            "tensor(int8)": np.int8,
            "tensor(uint64)": np.uint64,
            "tensor(uint32)": np.uint32,
            "tensor(uint16)": np.uint16,
            "tensor(uint8)": np.uint8,
        }

    @classmethod
    def mapping_sub_byte_types(cls):
        """
        Returns the sub-byte ONNX dtypes ORT packs into bytes through IOBinding.

        ORT stores these types packed (several values per byte), so the wrapper
        packs inputs and unpacks outputs around the raw memory buffers.

        Returns:
            A dictionary mapping the ORT type name (e.g. ``"tensor(int2)"``) to a
            tuple ``(bits, signed, onnx_element_type, numpy_dtype_name)``.
        """
        from onnx_light.onnx import TensorProto

        return {
            "tensor(int4)": (4, True, TensorProto.INT4, "int4"),
            "tensor(uint4)": (4, False, TensorProto.UINT4, "uint4"),
            "tensor(int2)": (2, True, TensorProto.INT2, "int2"),
            "tensor(uint2)": (2, False, TensorProto.UINT2, "uint2"),
        }

    @staticmethod
    def _packed_byte_count(n: int, bits: int) -> int:
        """
        Returns the number of bytes needed to pack ``n`` values of ``bits`` bits.

        Args:
            n: Number of logical values.
            bits: Number of bits per value (2 or 4).

        Returns:
            The byte count, always at least 1 so a zero-element tensor still has a
            valid (non-empty) buffer to bind.
        """
        return max((n * bits + 7) // 8, 1)

    @staticmethod
    def _pack_sub_byte(array: np.ndarray, bits: int) -> np.ndarray:
        """
        Packs a sub-byte array into a flat ``uint8`` buffer (low bits first).

        Args:
            array: Array of sub-byte values (one logical value per element).
            bits: Number of bits per value (2 or 4).

        Returns:
            A 1-D ``uint8`` buffer holding the values packed several per byte,
            matching the layout ONNX Runtime expects for sub-byte tensors.
        """
        flat = np.asarray(array).reshape(-1).astype(np.int64)
        mask = (1 << bits) - 1
        flat = flat & mask
        per_byte = 8 // bits
        n = flat.size
        nbytes = InferenceSessionAllTypes._packed_byte_count(n, bits)
        idx = np.arange(n)
        byte_idx = idx // per_byte
        shift = (idx % per_byte) * bits
        acc = np.zeros(nbytes, dtype=np.uint64)
        np.add.at(acc, byte_idx, (flat.astype(np.uint64) << shift.astype(np.uint64)))
        return acc.astype(np.uint8)

    @staticmethod
    def _unpack_sub_byte(
        buffer: np.ndarray, shape: Tuple[int, ...], bits: int, signed: bool, numpy_dtype_name: str
    ) -> np.ndarray:
        """
        Unpacks a flat ``uint8`` buffer into a sub-byte array of the given shape.

        Args:
            buffer: 1-D ``uint8`` buffer holding the values packed several per byte.
            shape: Target shape of the unpacked array.
            bits: Number of bits per value (2 or 4).
            signed: Whether the values are signed (two's complement).
            numpy_dtype_name: Name of the ``ml_dtypes`` dtype of the result.

        Returns:
            An array of the requested shape and dtype with one logical value per element.
        """
        n = int(np.prod(shape)) if shape else 1
        per_byte = 8 // bits
        mask = (1 << bits) - 1
        idx = np.arange(n)
        byte_idx = idx // per_byte
        shift = (idx % per_byte) * bits
        vals = (buffer[byte_idx].astype(np.int64) >> shift) & mask
        if signed:
            half = 1 << (bits - 1)
            vals = np.where(vals >= half, vals - (1 << bits), vals)
        return vals.astype(np.dtype(numpy_dtype_name)).reshape(shape)

    def __init__(self, model: "ModelProto", providers: Optional[List[str]] = None):  # type: ignore # noqa: F821
        import onnxruntime as ort

        if providers is None:
            providers = ["CPUExecutionProvider"]
        try:
            self._sess = ort.InferenceSession(model.SerializeToString(), providers=providers)
        except ort.capi.onnxruntime_pybind11_state.InvalidGraph as e:  # type: ignore
            from .tools.pretty_print import pretty_onnx

            raise AssertionError(
                f"Unable to load a model due to {e}\n---\n{pretty_onnx(model)}"
            ) from e
        self._mapping_to_onnx = self.mapping_numpy_dtype_to_onnx()
        self._mapping_to_numpy = self.mapping_ort_type_name_to_numpy_dtype()
        self._mapping_sub_byte = self.mapping_sub_byte_types()

    def run(
        self, output_names: Optional[List[str]], input_feed: dict[str, np.ndarray]
    ) -> List[np.ndarray]:
        """
        Runs the model with support for all ONNX dtypes.

        Uses IOBinding for special dtypes that ORT doesn't natively support.

        Args:
            output_names: Names of outputs to compute (None = all outputs)
            input_feed: Dictionary mapping input names to numpy arrays

        Returns:
            List of output arrays from the model
        """
        from onnx_light.onnx import TensorProto

        input_metas = self._sess.get_inputs()
        output_metas = self._sess.get_outputs()
        inputs = [input_feed[meta.name] for meta in input_metas]
        if output_names is None:
            output_names = [o.name for o in output_metas]

        for meta in input_metas:
            if meta.type == "tensor(string)":
                # IOBinding does not support strings.
                return self._sess.run(output_names, input_feed)  # type: ignore

        io_binding = self._sess.io_binding()

        # Keep packed buffers alive until run_with_iobinding completes.
        input_buffers: List[np.ndarray] = []
        for meta, inp in zip(input_metas, inputs):
            assert (
                meta.type not in self._mapping_to_numpy
                or inp.dtype == self._mapping_to_numpy[meta.type]
            ), (
                f"Unexpected type for input {meta.name!r}, "
                f"meta.type={meta.type!r}, inp.dtype={inp.dtype!r}"
            )
            tensor_type = self._mapping_to_onnx[inp.dtype]

            if tensor_type == TensorProto.STRING:
                io_binding.bind_cpu_input(meta.name, inp)
                continue

            sub_byte = self._mapping_sub_byte.get(meta.type)
            if sub_byte is not None:
                bits = sub_byte[0]
                packed = self._pack_sub_byte(inp, bits)
                input_buffers.append(packed)
                io_binding.bind_input(  # type: ignore[attr-defined]
                    meta.name,
                    "cpu",  # device_type
                    0,  # device_id
                    tensor_type,
                    list(inp.shape),
                    packed.ctypes.data,
                )
                continue

            io_binding.bind_input(  # type: ignore[attr-defined]
                meta.name,
                "cpu",  # device_type
                0,  # device_id
                tensor_type,
                list(inp.shape),
                inp.ctypes.data,
            )

        # Bind outputs. Sub-byte outputs are bound to raw packed buffers because
        # ORT cannot convert them to NumPy directly.
        sub_byte_outputs: dict = {}
        for index, meta in enumerate(output_metas):
            sub_byte = self._mapping_sub_byte.get(meta.type)
            if sub_byte is not None:
                bits, signed, tensor_type, numpy_dtype_name = sub_byte
                shape = tuple(meta.shape)
                assert shape and all(
                    isinstance(d, int) for d in shape
                ), f"Sub-byte output {meta.name!r} requires a static shape, got {shape!r}"
                n = int(np.prod(shape))
                buffer = np.zeros(self._packed_byte_count(n, bits), dtype=np.uint8)
                io_binding.bind_output(  # type: ignore[attr-defined]
                    meta.name,
                    "cpu",  # device_type
                    0,  # device_id
                    tensor_type,
                    list(shape),
                    buffer.ctypes.data,
                )
                sub_byte_outputs[index] = (shape, bits, signed, numpy_dtype_name, buffer)
            else:
                io_binding.bind_output(meta.name)

        # Run with IOBinding
        self._sess.run_with_iobinding(io_binding)

        # Get outputs
        ort_outputs = io_binding.get_outputs()
        outputs = []
        for index in range(len(output_metas)):
            if index in sub_byte_outputs:
                shape, bits, signed, numpy_dtype_name, buffer = sub_byte_outputs[index]
                outputs.append(
                    self._unpack_sub_byte(buffer, shape, bits, signed, numpy_dtype_name)
                )
            else:
                outputs.append(ort_outputs[index].numpy())
        return outputs

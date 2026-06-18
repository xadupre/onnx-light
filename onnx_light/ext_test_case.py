import os
import shutil
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from importlib import import_module
from io import StringIO
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy
from numpy.testing import assert_allclose


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

    def assertEqualArray(
        self,
        expected: numpy.ndarray,
        value: numpy.ndarray,
        atol: float = 0,
        rtol: float = 0,
        msg: Optional[str] = None,
    ):
        self.assertEqual(expected.dtype, value.dtype)
        self.assertEqual(expected.shape, value.shape)
        if msg:
            try:
                assert_allclose(expected, value, atol=atol, rtol=rtol)
            except AssertionError as e:
                raise AssertionError(msg) from e
        else:
            assert_allclose(expected, value, atol=atol, rtol=rtol)

    def assertAlmostEqual(  # type: ignore
        self, expected: numpy.ndarray, value: numpy.ndarray, atol: float = 0, rtol: float = 0
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        self.assertEqualArray(expected, value, atol=atol, rtol=rtol)

    def assertNotAlmostEqual(  # type: ignore
        self, expected: numpy.ndarray, value: numpy.ndarray, atol: float = 0, rtol: float = 0
    ):
        if not isinstance(expected, numpy.ndarray):
            expected = numpy.array(expected)
        if not isinstance(value, numpy.ndarray):
            value = numpy.array(value).astype(expected.dtype)
        try:
            self.assertEqualArray(expected, value, atol=atol, rtol=rtol)
            raise AssertionError("Arrays are equal.")
        except AssertionError:
            pass

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

    def assertNotEmpty(self, value: Any):
        if value is None:
            raise AssertionError(f"value is empty: {value!r}.")
        if isinstance(value, (list, dict, tuple, set)):
            if not value:
                raise AssertionError(f"value is empty: {value!r}.")

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
    """

    def __init__(self, model_bytes: bytes, providers: Optional[List[str]] = None):
        """
        Creates an inference session.

        Args:
            model_bytes: Serialized ONNX model
            providers: List of execution providers (defaults to ["CPUExecutionProvider"])
        """
        import onnxruntime as ort

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(model_bytes, providers=providers)

    def _get_numpy_dtype_to_onnx_mapping(self):
        """Returns mapping from NumPy dtype to ONNX TensorProto data type."""
        from onnx_light.onnx import TensorProto

        return {
            numpy.dtype("float32"): TensorProto.FLOAT,
            numpy.dtype("uint8"): TensorProto.UINT8,
            numpy.dtype("int8"): TensorProto.INT8,
            numpy.dtype("uint16"): TensorProto.UINT16,
            numpy.dtype("int16"): TensorProto.INT16,
            numpy.dtype("int32"): TensorProto.INT32,
            numpy.dtype("int64"): TensorProto.INT64,
            numpy.dtype("bool"): TensorProto.BOOL,
            numpy.dtype("float64"): TensorProto.DOUBLE,
            numpy.dtype("uint32"): TensorProto.UINT32,
            numpy.dtype("uint64"): TensorProto.UINT64,
        }

    def _get_special_dtype_mappings(self):
        """
        Returns mapping for special dtypes that need IOBinding.

        Maps ONNX dtype to (numpy view dtype, onnx tensor element type).
        """
        from onnx_light.onnx import TensorProto

        return {
            TensorProto.FLOAT16: (numpy.dtype("uint16"), TensorProto.FLOAT16),
            TensorProto.BFLOAT16: (numpy.dtype("uint16"), TensorProto.BFLOAT16),
            TensorProto.FLOAT8E4M3FN: (numpy.dtype("uint8"), TensorProto.FLOAT8E4M3FN),
            TensorProto.FLOAT8E4M3FNUZ: (numpy.dtype("uint8"), TensorProto.FLOAT8E4M3FNUZ),
            TensorProto.FLOAT8E5M2: (numpy.dtype("uint8"), TensorProto.FLOAT8E5M2),
            TensorProto.FLOAT8E5M2FNUZ: (numpy.dtype("uint8"), TensorProto.FLOAT8E5M2FNUZ),
            TensorProto.FLOAT8E8M0: (numpy.dtype("uint8"), TensorProto.FLOAT8E8M0),
            TensorProto.INT4: (numpy.dtype("uint8"), TensorProto.INT4),
            TensorProto.INT2: (numpy.dtype("uint8"), TensorProto.INT2),
            TensorProto.UINT4: (numpy.dtype("uint8"), TensorProto.UINT4),
            TensorProto.UINT2: (numpy.dtype("uint8"), TensorProto.UINT2),
            TensorProto.FLOAT4E2M1: (numpy.dtype("uint8"), TensorProto.FLOAT4E2M1),
        }

    def _get_onnx_tensor_element_type_from_array(self, arr: numpy.ndarray) -> Optional[int]:
        """
        Gets the ONNX tensor element type from a numpy array.

        For standard dtypes, uses the dtype directly.
        For special dtypes (FLOAT16, BFLOAT16, FLOAT8, etc.), infers from the
        array's dtype attribute.
        """
        from onnx_light.onnx import TensorProto

        special_mappings = self._get_special_dtype_mappings()

        # Check if array has ONNX dtype annotation (used by ml_dtypes)
        if hasattr(arr.dtype, "num"):
            # ml_dtypes assigns special dtype numbers
            if arr.dtype.num in special_mappings:
                return special_mappings[arr.dtype.num][1]

        # Check standard dtypes
        numpy_mapping = self._get_numpy_dtype_to_onnx_mapping()
        if arr.dtype in numpy_mapping:
            return numpy_mapping[arr.dtype]

        # Try to detect from dtype name for ml_dtypes
        dtype_name = arr.dtype.name.lower()
        if "float16" in dtype_name or "half" in dtype_name:
            return TensorProto.FLOAT16
        elif "bfloat16" in dtype_name:
            return TensorProto.BFLOAT16
        elif "float8e4m3fn" in dtype_name and "uz" not in dtype_name:
            return TensorProto.FLOAT8E4M3FN
        elif "float8e4m3fnuz" in dtype_name:
            return TensorProto.FLOAT8E4M3FNUZ
        elif "float8e5m2" in dtype_name and "uz" not in dtype_name:
            return TensorProto.FLOAT8E5M2
        elif "float8e5m2fnuz" in dtype_name:
            return TensorProto.FLOAT8E5M2FNUZ
        elif "float8e8m0" in dtype_name:
            return TensorProto.FLOAT8E8M0
        elif "int4" in dtype_name:
            return TensorProto.INT4
        elif "int2" in dtype_name:
            return TensorProto.INT2
        elif "uint4" in dtype_name:
            return TensorProto.UINT4
        elif "uint2" in dtype_name:
            return TensorProto.UINT2
        elif "float4e2m1" in dtype_name:
            return TensorProto.FLOAT4E2M1

        return None

    def run(
        self, output_names: Optional[List[str]], input_feed: dict[str, numpy.ndarray]
    ) -> List[numpy.ndarray]:
        """
        Runs the model with support for all ONNX dtypes.

        Uses IOBinding for special dtypes that ORT doesn't natively support.

        Args:
            output_names: Names of outputs to compute (None = all outputs)
            input_feed: Dictionary mapping input names to numpy arrays

        Returns:
            List of output arrays from the model
        """
        input_metas = self._sess.get_inputs()
        output_metas = self._sess.get_outputs()
        inputs = [input_feed[meta.name] for meta in input_metas]

        # Check if we need special dtype handling
        special_mappings = self._get_special_dtype_mappings()
        needs_iobinding = False
        for inp in inputs:
            onnx_dtype = self._get_onnx_tensor_element_type_from_array(inp)
            if onnx_dtype and onnx_dtype in special_mappings:
                needs_iobinding = True
                break

        # Use IOBinding for special dtypes, standard run otherwise
        if needs_iobinding:
            io_binding = self._sess.io_binding()

            for meta, inp in zip(input_metas, inputs):
                onnx_dtype = self._get_onnx_tensor_element_type_from_array(inp)

                if onnx_dtype and onnx_dtype in special_mappings:
                    # Use raw buffer binding for special dtypes
                    view_dtype, tensor_type = special_mappings[onnx_dtype]

                    # View the array as the compatible dtype for buffer access
                    buffer_view = inp.view(view_dtype)

                    # Bind the input directly with explicit element type
                    io_binding.bind_input(  # type: ignore[attr-defined]
                        meta.name,
                        "cpu",  # device_type
                        0,  # device_id
                        tensor_type,
                        list(inp.shape),
                        buffer_view.ctypes.data,
                    )
                else:
                    # Standard dtype, use normal binding
                    io_binding.bind_cpu_input(meta.name, inp)

            # Bind outputs
            for meta in output_metas:
                io_binding.bind_output(meta.name)

            # Run with IOBinding
            self._sess.run_with_iobinding(io_binding)

            # Get outputs
            outputs = io_binding.get_outputs()
            return [out.numpy() for out in outputs]
        else:
            # Standard path for normal dtypes
            return self._sess.run(output_names, input_feed)  # type: ignore[return-value]

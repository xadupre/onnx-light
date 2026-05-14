import os
import sys
import unittest
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy
from numpy.testing import assert_allclose


def is_windows() -> bool:
    return sys.platform == "win32"


def is_apple() -> bool:
    return sys.platform == "darwin"


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

    def dump_onnx(self, name: str, proto: Any, folder: Optional[str] = None) -> str:
        """Dumps an onnx file."""
        fullname = self.get_dump_file(name, folder=folder)
        with open(fullname, "wb") as f:
            f.write(proto.SerializeToString())
        return fullname

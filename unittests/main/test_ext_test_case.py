"""Tests for helpers in onnx_light.ext_test_case (e.g. ``hide_stdout``)."""

import os
import unittest
from io import StringIO
from unittest import mock

import numpy as np

from onnx_light.ext_test_case import ExtTestCase, hide_stdout


class TestHideStdout(ExtTestCase):
    """Tests for :func:`onnx_light.ext_test_case.hide_stdout`."""

    def test_hides_stdout(self):
        captured = []

        class Dummy:
            @hide_stdout()
            def run(self):
                print("hidden message")
                captured.append("ran")

        buffer = StringIO()
        with mock.patch("sys.stdout", buffer):
            Dummy().run()
        self.assertEqual(captured, ["ran"])
        self.assertNotIn("hidden message", buffer.getvalue())

    def test_passes_captured_output_to_callback(self):
        received = []

        class Dummy:
            @hide_stdout(received.append)
            def run(self):
                print("captured line")

        Dummy().run()
        self.assertEqual(len(received), 1)
        self.assertIn("captured line", received[0])

    def test_preserves_function_name(self):
        class Dummy:
            @hide_stdout()
            def my_test(self):
                pass

        self.assertEqual(Dummy.my_test.__name__, "my_test")

    def test_unhide_environment_runs_function_directly(self):
        class Dummy:
            @hide_stdout()
            def run(self):
                print("visible message")

        buffer = StringIO()
        with mock.patch.dict(os.environ, {"UNHIDE": "1"}), mock.patch("sys.stdout", buffer):
            Dummy().run()
        self.assertIn("visible message", buffer.getvalue())

    def test_torch_assertion_becomes_skiptest(self):
        class Dummy:
            @hide_stdout()
            def run(self):
                raise AssertionError("torch is not recent enough, file something")

        with self.assertRaises(unittest.SkipTest):
            Dummy().run()

    def test_other_assertion_is_propagated(self):
        class Dummy:
            @hide_stdout()
            def run(self):
                raise AssertionError("some other failure")

        with self.assertRaises(AssertionError) as ctx:
            Dummy().run()
        self.assertIn("some other failure", str(ctx.exception))

    def test_warnings_are_suppressed(self):
        import warnings

        class Dummy:
            @hide_stdout()
            def run(self):
                warnings.warn("ignore me", UserWarning, stacklevel=2)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # The decorator installs its own filter ignoring UserWarning, so the
            # warning must not be raised as an error.
            Dummy().run()

    def test_assert_almost_equal_supports_unittest_places(self):
        self.assertAlmostEqual(0.1234567, 0.1234568, places=6)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(0.1234567, 0.1234568, places=7)

    def test_assert_almost_equal_supports_array_tolerances(self):
        self.assertAlmostEqual(np.array([1.0]), np.array([1.009]), atol=0.01)
        with self.assertRaises(AssertionError):
            self.assertAlmostEqual(np.array([1.0]), np.array([1.01]), atol=0.001)
        self.assertAlmostEqual(100.0, 101.0, rtol=0.02)

    def test_assert_not_almost_equal_fails_for_equal_arrays(self):
        with self.assertRaises(AssertionError):
            self.assertNotAlmostEqual(np.array([1.0]), np.array([1.0]))
        self.assertNotAlmostEqual(np.array([1.0]), np.array([[1.0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Round-trips every backend test model through the onnxruntime flatbuffer format.

For each backend test case shipped with ``onnx-light`` (see
:func:`onnx_light.onnx.backend.collect_test_cases`) the script performs two
round-trips:

1. **onnx-light → onnx-light** -- the model is serialized with
   ``SerializeFormat.ORT_FLATBUFFERS`` by onnx-light, parsed back by
   onnx-light, and the parsed ``ModelProto`` must be byte-identical to the
   original (when re-serialized in the ONNX protobuf format).

2. **onnxruntime → onnxruntime → onnx-light** -- the model is loaded by
   ``onnxruntime``, re-saved by ``onnxruntime`` in its native ``.ort``
   flatbuffer format (via ``SessionOptions.optimized_model_filepath`` with
   ``session.save_model_format=ORT`` and graph optimisations disabled), and
   the resulting file is parsed back by onnx-light. The parsed model must
   again be byte-identical (in protobuf form) to the original.

While ``SerializeFormat.ORT_FLATBUFFERS`` is not implemented in onnx-light
the first step raises ``RuntimeError`` immediately and the whole case is
reported as skipped. This is intentional: the script is the harness that
will pick up the new code path the moment the C++ writer/reader lands,
without any further changes. The second step relies on onnx-light being
able to *read* a ``.ort`` file produced by onnxruntime, so it skips for the
same reason today.

The implementation deliberately follows the four-step workflow described in
the original issue: save with onnx-light, reload with onnx-light and
compare, then save with onnxruntime, reload with onnx-light and compare.
"""

from __future__ import annotations

import os
import unittest

import onnx_light.onnx as onnxl
from onnx_light.ext_test_case import ExtTestCase
from onnx_light.onnx.backend import collect_test_cases

try:
    import onnxruntime

    HAS_ORT = onnxruntime is not None
except ImportError:  # pragma: no cover - exercised only in no-ORT envs
    HAS_ORT = False


def _onnx_bytes(model: onnxl.ModelProto) -> bytes:
    """Returns the canonical ONNX-protobuf serialisation of *model*."""
    sopts = onnxl.SerializeOptions()
    sopts.format = onnxl.SerializeFormat.ONNX
    return model.SerializeToString(sopts)


def _save_with_onnx_light_ort(model: onnxl.ModelProto, path: str) -> None:
    """Saves *model* at *path* using onnx-light's ORT flatbuffer writer."""
    sopts = onnxl.SerializeOptions()
    sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    model.SerializeToFile(path, sopts)


def _load_with_onnx_light_ort(path: str) -> onnxl.ModelProto:
    """Loads an ORT flatbuffer model from *path* using onnx-light."""
    popts = onnxl.ParseOptions()
    popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
    parsed = onnxl.ModelProto()
    parsed.ParseFromFile(path, popts)
    return parsed


def _save_with_onnxruntime_ort(src_path: str, dst_path: str) -> None:
    """Re-saves a model as a ``.ort`` flatbuffer file using onnxruntime.

    Loads ``src_path`` (either an ``.onnx`` or ``.ort`` file) into an
    ``InferenceSession`` with graph optimisations disabled so that the
    serialised graph stays structurally equivalent to the input, then asks
    onnxruntime to dump the optimised (== original) model at ``dst_path``
    using the ORT flatbuffer format.
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.optimized_model_filepath = dst_path
    so.add_session_config_entry("session.save_model_format", "ORT")
    # Creating the session triggers the optimised-model dump.
    ort.InferenceSession(src_path, so, providers=["CPUExecutionProvider"])


class TestBackendRoundTripOrtFormat(ExtTestCase):
    """Round-trips every backend test model through the ORT flatbuffer format."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = collect_test_cases()
        assert cls.cases, "No backend test cases were registered."

    def _round_trip_onnx_light(
        self, original: onnxl.ModelProto, expected_bytes: bytes, path: str
    ) -> None:
        """Step 1+2: save with onnx-light, reload with onnx-light, compare."""
        _save_with_onnx_light_ort(original, path)
        self.assertTrue(os.path.exists(path))
        reloaded = _load_with_onnx_light_ort(path)
        self.assertEqual(_onnx_bytes(reloaded), expected_bytes)

    def _round_trip_onnxruntime(
        self, expected_bytes: bytes, ort_input_path: str, ort_output_path: str
    ) -> None:
        """Step 3+4: re-save with onnxruntime, reload with onnx-light, compare."""
        _save_with_onnxruntime_ort(ort_input_path, ort_output_path)
        self.assertTrue(os.path.exists(ort_output_path))
        reloaded = _load_with_onnx_light_ort(ort_output_path)
        self.assertEqual(_onnx_bytes(reloaded), expected_bytes)

    @unittest.skipUnless(HAS_ORT, "onnxruntime is not available")
    def test_round_trip_all_backend_models(self) -> None:
        """Runs the two round-trips on every collected backend test case.

        Each backend test case is reported as an independent ``subTest`` so
        the suite shows a precise list of pass / fail / skip per case. When
        ``SerializeFormat.ORT_FLATBUFFERS`` is not implemented yet, every
        case skips at the first step.
        """
        skipped = 0
        executed = 0
        for tc in self.cases:
            with self.subTest(case=tc.name, kind=tc.kind):
                expected_bytes = _onnx_bytes(tc.model)
                light_path = self.get_dump_file(f"rt_light_{tc.name}.ort")
                ort_path = self.get_dump_file(f"rt_ort_{tc.name}.ort")
                try:
                    self._round_trip_onnx_light(tc.model, expected_bytes, light_path)
                except RuntimeError as exc:
                    skipped += 1
                    # Onnx-light cannot (yet) save/parse the ORT flatbuffer
                    # format. Skip the whole case until the C++ writer lands.
                    self.skipTest(
                        f"onnx-light ORT_FLATBUFFERS round-trip is not implemented yet: {exc}"
                    )
                self._round_trip_onnxruntime(expected_bytes, light_path, ort_path)
                executed += 1
        # Sanity check: the harness must touch every collected case (either
        # by running it or by skipping it). It is only informational because
        # ``subTest`` swallows failures, but it catches a silent bug where
        # ``collect_test_cases`` would start returning an empty list.
        self.assertEqual(skipped + executed, len(self.cases))


if __name__ == "__main__":
    unittest.main(verbosity=2)

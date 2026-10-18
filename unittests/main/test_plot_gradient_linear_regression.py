import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import onnx_light.onnx.numpy_helper as onh
from onnx_light.ext_test_case import import_or_skip

import_or_skip("onnx_light.onnx_core.gradient")
runtime = import_or_skip("onnx_light.onnx_py._onnxpykernels.runtime")

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "examples"
    / "gradient"
    / "plot_gradient_linear_regression.py"
)


def test_gradient_linear_regression(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    try:
        example = runpy.run_path(str(SOURCE))
        losses = example["loss_history"]
        assert len(losses) == example["n_epochs"]
        assert losses[-1] < 0.01
        assert losses[-1] < losses[0] / 100
        assert isinstance(example["forward_session"], runtime.RuntimeSession)
        assert isinstance(example["backward_session"], runtime.RuntimeSession)
        assert set(example["forward_session"].used_kernels()) == {
            "ai.onnx:MatMul",
            "ai.onnx:Add",
            "ai.onnx:Sub",
            "ai.onnx:Mul",
            "ai.onnx:ReduceMean",
        }
        assert {"ai.onnx:MatMul", "ai.onnx:ReduceSum", "ai.onnx:Mul", "ai.onnx:Sub"} <= set(
            example["backward_session"].used_kernels()
        )
        np.testing.assert_allclose(example["W"], example["W_true"], atol=0.02)
        np.testing.assert_allclose(example["b"], example["b_true"], atol=0.02)
        assert np.ndim(example["b"]) == 0
        assert (tmp_path / "plot_gradient_linear_regression.png").is_file()
        assert (tmp_path / "plot_gradient_linear_regression_scatter.png").is_file()
        np.testing.assert_allclose(
            example["y_final"],
            example["X_data"] @ example["W"] + example["b"],
            rtol=1e-5,
            atol=1e-6,
        )

        rng = np.random.default_rng(0)
        for count in (1, 7):
            x = rng.standard_normal((count, 4)).astype(np.float32)
            w = rng.standard_normal((4, 1)).astype(np.float32)
            b = np.array(0.5, dtype=np.float32)
            target = rng.standard_normal((count, 1)).astype(np.float32)
            context = runtime.RuntimeContext(release_intermediates=True)
            for name, value in {"X": x, "W": w, "b": b, "y_true": target}.items():
                context.set(name, runtime.tensor_from_proto(onh.from_array(value)))
            for name, value in {"dy": 1.0, "learning_rate": example["learning_rate"]}.items():
                context.set(
                    name,
                    runtime.tensor_from_proto(onh.from_array(np.array(value, dtype=np.float32))),
                )

            # Reuses sessions and the context with new parameters and saved values.
            for _ in range(3):
                prediction = x @ w + b
                residual = prediction - target
                dy = (2.0 / count) * residual
                expected_grad_w = x.T @ dy
                expected_grad_b = np.sum(dy)
                example["forward_session"].run(context)
                np.testing.assert_allclose(
                    np.from_dlpack(context.get("y_pred")), prediction, rtol=1e-5, atol=1e-6
                )
                np.testing.assert_allclose(
                    np.from_dlpack(context.get("loss")),
                    np.mean(residual**2),
                    rtol=1e-5,
                    atol=1e-6,
                )
                assert context.has("residual") and context.has("squared")
                assert not context.has("mm")
                example["backward_session"].run(context)
                for name, expected in {
                    "grad_W": expected_grad_w,
                    "grad_b": dy,
                    "grad_b_scalar": expected_grad_b,
                }.items():
                    np.testing.assert_allclose(
                        np.from_dlpack(context.get(name)), expected, rtol=1e-5, atol=1e-6
                    )
                w = w - example["learning_rate"] * expected_grad_w
                b = b - example["learning_rate"] * expected_grad_b
                for name, expected in {"W": w, "b": b}.items():
                    updated = np.from_dlpack(context.get(f"{name}_next")).copy()
                    np.testing.assert_allclose(updated, expected, rtol=1e-5, atol=1e-6)
                    assert updated.shape == expected.shape
                    assert updated.dtype == np.float32
                    context.put(name, context.get(f"{name}_next"))
                assert not context.has("step_W")
                assert not context.has("step_b")
    finally:
        plt.close("all")

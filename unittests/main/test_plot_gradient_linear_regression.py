import runpy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from onnx_light.ext_test_case import import_or_skip

import_or_skip("onnx_light.onnx_core.gradient")
import_or_skip("onnx_light.onnx.reference")

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
        assert "ai.onnx:MatMul" in example["grad_evaluator"].used_kernels()
        np.testing.assert_allclose(example["W"], example["W_true"], atol=0.02)
        np.testing.assert_allclose(example["b"], example["b_true"], atol=0.02)
        assert np.ndim(example["b"]) == 0
        assert (tmp_path / "plot_gradient_linear_regression.png").is_file()
        assert (tmp_path / "plot_gradient_linear_regression_scatter.png").is_file()

        rng = np.random.default_rng(0)
        for count in (1, 7):
            x = rng.standard_normal((count, 4)).astype(np.float32)
            w = rng.standard_normal((4, 1)).astype(np.float32)
            b = np.float32(0.5)
            target = rng.standard_normal((count, 1)).astype(np.float32)
            dy = (2.0 / count) * (x @ w + b - target)
            grad_w, grad_b_samples = example["grad_evaluator"].run(
                None, {"W": w, "b": b, "X": x, "dy": dy}
            )
            np.testing.assert_allclose(grad_w, x.T @ dy, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(grad_b_samples, dy, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(np.sum(grad_b_samples), np.sum(dy), rtol=1e-5)
    finally:
        plt.close("all")

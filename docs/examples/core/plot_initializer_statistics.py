"""
.. _l-example-plot-initializer-statistics:

Statistics on the weights of an ONNX model
===========================================

This example walks through every initializer of an ONNX model whose rank
is greater than two (typically convolution or attention weights) and
computes a handful of descriptive statistics for each of them:

* **min** / **max** — the extreme values stored in the tensor,
* **median** — the middle value once the weights are sorted,
* **mean** — the arithmetic average,
* **distance to normal law** — a Kolmogorov-Smirnov style distance
  between the empirical distribution of the weights and the normal law
  :math:`\\mathcal{N}(\\mu, \\sigma^2)` fitted on the same weights.  A
  value close to ``0`` means the weights are well described by a normal
  distribution; larger values indicate a stronger departure from
  normality.

By default the script builds a small dummy model with a couple of 3D and
4D initializers so that it runs out of the box.  An existing model can be
analysed instead by passing its path on the command line::

    python plot_initializer_statistics.py --model model.onnx

The initializer values are read through
:func:`onnx_light.onnx.numpy_helper.to_array`, which relies on the
``ml_dtypes`` fallback mechanism to materialise tensors stored with dtypes
that have no native NumPy equivalent (``float16``, ``bfloat16``,
``float8`` ...).  The statistics are always computed in ``float64`` so the
same code path works for every element type.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

import onnx_light.onnx as onnxl
import onnx_light.onnx.helper as oh
import onnx_light.onnx.numpy_helper as onh
from onnx_light.onnx import load

# %%
# Command line
# ------------
#
# ``--model`` is optional: when omitted a dummy model is generated so the
# example is self-contained.


def parse_args() -> argparse.Namespace:
    """Parses the command-line arguments.

    Returns:
        The parsed arguments with a single ``model`` attribute holding the
        path to an ONNX model, or ``None`` when the dummy model must be
        generated.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Path to an existing ONNX model to analyse. "
            "When omitted, a dummy model is generated."
        ),
    )
    parsed, _ = parser.parse_known_args()
    return parsed


# %%
# Dummy model
# -----------
#
# The generated model only exists to carry a few initializers whose rank
# is greater than two.  The graph itself does not need to be runnable for
# the statistics to be computed.


def make_dummy_model() -> onnxl.ModelProto:
    """Returns a dummy model with a few initializers of rank greater than two.

    Returns:
        An :class:`onnx_light.onnx.ModelProto` carrying several 3D and 4D
        initializers alongside a 1D bias to illustrate the rank filtering.
    """

    rng = np.random.default_rng(0)
    initializers = [
        # A convolution-like weight drawn from a normal law.
        onh.from_array(rng.standard_normal((8, 4, 3, 3)).astype(np.float32), name="conv_w"),
        # A weight drawn from a uniform law, further from normality.
        onh.from_array(rng.uniform(-1.0, 1.0, (16, 8, 3)).astype(np.float32), name="uniform_w"),
        # A half-precision weight to exercise the ml_dtypes fallback.
        onh.from_array(
            rng.standard_normal((4, 4, 4)).astype(np.float16), name="fp16_w"
        ),
        # A 1D bias, ignored because its rank is not greater than two.
        onh.from_array(rng.standard_normal((8,)).astype(np.float32), name="bias"),
    ]
    graph = oh.make_graph([], "dummy_stats_graph", [], [], initializer=initializers)
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 18)], ir_version=9)


# %%
# Statistics
# ----------
#
# The distance to the normal law is the Kolmogorov-Smirnov statistic: the
# largest absolute gap between the empirical cumulative distribution of the
# weights and the cumulative distribution of the normal law fitted on those
# same weights.  It is computed without any extra dependency by evaluating
# the normal CDF through :func:`math.erf`.


def distance_to_normal_law(values: np.ndarray) -> float:
    """Returns the Kolmogorov-Smirnov distance to a fitted normal law.

    The normal law :math:`\\mathcal{N}(\\mu, \\sigma^2)` is fitted on
    ``values`` using their empirical mean and standard deviation, and the
    returned value is the maximum absolute difference between the empirical
    cumulative distribution function and the normal one.

    Args:
        values: The tensor values as a flat array.

    Returns:
        The distance in ``[0, 1]``; ``0.0`` for a degenerate distribution
        with a null standard deviation.
    """

    flat = np.sort(np.asarray(values, dtype=np.float64).ravel())
    n = flat.size
    if n == 0:
        return 0.0
    mean = float(flat.mean())
    std = float(flat.std())
    if std == 0.0:
        return 0.0

    # Normal cumulative distribution function evaluated at every value.
    scaled = (flat - mean) / (std * math.sqrt(2.0))
    normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(scaled))

    # Empirical cumulative distribution function bracketing each value.
    upper = np.arange(1, n + 1, dtype=np.float64) / n
    lower = np.arange(0, n, dtype=np.float64) / n
    return float(np.maximum(np.abs(upper - normal_cdf), np.abs(normal_cdf - lower)).max())


def compute_statistics(tensor: onnxl.TensorProto) -> dict[str, float]:
    """Returns the descriptive statistics of an initializer.

    Args:
        tensor: The initializer to analyse.

    Returns:
        A mapping with the ``min``, ``max``, ``median``, ``mean`` and
        ``normal_distance`` of the tensor values.
    """

    values = onh.to_array(tensor).astype(np.float64)
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "normal_distance": distance_to_normal_law(values),
    }


# %%
# Main
# ----
#
# The model is loaded (or generated), then every initializer with a rank
# greater than two is analysed and its statistics printed.

args = parse_args()

if args.model:
    print(f"Loading model {args.model!r}.")
    model = load(args.model)
else:
    print("No model provided, generating a dummy model.")
    model = make_dummy_model()

print()
header = f"{'name':<16} {'rank':>4} {'min':>10} {'max':>10} {'median':>10} {'mean':>10} {'normal':>10}"
print(header)
print("-" * len(header))

n_analysed = 0
for init in model.graph.initializer:
    rank = len(init.dims)
    if rank <= 2:
        continue
    n_analysed += 1
    stats = compute_statistics(init)
    print(
        f"{init.name:<16} {rank:>4} "
        f"{stats['min']:>10.4f} {stats['max']:>10.4f} "
        f"{stats['median']:>10.4f} {stats['mean']:>10.4f} "
        f"{stats['normal_distance']:>10.4f}"
    )

print()
print(f"Analysed {n_analysed} initializer(s) with more than two dimensions.")

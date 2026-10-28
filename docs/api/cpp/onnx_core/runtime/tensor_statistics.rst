tensor_statistics.h
===================

.. doxygenfile:: onnx_core/runtime/kernels/tensor_statistics.h
   :project: onnx-light

Exact statistics on large tensors
--------------------------------

Quantiles retain linear interpolation at ``q * (count - 1)`` and the documented
NaN/infinity behavior. The implementation deduplicates required ranks, including
the median when Q2 is already requested.

For FLOAT tensors with at least 32768 elements and at most 128 unique requested
ranks, it locates those ranks in the histogram already computed for the result.
Only bins containing required ranks are refined, using three distributions over
the order-preserving FLOAT bit keys (12, 10 and 10 bits). The final keys identify
the exact stored values, not bin-center approximations. Infinities are accounted
for separately, and signed zeros remain in the same initial histogram bin.
Private histogram counters are bounded by 8 MiB independently of tensor size.
No tensor-sized selection copy is allocated, even for concentrated distributions.
Constant FLOAT data needs no refinement.

Other inputs recursively partition only relevant ranges with
``std::nth_element``. For ``K`` requested ranks this takes average
``O(N log(K + 1))`` work rather than sorting all ``N`` values. This fallback uses
four bytes per FLOAT element, eight per DOUBLE or 64-bit integer element, and
DOUBLE for other real types. No selection workspace is allocated without a
quantile/median request or when the input contains NaNs.

Rectangle reduction reads the original tensor directly. The histogram reuses
the summary's finite bounds instead of scanning the input again. Input storage
remains unchanged.

Progress and cancellation
-------------------------

The original overloads remain available, including their existing binary
symbols. New overloads accept a trailing ``TensorStatisticsProgressCallback``
after the output budget (and after the catalogue for encoded values):

.. code-block:: cpp

    namespace rt = onnx_light::core::runtime;
    const auto stats = rt::ComputeTensorStatistics(
        tensor, 0, {0.25, 0.5, 0.75}, true, {32, 32},
        rt::kTensorStatisticsMaxOutputBytes,
        [&](rt::TensorStatisticsPhase phase, int64_t completed, int64_t total) {
            display_phase_progress(phase, completed, total);
            return !cancel_requested();
        });

Callbacks execute synchronously on the calling thread, never concurrently on
worker threads. Decoding, summary, mean, distinct counting, histogram and rectangle
report their start and end as ``0/1`` and ``1/1``. Unneeded phases are omitted.
Large-FLOAT quantiles report ``0/3`` and one update after each refinement pass,
or ``0/1`` and ``1/1`` when no refinement is needed. The selection fallback
reports ``0/K`` before copying the input, then one update per selected unique
rank. Completion reports ``1/1``; an empty tensor reports only completion.
These counts describe each phase, not global progress or elapsed-time estimates.

Returning false throws ``std::runtime_error`` with
``Tensor statistics cancelled.``; exceptions from callbacks propagate unchanged.
Cancellation is checked between phases and selections, not inside an individual
decode, reduction, refinement pass or ``nth_element``. No partial result is returned. The callback
must not mutate the input tensor. Option and output-budget validation precede
callbacks and decoding. The Python API continues to use the callback-free path.

include_math_kernels.h
=============================

.. doxygenfile:: onnx_extensions/kernels/kernels/math/include_math_kernels.h
   :project: onnx-light

``LogCumSumExp`` (opset 29) computes a stable inclusive forward logsumexp
along the ``axis`` attribute (default 0, negative axes supported). It preserves
the input shape and type. FLOAT16, BFLOAT16 and FLOAT accumulate in float;
DOUBLE accumulates in double without rounding intermediate prefixes to the
output storage type. NaN propagates ahead of positive infinity, and all-negative
infinity prefixes remain negative infinity. Empty tensors are supported;
rank and axis validation still applies.

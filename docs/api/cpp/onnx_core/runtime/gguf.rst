gguf.h
======

See :ref:`the exact block support matrix <l-gguf-blocks>` for the pinned
layouts and the boundary between payload and checkpoint/container support.

.. doxygenfile:: onnx_core/runtime/gguf.h
    :project: onnx-light

The standard-domain ``DequantizePacked`` raw-byte decoder is independent of
the structured GGUF import API. It supports all 27 fixed profiles, zero-sized
outputs and nonfinite scale values, with FP32 arithmetic before conversion.

.. doxygenfile:: onnx_core/runtime/dequantize_packed.h
    :project: onnx-light

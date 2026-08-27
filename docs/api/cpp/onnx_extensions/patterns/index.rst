onnx_patterns
=============

This module documents the C++ optimization-pattern headers used by
``lib_onnx_patterns``.

.. rubric:: Core

.. doxygenfile:: onnx_extensions/patterns/dispatch_table.h

.. rubric:: Algebra

.. doxygenfile:: onnx_extensions/patterns/algebra/common_pattern.h
.. doxygenfile:: onnx_extensions/patterns/algebra/mul_pattern.h
.. doxygenfile:: onnx_extensions/patterns/algebra/range_pattern.h
.. doxygenfile:: onnx_extensions/patterns/algebra/reduce_pattern.h
.. doxygenfile:: onnx_extensions/patterns/algebra/shape_pattern.h
.. doxygenfile:: onnx_extensions/patterns/algebra/sub_pattern.h

.. rubric:: Attention

.. doxygenfile:: onnx_extensions/patterns/attention/attention_pattern.h

.. rubric:: Canonicalization

.. doxygenfile:: onnx_extensions/patterns/canonicalization/cast_pattern.h
.. doxygenfile:: onnx_extensions/patterns/canonicalization/clip_pattern.h
.. doxygenfile:: onnx_extensions/patterns/canonicalization/constant_pattern.h
.. doxygenfile:: onnx_extensions/patterns/canonicalization/conv_pattern.h
.. doxygenfile:: onnx_extensions/patterns/canonicalization/dropout_pattern.h
.. doxygenfile:: onnx_extensions/patterns/canonicalization/identity_pattern.h
.. doxygenfile:: onnx_extensions/patterns/canonicalization/not_pattern.h

.. rubric:: Collections

.. doxygenfile:: onnx_extensions/patterns/collections/collections_utils.h
.. doxygenfile:: onnx_extensions/patterns/collections/concat_pattern.h
.. doxygenfile:: onnx_extensions/patterns/collections/gather_pattern.h
.. doxygenfile:: onnx_extensions/patterns/collections/sequence_pattern.h
.. doxygenfile:: onnx_extensions/patterns/collections/shape_pattern.h
.. doxygenfile:: onnx_extensions/patterns/collections/slice_pattern.h
.. doxygenfile:: onnx_extensions/patterns/collections/split_pattern.h

.. rubric:: Expand

.. doxygenfile:: onnx_extensions/patterns/expand/expand_pattern.h
.. doxygenfile:: onnx_extensions/patterns/expand/where_pattern.h

.. rubric:: Layout

.. doxygenfile:: onnx_extensions/patterns/layout/layout_pattern.h

.. rubric:: MatMul

.. doxygenfile:: onnx_extensions/patterns/matmul/matmul_pattern.h

.. rubric:: Normalization

.. doxygenfile:: onnx_extensions/patterns/normalization/activation_pattern.h
.. doxygenfile:: onnx_extensions/patterns/normalization/normalization_pattern.h

.. rubric:: Reshape

.. doxygenfile:: onnx_extensions/patterns/reshape/reshape_pattern.h

.. rubric:: Traditional ML

.. doxygenfile:: onnx_extensions/patterns/traditionalml/tree_ensemble_pattern.h

.. rubric:: Transpose

.. doxygenfile:: onnx_extensions/patterns/transpose/transpose_pattern.h

.. rubric:: Unsqueeze

.. doxygenfile:: onnx_extensions/patterns/unsqueeze/unsqueeze_pattern.h

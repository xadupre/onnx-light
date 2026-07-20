onnx_shapes
==========

The ``onnx_shapes`` library runs shape inference over ONNX graphs. It
consumes the symbolic value descriptors defined in ``onnx_core``
(:cpp:class:`onnx_light::core::symbolic::SymDim`,
:cpp:class:`onnx_light::core::symbolic::SymShape` and
:cpp:class:`onnx_light::core::symbolic::SymTensor`), which together
describe a tensor whose shape may be fully known, fully symbolic, or
any mix in between. See :doc:`../onnx_core/sym_tensor` for those
types.

.. toctree::
    :maxdepth: 1

    dispatch_table
    shapes/index

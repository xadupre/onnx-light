onnx_manipulations
==================

Lightweight ``ModelProto`` manipulation helpers that are independent from the
ONNX operator schemas: the ONNX text parser / printer, attribute and tensor
proto construction helpers, the data-type name utilities and the graph
manipulation helpers. ``lib_onnx_lib`` depends publicly on this library.

.. toctree::
    :maxdepth: 1

    attr_proto_util
    compose
    data_type_utils
    parser
    printer
    tensor_proto_util
    tensor_util

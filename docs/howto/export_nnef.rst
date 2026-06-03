.. _l-howto-export-nnef:

Export a model to NNEF format
=============================

``onnx_light`` ships with a small, dependency-free exporter that
translates an ONNX :cpp:class:`ModelProto` into the
`Khronos NNEF v1.0 <https://www.khronos.org/nnef>`_ representation.

The exporter produces a directory containing:

* ``graph.nnef`` – a textual description of the graph;
* one ``<label>.dat`` file per initializer, encoded with the standard
  NNEF binary tensor format (128-byte header followed by raw data).

The implementation lives in the ``lib_onnx_nnef`` C++ library and is
exposed through the public C++ API
(:cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::ExportToNNEF`,
:cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::ToNNEFText`,
:cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::SaveNNEF`).

Quick start (C++)
-----------------

Load an ONNX model with the proto reader and call ``SaveNNEF``:

.. code-block:: cpp

    #include "nnef/exporter.h"
    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    #include <iostream>

    int main() {
      ONNX_LIGHT_NAMESPACE::ModelProto model;
      ONNX_LIGHT_NAMESPACE::utils::FileStream stream("model.onnx");
      ONNX_LIGHT_NAMESPACE::ParseOptions opts;
      ONNX_LIGHT_NAMESPACE::ParseModelProtoFromStream(model, stream, opts);

      // graph.nnef text only:
      std::cout << ONNX_LIGHT_NAMESPACE::nnef::ToNNEFText(model);

      // graph.nnef + one <label>.dat file per initializer:
      ONNX_LIGHT_NAMESPACE::nnef::SaveNNEF(model, "model_nnef");
    }

A complete standalone CMake project (build + run instructions) is
documented in :ref:`l-cpp-export-nnef-example`.

Supported operators
-------------------

The list of ONNX op types that have a builtin NNEF converter is returned
by :cpp:func:`ONNX_LIGHT_NAMESPACE::nnef::SupportedOps`.  It currently
covers the most common operators of vision-style networks: ``Conv``,
``BatchNormalization``, ``MaxPool``, ``AveragePool``,
``GlobalAveragePool``, ``GlobalMaxPool``, ``Relu``, ``Sigmoid``,
``Tanh``, ``Softmax``, ``Gemm``, ``MatMul``, ``Reshape``, ``Flatten``,
``Transpose``, ``Concat``, ``Add``/``Sub``/``Mul``/``Div``/``Pow``,
``Min``/``Max``, ``Identity`` and ``Clip``.

Extending the converter
-----------------------

If you need to export an operator that is not built in, register a
custom converter:

.. code-block:: cpp

    #include "nnef/exporter.h"

    using namespace ONNX_LIGHT_NAMESPACE::nnef;

    RegisterOpConverter("MyOp",
        [](ExportContext &ctx, const ONNX_LIGHT_NAMESPACE::NodeProto &,
           const std::map<std::string, AttributeValue> &,
           const std::vector<std::string> &inputs,
           const std::vector<std::string> &outputs) {
          // ``inputs`` and ``outputs`` are NNEF identifiers already mapped
          // from the ONNX names.
          ctx.AddStatement(outputs[0] + " = my_op(" + inputs[0] + ");");
        });

If the exporter encounters an op without a registered converter it
throws :cpp:class:`ONNX_LIGHT_NAMESPACE::nnef::NNEFExportError` with a
descriptive message.

Python bindings
---------------

The same C++ API is also reachable from Python via the
``onnx_light.nnef`` package (``save_nnef``, ``to_nnef_text``,
``export_to_nnef``, ``register_op_converter``, ``supported_ops``).  The
Python module is a thin re-export shim over the C++ implementation, so
both entry points share the converter registry.

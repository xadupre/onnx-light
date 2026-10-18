.. _l-howto-save-ort-flatbuffers-cpp:

:html_theme.sidebar_secondary.remove:

How to save and load a model in the ORT flatbuffer format
=========================================================

This guide covers the C++ interface. For the other language, see
:doc:`../python/save_ort_flatbuffers`.

`onnxruntime <https://onnxruntime.ai/>`_ defines a compact flatbuffer
serialization (``.ort``) that avoids the protobuf parsing step when loading
models in ONNX Runtime.

*onnx-light* exposes the format through
``onnx::SerializeFormat::kOrtFlatbuffers``.
The native C++ writer produces files with the ``ORTM`` identifier and ORT
format version ``4``, readable by a full :epkg:`onnxruntime` build. The
runtime resolves the graph and upgrades kernel metadata when loading it.
No ONNX Runtime conversion session is needed to serialize the model.
The native reader accepts both these version-4 files and current ORT
version-6 files.

Save with the native writer
---------------------------

For an in-memory model whose tensor payloads have been loaded, select
``ORT_FLATBUFFERS`` in the serialization options:


.. code-block:: cpp

    #include "onnx.h"
    #include "onnx_helper.h"
    #include "stream.h"

    onnx::SerializeOptions options;
    options.format = onnx::SerializeFormat::kOrtFlatbuffers;
    onnx::utils::FileWriteStream stream("model.ort");
    onnx::SerializeModelProtoToStream(model, stream, options);

Memory and file-descriptor output
---------------------------------

The Python ``SerializeToString`` and ``SerializeToFileDescriptor`` helpers
accept the same format:

See the :doc:`Python recipe </howto/python/save_ort_flatbuffers>`
for this example.

The Python recipe also shows how to load the result with ONNX Runtime:

See the :doc:`Python recipe </howto/python/save_ort_flatbuffers>`
for this example.

Read with the native reader
---------------------------

Set ``ParseOptions.format`` to ``onnx::SerializeFormat::kOrtFlatbuffers``
before calling ``ParseModelProtoFromStream`` to select native ORT decoding.

See the :doc:`Python recipe </howto/python/save_ort_flatbuffers>`
for this example.

The reader reconstructs an ONNX model for execution and reserialization,
not an exact copy of the original protobuf bytes. ORT may normalize the
graph, and original names or documentation may be unavailable. Validate
roundtrips through model behavior rather than protobuf byte equality.

The reader honors tensor-byte and recursion limits and invokes raw-data
and node callbacks. Decoding is sequential even when ``num_threads`` is
set. Tensor bytes are copied into owned storage even when ``no_copy`` is
set; native ORT reading is not a zero-copy path.

External tensor offsets are not supported by the reader. They are rejected
explicitly; the reader does not infer a sidecar filename or open an
unspecified external file.

Current writer limitations
--------------------------

* A full ONNX Runtime build is required to execute the generated version-4
  files directly; minimal runtime builds are not supported.
* Local model functions, node overloads, device configurations, and
  quantization annotations are not supported.
* Only model-level ``metadata_props`` are supported. Metadata properties on
  graphs, nodes (including ``Constant`` nodes), tensors, and value infos are
  rejected, including inside nested graphs and tensor attributes.
* Optional, sparse, opaque, and structured value types, complex tensors,
  sparse initializers, onnx-light's ``GraphProto.encoded_initializer``
  extension, and ``GRAPHS``, sparse-tensor, or type-proto attributes are not
  supported.
* Tensor element types newer than ``FLOAT8E5M2FNUZ`` (20) are not supported:
  ``UINT4``, ``INT4``, ``FLOAT4E2M1``, ``FLOAT8E8M0``, ``UINT2``, and ``INT2``.
* Intermediate tensor types are inferred for common standard operators.
  Other operators require explicit type information in ``graph.value_info``;
  missing information causes serialization to fail.
* Formal input counts are resolved from a compact snapshot of registered
  operator schemas using the imported domain and opset. Variadic inputs are
  grouped and omitted trailing optional inputs have zero counts. Operators
  without an input schema in that snapshot, including unknown custom-domain
  operators, are rejected rather than assigned guessed input counts.
* Tensor data is embedded inline. Unresolved external tensor payloads and
  external sidecar output are rejected. Load external tensor bytes into
  ``raw_data`` before serialization.
* Flatbuffer assembly is single-threaded. ``num_threads`` may be set, but
  does not parallelize that assembly step.
* ``SerializeOptions.alignment`` aligns inline tensor payload offsets within
  the serialized buffer (and thus within a file written from offset zero).
  It does not guarantee alignment of the base pointer of a returned Python
  ``bytes`` object or another allocation. Aligned in-memory access also
  requires a suitably aligned buffer base.

See also
--------

* :ref:`l-design-ort-flatbuffer-format` - native writer layout,
  parallelization, and alignment.
* :ref:`l-example-plot-save-ort-flatbuffers` - end-to-end example that saves
  the same model in both formats and compares the resulting file sizes.
* :ref:`l-howto-load-save-onnx-files-cpp` - load/save recipes for the regular
  ``.onnx`` protobuf format.

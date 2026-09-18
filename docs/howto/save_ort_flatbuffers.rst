.. _l-howto-save-ort-flatbuffers:

:html_theme.sidebar_secondary.remove:

How to save and load a model in the ORT flatbuffer format
=========================================================

`onnxruntime <https://onnxruntime.ai/>`_ defines a compact flatbuffer
serialization (``.ort``) that avoids the protobuf parsing step when loading
models in ONNX Runtime.

*onnx-light* exposes the format through
:py:class:`onnx_light.onnx.SerializeFormat` (value ``ORT_FLATBUFFERS``).
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

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: python

          import onnx_light.onnx as onnxl

          sopts = onnxl.SerializeOptions()
          sopts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS
          model.SerializeToFile("model.ort", sopts)

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: cpp

          #include "onnx.h"
          #include "onnx_helper.h"
          #include "stream.h"

          onnx::SerializeOptions options;
          options.format = onnx::SerializeFormat::kOrtFlatbuffers;
          onnx::utils::FileWriteStream stream("model.ort");
          onnx::SerializeModelProtoToStream(model, stream, options);

Memory and file-descriptor output
--------------------------------

The same options also apply to ``SerializeToString`` and
``SerializeToFileDescriptor``:

.. code-block:: python

    payload = model.SerializeToString(sopts)
    assert payload[4:8] == b"ORTM"

    with open("model.ort", "wb") as output:
        model.SerializeToFileDescriptor(output.fileno(), sopts)

To load the result for inference, use ONNX Runtime directly:

.. code-block:: python

    import onnxruntime

    session = onnxruntime.InferenceSession(
        "model.ort", providers=["CPUExecutionProvider"]
    )

Read with the native reader
---------------------------

Set ``ParseOptions.format`` to select native ORT decoding, whether reading
from a file or from bytes:

.. code-block:: python

    popts = onnxl.ParseOptions()
    popts.format = onnxl.SerializeFormat.ORT_FLATBUFFERS

    restored = onnxl.ModelProto()
    restored.ParseFromFile("model.ort", popts)

    restored_from_bytes = onnxl.ModelProto()
    restored_from_bytes.ParseFromString(payload, popts)

    # Default serialization produces ONNX protobuf for execution or saving.
    session = onnxruntime.InferenceSession(
        restored.SerializeToString(), providers=["CPUExecutionProvider"]
    )

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
  missing information causes serialization to fail. Custom operators also
  require their kernels to be registered with the consuming ONNX Runtime
  session.
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
* :ref:`l-howto-load-save-onnx-files` - load/save recipes for the regular
  ``.onnx`` protobuf format.

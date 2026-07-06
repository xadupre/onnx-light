.. _l-howto-onnxruntime-migration:

Replacing onnxruntime's protobuf usage with onnx-light
======================================================

:epkg:`onnxruntime` (ORT) is written against the *protobuf* implementation of
:epkg:`onnx`.  Besides the message types (:class:`~onnx_light.onnx_lib.ModelProto`,
:class:`~onnx_light.onnx_lib.TensorProto`, ...), ORT calls the **protobuf
message API** (``ParseFromArray``, ``SerializeToArray``, ``ByteSizeLong``, ...)
and the **protobuf I/O streams** (``google::protobuf::io::ZeroCopyInputStream``,
``FileInputStream``, ``FileOutputStream``, ...).

``onnx-light`` is *protobuf-free*: its message classes do **not** expose these
methods.  This page documents, method by method, how to replace every protobuf
/ onnx C++ construct that onnxruntime relies on with its ``onnx-light``
equivalent.  It is the reference used by the ``onnxruntime_USE_ONNX_LIGHT``
build option.

.. contents::
    :local:

Detecting the backend at compile time
-------------------------------------

When onnxruntime is configured with ``-Donnxruntime_USE_ONNX_LIGHT=ON`` the
build defines the preprocessor macro ``ORT_USE_ONNX_LIGHT`` for every
translation unit (``add_compile_definitions(ORT_USE_ONNX_LIGHT)``).  Code that
must behave differently between the two backends branches on that macro::

    #if defined(ORT_USE_ONNX_LIGHT)
      // onnx-light native API
    #else
      // protobuf API
    #endif

The ``onnx-light`` drop-in header ``<onnx/onnx_pb.h>`` also defines
``ONNX_LIGHT_NAMESPACE`` (the upstream ``onnx`` package does not), which can be
used as a header-only fallback detector.

The message (de)serialization API
---------------------------------

onnx-light message classes expose the following **native** API (declared by the
``SERIALIZATION_METHOD()`` macro in ``onnx_light/onnx_proto/stream_class.h``):

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - onnx-light method
      - Meaning
    * - ``bool ParseFromString(const std::string&)``
      - Parse the message from a byte buffer (returns ``true``; throws on error).
    * - ``bool ParseFromIstream(std::istream*)``
      - Parse the message from a ``std::istream`` (reads to EOF).
    * - ``bool SerializeToString(std::string&) const``
      - Serialize the message into a ``std::string`` (returns ``true``; throws on error).
    * - ``SerializeSizeResult SerializeSize() const``
      - Compute the serialized-size split (proto / small / big data).
    * - ``size_t ByteSizeLong() const``
      - Total serialized size in bytes (``SerializeSize().size()``); no serialization.

Every protobuf message method used by onnxruntime maps onto that native API:

.. list-table::
    :header-rows: 1
    :widths: 46 54

    * - protobuf (``onnx``) call
      - onnx-light replacement
    * - ``bool p.ParseFromArray(data, size)``
      - ``p.ParseFromString(std::string(reinterpret_cast<const char*>(data), size)); // returns bool``
    * - ``bool p.ParseFromString(str)``
      - ``p.ParseFromString(str); // returns bool, throws on error``
    * - ``bool p.ParseFromZeroCopyStream(&IstreamInputStream)``
      - ``p.ParseFromIstream(&istream)``
    * - ``bool p.ParseFromZeroCopyStream(&FileInputStream)`` (fd)
      - read the descriptor into a ``std::string`` then ``p.ParseFromString(...)``
    * - ``bool p.SerializeToString(&str)``
      - ``p.SerializeToString(str); // returns bool``
    * - ``std::string p.SerializeAsString()``
      - ``std::string s; p.SerializeToString(s); return s;``
    * - ``size_t p.ByteSizeLong()``
      - ``p.ByteSizeLong(); // native, calls SerializeSize() – no serialization``
    * - ``bool p.SerializeToArray(data, size)``
      - ``std::string s; p.SerializeToString(s); memcpy(data, s.data(), s.size());``
    * - ``bool p.SerializeToOstream(&os)``
      - ``std::string s; p.SerializeToString(s); os.write(s.data(), s.size());``
    * - ``bool p.SerializeToFileDescriptor(fd)``
      - ``std::string s; p.SerializeToString(s); write(fd, s.data(), s.size());``
    * - ``FileOutputStream o(fd); p.SerializeToZeroCopyStream(&o) && o.Flush()``
      - ``std::string s; p.SerializeToString(s); write(fd, s.data(), s.size());``

.. note::

    ``onnx-light`` has no 2 GB message-size limit and lets exceptions propagate
    instead of returning ``false`` (see :ref:`the design notes <l-how-to>`).
    The wrappers above return ``true`` on success so onnxruntime's ``bool``
    call sites keep working unchanged.

A ready-made onnxruntime helper
-------------------------------

To keep the two backends in a single place, onnxruntime ships a header-only
adapter, ``include/onnxruntime/core/graph/onnx_proto_serialize.h``, exposing
templated free functions in ``namespace onnxruntime::proto_io``.  Each function
compiles to the protobuf call or the onnx-light call depending on
``ORT_USE_ONNX_LIGHT``:

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - Helper
      - Replaces
    * - ``proto_io::ParseFromArray(p, data, size)``
      - ``p.ParseFromArray(data, size)``
    * - ``proto_io::ParseFromString(p, str)``
      - ``p.ParseFromString(str)`` (bool context)
    * - ``proto_io::ParseFromIStream(p, istream)``
      - ``ParseFromZeroCopyStream(&IstreamInputStream)``
    * - ``proto_io::ParseFromFileDescriptor(p, fd)``
      - ``ParseFromZeroCopyStream(&FileInputStream)``
    * - ``proto_io::SerializeToString(p, str)``
      - ``p.SerializeToString(&str)``
    * - ``proto_io::SerializeAsString(p)``
      - ``p.SerializeAsString()``
    * - ``proto_io::ByteSize(p)``
      - ``p.ByteSizeLong()``
    * - ``proto_io::SerializeToArray(p, data, size)``
      - ``p.SerializeToArray(data, size)``
    * - ``proto_io::SerializeToOStream(p, os)``
      - ``p.SerializeToOstream(&os)``
    * - ``proto_io::SerializeToFileDescriptor(p, fd)``
      - ``p.SerializeToFileDescriptor(fd)``
    * - ``proto_io::SaveToFileDescriptor(p, fd)``
      - ``FileOutputStream(fd) + SerializeToZeroCopyStream + Flush``

.. tab-set::

   .. tab-item:: Before (protobuf)

      .. code-block:: cpp

          #include <google/protobuf/io/zero_copy_stream_impl.h>

          google::protobuf::io::IstreamInputStream zci(&istream);
          bool ok = model_proto.ParseFromZeroCopyStream(&zci) && istream.eof();

          auto n = model_proto.ByteSizeLong();
          model_proto.SerializeToArray(buffer, n);

   .. tab-item:: After (onnx-light)

      .. code-block:: cpp

          #include "core/graph/onnx_proto_serialize.h"

          bool ok = onnxruntime::proto_io::ParseFromIStream(model_proto, istream);

          auto n = onnxruntime::proto_io::ByteSize(model_proto);
          onnxruntime::proto_io::SerializeToArray(model_proto, buffer, n);

The ``google::protobuf`` symbols still referenced by onnxruntime
----------------------------------------------------------------

A few onnxruntime files reference the ``google::protobuf`` namespace for
non-serialization purposes.  ``onnx-light`` provides drop-in equivalents in
``onnx_light/onnx_proto/google_protobuf_compat.h`` (pulled in automatically by
the ``<onnx/onnx_pb.h>`` compatibility header):

.. list-table::
    :header-rows: 1
    :widths: 50 50

    * - protobuf symbol
      - onnx-light shim
    * - ``google::protobuf::RepeatedField<T>``
      - alias of ``onnx_light::utils::RepeatedField<T>``
    * - ``google::protobuf::RepeatedPtrField<T>``
      - alias of ``onnx_light::utils::RepeatedProtoField<T>``
    * - ``google::protobuf::RepeatedFieldBackInsertIterator<T>``
      - alias of ``onnx_light::utils::RepeatedFieldBackInsertIterator<T>``
    * - ``google::protobuf::RepeatedFieldBackInserter(&f)``
      - forwards to ``onnx_light::utils::RepeatedFieldBackInserter``
    * - ``google::protobuf::ShutdownProtobufLibrary()``
      - no-op (onnx-light has no global protobuf state)
    * - ``google::protobuf::io::ArrayInputStream``
      - alias of ``onnx_light::utils::StringStream``
    * - ``google::protobuf::io::CodedInputStream``
      - alias of ``onnx_light::utils::CodedInputStream``
    * - ``google::protobuf::io::StringOutputStream``
      - alias of ``onnx_light::utils::StdStringWriteStream``
    * - ``google::protobuf::io::FileOutputStream``
      - alias of ``onnx_light::utils::FdWriteStream``
    * - ``google::protobuf::io::IstreamInputStream``
      - alias of ``onnx_light::utils::IstreamStream``
    * - ``google::protobuf::io::OstreamOutputStream``
      - alias of ``onnx_light::utils::OstreamWriteStream``

Every class in ``google_protobuf_compat.h`` is now a pure ``using`` alias to a
concrete ``onnx-light`` class.  The ZeroCopy stream interface
(``Next`` / ``BackUp`` / ``ByteCount`` and, for output streams, ``Flush``) is
implemented directly on the ``onnx-light`` ``BinaryStream`` / ``BinaryWriteStream``
hierarchy in ``stream.h`` / ``stream.cc``, so the compat header carries no
implementation of its own.

Version guards
--------------

Blocks guarded by ``GOOGLE_PROTOBUF_VERSION`` must also exclude the onnx-light
build, because the macro is undefined there (and would evaluate to ``0``).  The
protobuf-arena helpers ``RepeatedPtrField::ClearedCount`` /
``ReleaseCleared`` have no onnx-light counterpart and are simply skipped
(onnx-light stores repeated fields in ``std::vector`` and frees them on
``Clear()``)::

    #if !defined(ORT_USE_ONNX_LIGHT) && GOOGLE_PROTOBUF_VERSION < 5026000
      // protobuf-only cleared-object reclamation
    #endif

onnxruntime files converted
---------------------------

The core library (always built) has been converted to route through
``onnxruntime::proto_io`` or to guard protobuf-only code:

* ``core/graph/model.cc`` — istream / file-descriptor load and save,
  ``ByteSizeLong``, ``SerializeToArray``, ``ParseFromArray``.
* ``core/graph/graph.cc`` — ``ClearedCount`` / ``ReleaseCleared`` guards.
* ``core/session/inference_session.cc`` — ``ParseFromArray``, ``ByteSizeLong``.
* ``core/session/provider_bridge_ort.cc`` — ``ModelProto`` / ``FunctionProto``
  ``SerializeToString`` / ``SerializeToOstream`` / ``ParseFromString`` /
  ``SerializeAsString`` bridge methods.
* ``core/framework/graph_partitioner.cc`` — ``ByteSizeLong``,
  ``SerializeToArray``, ``SerializeToOstream``, file-descriptor save.
* ``core/framework/debug_node_inputs_outputs_utils.cc`` —
  ``SerializeToFileDescriptor``, ``SerializeAsString``.
* ``core/framework/allocation_planner.cc`` — ``ByteSizeLong``.

The same substitution (include ``core/graph/onnx_proto_serialize.h`` and call
the matching ``proto_io`` helper) applies to the optional components that are
only built on demand and therefore not covered by the default build:

* execution providers — ``providers/tensorrt``, ``providers/nv_tensorrt_rtx``,
  ``providers/openvino``, ``providers/cann``, ``providers/migraphx``,
  ``providers/vitisai`` (``SerializeToString`` / ``ParseFromString`` /
  ``SerializeAsString`` used in a ``bool`` / return context).
* training — ``orttraining/...`` (``ParseFromArray`` / ``SerializeToZeroCopyStream``).
* tests — ``test/onnx``, ``test/framework``, ``test/fuzzing`` (model load / save
  helpers).

See also
--------

* :ref:`l-howto-replace-onnx` — the general onnx → onnx-light migration guide.
* :ref:`l-design-cpp-linking` — the exported CMake targets.

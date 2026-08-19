.. _l-next-steps-ort-onnx-light:

Integrating ``onnx-light`` into onnxruntime (PR #29723)
=======================================================

:Date: 2026-07

**complete**

Objective
+++++++++

:epkg:`onnxruntime` (ORT) is written against the *protobuf* implementation of
:epkg:`onnx`.  Besides the message types
(:class:`~onnx_light.onnx_lib.ModelProto`,
:class:`~onnx_light.onnx_lib.TensorProto`, ...) it calls the protobuf message
API (``ParseFromArray``, ``SerializeToArray``, ``ByteSizeLong``, ...) and the
protobuf I/O streams (``google::protobuf::io::ZeroCopyInputStream``,
``FileInputStream``, ``FileOutputStream``, ...).

`microsoft/onnxruntime#29723
<https://github.com/microsoft/onnxruntime/pull/29723>`_ made ORT build against
``onnx-light`` instead of ``libprotobuf`` behind a new
``onnxruntime_USE_ONNX_LIGHT`` build option.  The objective was to route every
protobuf / onnx C++ construct ORT relies on through its ``onnx-light``
equivalent without regressing the default protobuf build.  The full method-by
-method mapping is documented in
:ref:`l-howto-onnxruntime-migration`.

Post-mortem
+++++++++++

The change was structured so the two backends coexist in one source tree and
are selected at configure time.

Compile-time backend selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configuring ORT with ``-Donnxruntime_USE_ONNX_LIGHT=ON`` defines the
preprocessor macro ``ORT_USE_ONNX_LIGHT`` for every translation unit.  Code
that must differ between backends branches on that macro::

    #if defined(ORT_USE_ONNX_LIGHT)
      // onnx-light native API
    #else
      // protobuf API
    #endif

The drop-in header ``<onnx/onnx_pb.h>`` also defines ``ONNX_LIGHT_NAMESPACE``
(the upstream ``onnx`` package does not), which acts as a header-only fallback
detector.

A single serialization adapter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Rather than sprinkling ``#if`` blocks over every call site, the PR added a
header-only adapter, ``core/graph/onnx_proto_serialize.h``, exposing templated
free functions in ``namespace onnxruntime::proto_io``.  Each function compiles
to the protobuf call or the ``onnx-light`` call depending on
``ORT_USE_ONNX_LIGHT``:

* ``ParseFromArray`` / ``ParseFromString`` / ``ParseFromIStream`` /
  ``ParseFromFileDescriptor``;
* ``SerializeToString`` / ``SerializeAsString`` / ``ByteSize`` /
  ``SerializeToArray`` / ``SerializeToOStream`` / ``SerializeToFileDescriptor`` /
  ``SaveToFileDescriptor``.

``onnx-light`` lets exceptions propagate instead of returning ``false``; the
wrappers return ``true`` on success so ORT's ``bool`` call sites keep working
unchanged, and the missing 2 GB message-size limit is a pure improvement.

The ``google::protobuf`` shim
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A few ORT files name the ``google::protobuf`` namespace for non-serialization
purposes.  ``onnx-light`` provides drop-in equivalents in
``onnx_light/onnx_proto/google_protobuf_compat.h`` (pulled in by the
``<onnx/onnx_pb.h>`` compatibility header): ``RepeatedField<T>``,
``RepeatedPtrField<T>``, the back-insert iterators, ``ShutdownProtobufLibrary``
(a no-op), and the ``google::protobuf::io`` stream classes.  Each is a pure
``using`` alias onto a concrete ``onnx-light`` type, and the ZeroCopy stream
interface is implemented directly on the ``BinaryStream`` /
``BinaryWriteStream`` hierarchy, so the shim carries no implementation of its
own.

Version guards
^^^^^^^^^^^^^^

Blocks guarded by ``GOOGLE_PROTOBUF_VERSION`` also exclude the ``onnx-light``
build, because the macro is undefined there (and would evaluate to ``0``).  The
protobuf-arena helpers ``RepeatedPtrField::ClearedCount`` / ``ReleaseCleared``
have no ``onnx-light`` counterpart and are skipped — repeated fields are stored
in ``std::vector`` and freed on ``Clear()``::

    #if !defined(ORT_USE_ONNX_LIGHT) && GOOGLE_PROTOBUF_VERSION < 5026000
      // protobuf-only cleared-object reclamation
    #endif

Files converted
^^^^^^^^^^^^^^^

The always-built core library was routed through ``onnxruntime::proto_io`` or
guarded: ``core/graph/model.cc`` (istream / file-descriptor load and save,
``ByteSizeLong``, ``SerializeToArray``, ``ParseFromArray``),
``core/graph/graph.cc`` (``ClearedCount`` / ``ReleaseCleared`` guards),
``core/session/inference_session.cc``, ``core/session/provider_bridge_ort.cc``,
``core/framework/graph_partitioner.cc``,
``core/framework/debug_node_inputs_outputs_utils.cc``, and
``core/framework/allocation_planner.cc``.  The same substitution applies to the
optional, build-on-demand components (execution providers, training, tests).

What worked
^^^^^^^^^^^

* Centralizing the serialization differences in ``proto_io`` kept the call
  sites identical between backends and made the diff auditable file by file.
* Making the ``google_protobuf_compat.h`` symbols pure aliases meant the shim
  had no behavior to keep in sync — the stream semantics live in ``onnx-light``.
* Guarding the protobuf-only arena helpers, rather than emulating them, avoided
  reintroducing protobuf concepts that ``onnx-light`` deliberately does not
  have.
* Preserving the ``bool`` return convention while letting exceptions propagate
  meant unchanged error handling at the ORT call sites.

What remains
^^^^^^^^^^^^

The default ORT build still uses protobuf; ``onnx-light`` is opt-in via
``onnxruntime_USE_ONNX_LIGHT``.  Optional components only compiled on demand
(some execution providers, training, and fuzzing tests) follow the same
substitution but are not exercised by the default build matrix.

This integration preserves ORT's existing loading pipeline. It does not yet
remove the complete staging buffer in ``ParseFromFileDescriptor``, control
external-weight ownership, avoid runtime prepacking, or separate metadata,
session-ready, and first-token timings. Those performance steps are specified
in :ref:`l-next-steps-model-loading`.

See also
++++++++

* :ref:`l-howto-onnxruntime-migration` — the method-by-method migration
  reference used by ``onnxruntime_USE_ONNX_LIGHT``.
* :ref:`l-next-steps-onnx-proto` — how the protobuf-free ``onnx_proto`` library
  was built.
* :ref:`l-next-steps-model-loading` — the loading-performance roadmap following
  the compatibility milestone.

.. _l-next-steps-lib-onnx:

Porting the ONNX C++ library on top of ``onnx_proto``
=====================================================

:Date: 2026-06

**complete**

Objective
+++++++++

:ref:`l-next-steps-onnx-proto` built the protobuf-free message layer
(``lib_onnx_proto``): the ``ModelProto`` / ``GraphProto`` / ``TensorProto``
hierarchy, the binary streams, and a wire-compatible parser and serializer.
Messages alone are not enough — the upstream :epkg:`onnx` package also provides
the *library* built on those messages: the operator schema registry, the model
checker, shape inference, the version converter, the function inliner, and the
text parser/printer.

The objective was to port that library so it compiles and runs against
``onnx_proto`` instead of ``libprotobuf``, exposes the same C++ API surface as
upstream ONNX (so existing consumers link it unchanged), and carries no
protobuf dependency of its own.

Post-mortem
+++++++++++

The port keeps upstream's public entry points and internal structure and swaps
only the message layer underneath, using compatibility headers so the source
diff against upstream stays small.

One library: ``lib_onnx_lib``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``onnx_light/onnx_lib`` builds the ``lib_onnx_lib`` target from the ported
sources: the operator definitions under ``defs``, ``checker.cc``,
``shape_inference``, ``version_converter``, ``inliner``, and the IR/proto
converter in ``common``.  It links ``lib_onnx_proto`` (messages and wire
format) and ``lib_onnx_manipulations`` (helper utilities) and nothing from
protobuf.  The library can be built STATIC for pure C++ consumers or SHARED
when the Python bindings are enabled.

The protobuf drop-in
^^^^^^^^^^^^^^^^^^^^^

Upstream code includes ``<onnx/onnx_pb.h>`` and occasionally names
``google::protobuf`` types (``RepeatedField``, ``RepeatedPtrField``, the I/O
streams).  Rather than rewrite those call sites, the port supplies drop-in
headers: ``onnx_lib/common/onnx_pb.h`` replaces the protobuf-generated header,
``onnx_proto/google_protobuf_compat.h`` aliases the ``google::protobuf`` names
onto concrete ``onnx_proto`` types, and the forwarding tree in
``onnx_compat_include`` resolves ``onnx/...`` and ``google/protobuf/...``
includes to those replacements.  Because ``ONNX_LIGHT_NAMESPACE`` mirrors the
upstream ``onnx`` namespace, ``onnx::checker::check_model`` and
``onnx::shape_inference::InferShapes`` keep the same spelling for callers.

What was ported
^^^^^^^^^^^^^^^

* **Operator schemas** (``defs/``) — ``OpSchema`` (formal parameters,
  type constraints, attributes, domain/version) and the ``OpSchemaRegistry``
  singleton, populated through the ``ONNX_OPERATOR_SCHEMA`` static-registration
  macro.  The definitions are grouped by domain family (math, nn, rnn,
  sequence, quantization, ...), with the ``ai.onnx.ml`` operators compiled only
  when ``ONNX_ML`` is set.
* **Checker** (``checker.cc``) — ``check_model`` and friends, with a
  ``ValidationError`` that appends lexical context as it unwinds.
* **Shape inference** (``shape_inference/``) — per-node and whole-model
  type/shape inference with the symbol table, graph inferencer, and inference
  context implementations.
* **Version converter** (``version_converter/``) — the adapter chain
  that migrates a model between opset versions, one adapter per operator
  transition.
* **Inliner** (``inliner/``) — local-function inlining with
  cyclic-reference detection.
* **Parser / printer** — the ONNX text format, reachable from C++ and from the
  Python bindings that wrap it.

What worked
^^^^^^^^^^^

* Preserving upstream's public API meant a project already using ONNX's C++
  library could link ``lib_onnx_lib`` instead, exactly as
  :ref:`l-next-steps-ort-onnx-light` did for onnxruntime.
* Confining the substitution to a handful of compatibility headers
  (``onnx_pb.h``, ``google_protobuf_compat.h``, the ``onnx_compat_include``
  tree) kept the diff against upstream small and mechanical, which makes future
  re-syncs tractable.
* Keeping the schema registry, checker, shape inference, and converter as one
  library matched upstream's layering and let each component call the others
  without new seams.
* Building on the wire-compatible ``onnx_proto`` layer meant the checker and
  shape inference operated on byte-identical models, so their results can be
  compared directly against upstream ONNX.

What remains
^^^^^^^^^^^^

The ported surface covers the parts of the ONNX C++ library that onnx-light
consumers rely on; less-used corners of the upstream API are added on demand.
The compatibility shims expose only the ``google::protobuf`` names the ported
code actually uses, so unusual direct protobuf usage in third-party code may
need a small addition to the alias headers.

See also
++++++++

* :ref:`l-next-steps-onnx-proto` — the protobuf-free message layer this library
  is built on.
* :ref:`l-next-steps-ort-onnx-light` — using the same drop-in headers to build
  onnxruntime against onnx-light.

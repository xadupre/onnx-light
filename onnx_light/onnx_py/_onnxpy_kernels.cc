// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/random.h"
#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/runtime_context.h"
#include "onnx_kernels/simple_tensor.h"

#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using onnx_kernels::RuntimeContext;
using onnx_kernels::Tensor;
using onnx_kernels::kernel::KernelContext;
using onnx_kernels::kernel::OpsetId;

void AddOnnxPyKernels(nb::module_ &m);
void AddOnnxPyRuntime(nb::module_ &m);

namespace {

onnx_kernels::TensorEventKind ParseTensorEventKind(const std::string &kind) {
  if (kind == "unknown")
    return onnx_kernels::TensorEventKind::kUnknown;
  if (kind == "initializer")
    return onnx_kernels::TensorEventKind::kInitializer;
  if (kind == "input")
    return onnx_kernels::TensorEventKind::kInput;
  if (kind == "intermediate")
    return onnx_kernels::TensorEventKind::kIntermediate;
  if (kind == "output")
    return onnx_kernels::TensorEventKind::kOutput;
  throw std::invalid_argument(
      "RuntimeContext: unknown tensor event kind '" + kind +
      "' (expected one of: unknown, initializer, input, intermediate, output).");
}

} // namespace

NB_MODULE(_onnxpykernels, m) {
  m.doc() = "onnx_light kernels bindings: deterministic pseudo-random helpers "
            "backing _onnxpybackend_test, plus the RunNode/RunGraph/RunFunction/"
            "RunModel dispatcher and its supporting RuntimeContext/KernelContext "
            "types.";

  // The ``runtime`` submodule exposes :cpp:func:`RunNode`,
  // :cpp:func:`RunNodes`, :cpp:func:`RunGraph`, :cpp:func:`RunFunction` and
  // :cpp:func:`RunModel`. These take/return ``Tensor`` and proto types whose
  // ``nb::class_`` bindings live in sibling extensions. Importing those
  // extensions here guarantees the cross-module typeid registry has the
  // necessary entries by the time we register the ``runtime`` callables and
  // makes the runtime submodule usable even when consumers import
  // ``_onnxpykernels`` directly (bypassing the ``_onnxpy.py`` shim).
  nb::module_::import_("onnx_light.onnx_py._onnxpyprotoop");
  nb::module_::import_("onnx_light.onnx_py._onnxpybackend");

  AddOnnxPyKernels(m);
  AddOnnxPyRuntime(m);
}

void AddOnnxPyKernels(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `backend`
  // Deterministic pseudo-random helpers backing ``onnx_light.backend``.
  // -----------------------------------------------------------------------
  auto backend_mod = m.def_submodule("backend");
  backend_mod.doc() =
      "Deterministic pseudo-random helpers (SplitMix64) used by onnx_light.backend.";

  backend_mod.def(
      "next_uint64", [](uint64_t state) { return onnx_kernels::NextUint64(state); },
      nb::arg("state"), "Returns ``(next_state, value)`` for the SplitMix64 generator.");

  backend_mod.def(
      "rand",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return onnx_kernels::Rand(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic uniform values in ``[0, 1)`` as a flat list.");

  backend_mod.def(
      "randint",
      [](int64_t low, int64_t high, const std::vector<int64_t> &shape,
         std::optional<uint64_t> seed) { return onnx_kernels::RandInt(low, high, shape, seed); },
      nb::arg("low"), nb::arg("high"), nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic integers in ``[low, high)`` as a flat list.");

  backend_mod.def(
      "randn",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return onnx_kernels::Randn(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` approximately normal-distributed values (Irwin-Hall) "
      "as a flat list.");
}

void AddOnnxPyRuntime(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `runtime`
  // Exposes the C++ RunNode/RunNodes/RunGraph/RunFunction/RunModel
  // dispatcher together with its supporting RuntimeContext / KernelContext
  // / OpsetId types and the TensorFromProto helper.
  // -----------------------------------------------------------------------
  auto rt_mod = m.def_submodule("runtime");
  rt_mod.doc() = "C++ kernel dispatcher exposed to Python. RunNode/RunNodes/RunGraph/"
                 "RunFunction/RunModel evaluate one or more nodes through the static "
                 "KernelDispatchTable (with transparent dispatch to model-local "
                 "FunctionProto's) using a name-keyed RuntimeContext for tensor I/O.";

  // OpsetId — (domain, version) opset identifier consumed by KernelContext.
  nb::class_<OpsetId>(rt_mod, "OpsetId",
                      "(domain, version) opset identifier. The empty string denotes the "
                      "default ai.onnx domain.")
      .def(nb::init<>())
      .def(nb::init<std::string, int64_t>(), nb::arg("domain"), nb::arg("version"))
      .def_rw("domain", &OpsetId::domain)
      .def_rw("version", &OpsetId::version)
      .def("__repr__", [](const OpsetId &o) {
        return "OpsetId(domain='" + o.domain + "', version=" + std::to_string(o.version) + ")";
      });

  rt_mod.def(
      "default_opset", [](int64_t version) { return onnx_kernels::kernel::DefaultOpset(version); },
      nb::arg("version"),
      "Returns an :class:`OpsetId` for the default ai.onnx domain at ``version``.");

  // KernelContext — construction-time context passed to each kernel.
  nb::class_<KernelContext>(rt_mod, "KernelContext",
                            "Construction-time context shared by every kernel invoked through "
                            ":func:`RunNode`. Bundles the target opset.")
      .def(nb::init<>())
      .def(nb::init<OpsetId>(), nb::arg("opset"))
      .def_rw("opset", &KernelContext::opset)
      .def("__repr__", [](const KernelContext &k) {
        return "KernelContext(opset=OpsetId(domain='" + k.opset.domain +
               "', version=" + std::to_string(k.opset.version) + "))";
      });

  // TensorEvent — append-only log entry for a single tensor map mutation.
  // Mirrors :cpp:class:`onnx_kernels::TensorEvent`; ``values`` / ``string_values``
  // expose the populated prefix of the underlying fixed-size buffer as Python
  // lists of length ``value_count``.
  nb::class_<onnx_kernels::TensorEvent>(
      rt_mod, "TensorEvent",
      "One entry of the :meth:`RuntimeContext.events` log describing a single "
      "tensor map mutation (``add`` / ``replace`` / ``remove``). For ``add`` and "
      "``replace`` events the first ``value_count`` element values of the tensor "
      "are captured inline (``values`` for numeric dtypes, ``string_values`` for "
      "``STRING``); the underlying buffer is fixed-size (capped at 8 entries). "
      "For tensors with more than 8 elements only the first 8 are kept, "
      "``data_type`` is set to ``-1`` and ``shape`` is empty to signal the "
      "truncated payload. ``remove`` events carry ``data_type = UNDEFINED``, "
      "empty ``shape`` and ``value_count = 0`` (the tensor is already gone).")
      .def_prop_ro(
          "action",
          [](const onnx_kernels::TensorEvent &ev) {
            return std::string(onnx_kernels::TensorEventActionName(ev.action));
          },
          "Mutation kind: ``\"add\"``, ``\"replace\"`` or ``\"remove\"``.")
      .def_prop_ro(
          "kind",
          [](const onnx_kernels::TensorEvent &ev) {
            return std::string(onnx_kernels::TensorEventKindName(ev.kind));
          },
          "Tensor role: ``\"unknown\"``, ``\"initializer\"``, ``\"input\"``, "
          "``\"intermediate\"`` or ``\"output\"``.")
      .def_ro("timestamp_ns", &onnx_kernels::TensorEvent::timestamp_ns,
              "Nanoseconds since the Unix epoch (``std::chrono::system_clock``).")
      .def_ro("name", &onnx_kernels::TensorEvent::name, "Tensor name targeted by the mutation.")
      .def_ro("data_type", &onnx_kernels::TensorEvent::data_type,
              "``TensorProto.DataType`` integer of the tensor, or ``-1`` when "
              "the payload was truncated (more than 8 elements).")
      .def_ro("shape", &onnx_kernels::TensorEvent::shape,
              "Tensor shape, or empty list when truncated / for ``remove`` events.")
      .def_ro("value_count", &onnx_kernels::TensorEvent::value_count,
              "Number of populated entries in :attr:`values` / :attr:`string_values` "
              "(``min(element_count, 8)``, ``0`` for ``remove`` and ``run_node`` events).")
      .def_ro("op_domain", &onnx_kernels::TensorEvent::op_domain,
              "For ``run_node`` events: normalised ONNX op domain of the dispatched "
              "node (default domain reported as ``\"ai.onnx\"``). Empty for other "
              "event actions.")
      .def_ro("op_type", &onnx_kernels::TensorEvent::op_type,
              "For ``run_node`` events: ONNX ``op_type`` of the dispatched node. "
              "Empty for other event actions.")
      .def_ro("inputs", &onnx_kernels::TensorEvent::inputs,
              "For ``run_node`` events: ordered list of input names consumed by the "
              "node (matching ``NodeProto.input``). Empty for other event actions.")
      .def_ro("duration_ns", &onnx_kernels::TensorEvent::duration_ns,
              "For ``run_node`` events: wall-clock duration of the kernel dispatch in "
              "nanoseconds (``std::chrono::steady_clock``). ``0`` for other event "
              "actions.")
      .def_prop_ro(
          "values",
          [](const onnx_kernels::TensorEvent &ev) {
            nb::list out;
            if (static_cast<onnx_kernels::DataType>(ev.data_type) !=
                onnx_kernels::DataType::STRING) {
              for (int32_t i = 0; i < ev.value_count; ++i) {
                out.append(ev.values[i]);
              }
            }
            return out;
          },
          "First ``value_count`` numeric values of the tensor (empty for ``STRING`` "
          "tensors and ``remove`` events).")
      .def_prop_ro(
          "string_values",
          [](const onnx_kernels::TensorEvent &ev) {
            nb::list out;
            if (static_cast<onnx_kernels::DataType>(ev.data_type) ==
                onnx_kernels::DataType::STRING) {
              for (int32_t i = 0; i < ev.value_count; ++i) {
                out.append(ev.string_values[i]);
              }
            }
            return out;
          },
          "First ``value_count`` string values of the tensor (empty for non-``STRING`` "
          "tensors and ``remove`` events).")
      .def(
          "as_dict",
          [](const onnx_kernels::TensorEvent &ev) {
            nb::dict d;
            d["action"] = std::string(onnx_kernels::TensorEventActionName(ev.action));
            d["kind"] = std::string(onnx_kernels::TensorEventKindName(ev.kind));
            d["timestamp_ns"] = ev.timestamp_ns;
            d["name"] = ev.name;
            d["data_type"] = ev.data_type;
            d["shape"] = ev.shape;
            d["value_count"] = ev.value_count;
            d["op_domain"] = ev.op_domain;
            d["op_type"] = ev.op_type;
            d["inputs"] = ev.inputs;
            d["duration_ns"] = ev.duration_ns;
            const int32_t n = ev.value_count;
            if (static_cast<onnx_kernels::DataType>(ev.data_type) ==
                onnx_kernels::DataType::STRING) {
              nb::list svals;
              for (int32_t i = 0; i < n; ++i) {
                svals.append(ev.string_values[i]);
              }
              d["string_values"] = std::move(svals);
              d["values"] = nb::list();
            } else {
              nb::list nvals;
              for (int32_t i = 0; i < n; ++i) {
                nvals.append(ev.values[i]);
              }
              d["values"] = std::move(nvals);
              d["string_values"] = nb::list();
            }
            return d;
          },
          "Returns the event fields as a plain Python ``dict`` (trivially "
          "renderable as a table, serialisable, etc.).")
      .def("__repr__", [](const onnx_kernels::TensorEvent &ev) {
        return std::string("TensorEvent(action='") +
               onnx_kernels::TensorEventActionName(ev.action) + "', kind='" +
               onnx_kernels::TensorEventKindName(ev.kind) + "', name='" + ev.name +
               "', data_type=" + std::to_string(ev.data_type) +
               ", value_count=" + std::to_string(ev.value_count) + ")";
      });

  // RuntimeContext — name-keyed tensor map + kernel context + function registry.
  nb::class_<RuntimeContext>(
      rt_mod, "RuntimeContext",
      "Per-invocation runtime state passed to :func:`RunNode` / :func:`RunNodes` / "
      ":func:`RunGraph` / :func:`RunFunction` / :func:`RunModel`. Owns the name-keyed "
      "tensor map carrying graph inputs/initializers and every intermediate value "
      "produced by previously executed nodes.")
      .def(nb::init<>())
      .def(nb::init<KernelContext>(), nb::arg("kernel_ctx"))
      .def_prop_rw(
          "kernel_ctx", [](RuntimeContext &rt) -> KernelContext & { return rt.kernel_ctx(); },
          [](RuntimeContext &rt, KernelContext k) { rt.kernel_ctx() = std::move(k); },
          nb::rv_policy::reference_internal, "Kernel construction context (opset).")
      .def("has", &RuntimeContext::Has, nb::arg("name"),
           "Returns ``True`` if a tensor named ``name`` is currently held.")
      .def("remove", &RuntimeContext::Remove, nb::arg("name"),
           "Removes the tensor stored under ``name`` if present. Returns ``True`` if "
           "an entry was erased.")
      .def(
          "set",
          [](RuntimeContext &rt, const std::string &name, Tensor tensor, const std::string &kind) {
            rt.Set(name, std::move(tensor), ParseTensorEventKind(kind));
          },
          nb::arg("name"), nb::arg("tensor"), nb::arg("kind") = "input",
          "Inserts ``tensor`` under ``name``. Raises if ``name`` already exists. "
          "Records an ``add`` event in :func:`events` with the supplied ``kind`` "
          "(default ``\"input\"``).")
      .def(
          "put",
          [](RuntimeContext &rt, const std::string &name, Tensor tensor, const std::string &kind) {
            rt.Put(name, std::move(tensor), ParseTensorEventKind(kind));
          },
          nb::arg("name"), nb::arg("tensor"), nb::arg("kind") = "intermediate",
          "Inserts or overwrites the tensor stored under ``name``. Records an ``add`` "
          "or ``replace`` event in :func:`events` with the supplied ``kind`` "
          "(default ``\"intermediate\"``).")
      .def(
          "get",
          [](RuntimeContext &rt, const std::string &name) -> Tensor & { return rt.Get(name); },
          nb::arg("name"), nb::rv_policy::reference_internal,
          "Returns the tensor stored under ``name``. Raises if absent.")
      .def(
          "names",
          [](const RuntimeContext &rt) {
            std::vector<std::string> out;
            out.reserve(rt.tensors().size());
            for (const auto &kv : rt.tensors()) {
              out.push_back(kv.first);
            }
            return out;
          },
          "Returns the list of tensor names currently held by the context.")
      .def("has_sequence", &RuntimeContext::HasSequence, nb::arg("name"),
           "Returns ``True`` if a sequence named ``name`` is currently held.")
      .def(
          "sequence_names",
          [](const RuntimeContext &rt) {
            std::vector<std::string> out;
            out.reserve(rt.sequences().size());
            for (const auto &kv : rt.sequences()) {
              out.push_back(kv.first);
            }
            return out;
          },
          "Returns the list of sequence names currently held by the context.")
      .def(
          "get_sequence",
          [](const RuntimeContext &rt, const std::string &name) {
            return rt.GetSequence(name).values;
          },
          nb::arg("name"),
          "Returns the tensors in the sequence stored under ``name`` as a list of "
          ":class:`Tensor` objects. Raises ``std::out_of_range`` if absent.")
      .def(
          "events",
          [](const RuntimeContext &rt) {
            // Returns a copy of the append-only log as a list of
            // :class:`TensorEvent` instances. Use :meth:`TensorEvent.as_dict`
            // to materialise an individual entry as a plain ``dict`` (e.g.
            // for serialisation or tabular rendering).
            return rt.events();
          },
          "Returns the append-only log of tensor map mutations (add/replace/remove) "
          "as a list of :class:`TensorEvent` instances. Each entry carries "
          "``action`` (``\"add\"`` / ``\"replace\"`` / ``\"remove\"``), ``kind`` "
          "(``\"unknown\"`` / ``\"initializer\"`` / ``\"input\"`` / "
          "``\"intermediate\"`` / ``\"output\"``), ``timestamp_ns`` "
          "(``int`` nanoseconds since the Unix epoch), ``name``, ``data_type`` "
          "(``TensorProto.DataType`` integer), ``shape``, ``value_count``, and "
          "the first ``value_count`` element values from the tensor (``values`` "
          "for numeric dtypes, ``string_values`` for ``STRING``). The value "
          "buffer is fixed-size (capped at 8 entries); for tensors with more "
          "than 8 elements only the first 8 are kept, ``data_type`` is set to "
          "``-1`` and ``shape`` is empty to signal the truncated payload. Call "
          ":meth:`TensorEvent.as_dict` to convert an individual entry to a "
          "plain Python ``dict``.")
      .def("clear_events", &RuntimeContext::ClearEvents,
           "Empties the event log without otherwise touching the tensor map.");

  // Top-level run helpers.
  rt_mod.def(
      "run_node",
      [](const NodeProto &node, RuntimeContext &rt) { onnx_kernels::RunNode(node, rt); },
      nb::arg("node"), nb::arg("rt"),
      "Runs the kernel registered for ``node`` and stores its outputs in ``rt``. "
      "``rt`` must already contain entries for every input referenced by ``node``; "
      "on return it also contains entries for every output declared by ``node``.");

  rt_mod.def(
      "run_nodes",
      [](const std::vector<NodeProto> &nodes, RuntimeContext &rt) {
        onnx_kernels::RunNodes(nodes.begin(), nodes.end(), rt);
      },
      nb::arg("nodes"), nb::arg("rt"),
      "Runs :func:`run_node` on every node of ``nodes`` in order. The sequence "
      "must be topologically sorted with respect to data dependencies.");

  rt_mod.def(
      "run_graph",
      [](const GraphProto &graph, RuntimeContext &rt) { onnx_kernels::RunGraph(graph, rt); },
      nb::arg("graph"), nb::arg("rt"),
      "Runs all nodes in ``graph`` using ``rt``. Before executing the node sequence "
      "the function seeds ``rt`` with every ``TensorProto`` in ``graph.initializer``.");

  rt_mod.def(
      "run_function",
      [](const FunctionProto &func, RuntimeContext &rt) { onnx_kernels::RunFunction(func, rt); },
      nb::arg("func"), nb::arg("rt"),
      "Runs all nodes in ``func`` using ``rt``. The caller is responsible for "
      "inserting the function's input tensors into ``rt`` before calling.");

  rt_mod.def(
      "run_model",
      [](const ModelProto &model, RuntimeContext &rt) { onnx_kernels::RunModel(model, rt); },
      nb::arg("model"), nb::arg("rt"),
      "Runs the graph embedded in ``model`` using ``rt``. Before delegating to "
      ":func:`run_graph`, every ``FunctionProto`` in ``model.functions`` is registered "
      "in the runtime's function registry so nodes referring to model-local functions "
      "by ``(domain, op_type, overload)`` are dispatched transparently.");

  rt_mod.def(
      "tensor_from_proto",
      [](const TensorProto &tp) {
        // ``TensorFromProto`` returns a borrowed (zero-copy) view into
        // ``tp.raw_data()`` when raw data is used. That view becomes
        // dangling as soon as ``tp`` is garbage-collected on the Python
        // side, so we materialize an owned ``Tensor`` here.
        Tensor t = onnx_kernels::TensorFromProto(tp);
        if (t.data.empty() && t.size_bytes() > 0) {
          std::vector<uint8_t> owned(t.bytes(), t.bytes() + t.size_bytes());
          return Tensor(t.name, t.data_type, t.shape, std::move(owned));
        }
        return t;
      },
      nb::arg("tp"),
      "Converts a ``TensorProto`` to a :class:`Tensor`. Supports all numeric data "
      "types stored either in the typed repeated fields or in the raw little-endian "
      "``raw_data`` field; ``STRING`` tensors are read from ``string_data``. The "
      "returned tensor owns its bytes, so it remains valid after ``tp`` is "
      "garbage-collected.");
}

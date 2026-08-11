// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "_onnxpy_node_list.h"
#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/compute/execute_action.h"
#include "onnx_core/compute/execution_plan.h"
#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/random.h"
#include "onnx_core/runtime/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_core/runtime/simple_tensor.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"

#include <algorithm>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <unordered_set>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::ExecuteAction;
using core::runtime::ExecuteActionKind;
using core::runtime::ExecutionPlan;
using core::runtime::KernelContext;
using core::runtime::Map;
using core::runtime::OpsetId;
using core::runtime::RawBufferAllocator;
using core::runtime::RuntimeContext;
using core::runtime::RuntimeParameters;
using core::runtime::RuntimeSession;
using core::runtime::RuntimeSessionOptions;
using core::runtime::Sequence;
using core::runtime::SimpleRawBufferAllocator;
using core::runtime::Tensor;
using core::runtime::Tensors;

void AddOnnxPyKernels(nb::module_ &m);
void AddOnnxPyRuntime(nb::module_ &m);

namespace {

core::runtime::RuntimeEventKind ParseRuntimeEventKind(const std::string &kind) {
  if (kind == "unknown")
    return core::runtime::RuntimeEventKind::kUnknown;
  if (kind == "initializer")
    return core::runtime::RuntimeEventKind::kInitializer;
  if (kind == "input")
    return core::runtime::RuntimeEventKind::kInput;
  if (kind == "intermediate")
    return core::runtime::RuntimeEventKind::kIntermediate;
  if (kind == "output")
    return core::runtime::RuntimeEventKind::kOutput;
  EXT_THROW_INVALID("RuntimeContext: unknown tensor event kind '", kind,
                    "' (expected one of: unknown, initializer, input, intermediate, output).");
}

int32_t OnnxTypeFromNumpyDtype(nb::handle dtype) {
  const std::string kind = nb::cast<std::string>(nb::str(dtype.attr("kind")));
  const size_t item_size = nb::cast<size_t>(dtype.attr("itemsize"));
  if (kind == "b" && item_size == 1)
    return static_cast<int32_t>(TensorProto::BOOL);
  if (kind == "i") {
    if (item_size == 1)
      return static_cast<int32_t>(TensorProto::INT8);
    if (item_size == 2)
      return static_cast<int32_t>(TensorProto::INT16);
    if (item_size == 4)
      return static_cast<int32_t>(TensorProto::INT32);
    if (item_size == 8)
      return static_cast<int32_t>(TensorProto::INT64);
  }
  if (kind == "u") {
    if (item_size == 1)
      return static_cast<int32_t>(TensorProto::UINT8);
    if (item_size == 2)
      return static_cast<int32_t>(TensorProto::UINT16);
    if (item_size == 4)
      return static_cast<int32_t>(TensorProto::UINT32);
    if (item_size == 8)
      return static_cast<int32_t>(TensorProto::UINT64);
  }
  if (kind == "f") {
    if (item_size == 2)
      return static_cast<int32_t>(TensorProto::FLOAT16);
    if (item_size == 4)
      return static_cast<int32_t>(TensorProto::FLOAT);
    if (item_size == 8)
      return static_cast<int32_t>(TensorProto::DOUBLE);
  }
  if (kind == "V") {
    const std::string name = nb::cast<std::string>(nb::str(dtype));
    if (name == "bfloat16")
      return static_cast<int32_t>(TensorProto::BFLOAT16);
    if (name == "float8_e4m3fn")
      return static_cast<int32_t>(TensorProto::FLOAT8E4M3FN);
    if (name == "float8_e4m3fnuz")
      return static_cast<int32_t>(TensorProto::FLOAT8E4M3FNUZ);
    if (name == "float8_e5m2")
      return static_cast<int32_t>(TensorProto::FLOAT8E5M2);
    if (name == "float8_e5m2fnuz")
      return static_cast<int32_t>(TensorProto::FLOAT8E5M2FNUZ);
  }
  return static_cast<int32_t>(TensorProto::UNDEFINED);
}

Tensor TensorFromNumpy(const std::string &name, nb::handle value, std::vector<nb::object> &owners) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::object array = numpy.attr("asarray")(value);
  int32_t data_type = OnnxTypeFromNumpyDtype(array.attr("dtype"));
  if (data_type == static_cast<int32_t>(TensorProto::UNDEFINED)) {
    nb::object proto =
        nb::module_::import_("onnx_light.onnx_lib.numpy_helper").attr("from_array")(array, name);
    Tensor tensor = core::runtime::TensorFromProto(nb::cast<const TensorProto &>(proto));
    return tensor.is_borrowed() ? tensor.ToOwned() : std::move(tensor);
  }
  if (!nb::cast<bool>(array.attr("flags").attr("c_contiguous"))) {
    array = numpy.attr("ascontiguousarray")(array);
  }
  std::vector<int64_t> shape = nb::cast<std::vector<int64_t>>(array.attr("shape"));
  Py_buffer buffer;
  if (PyObject_GetBuffer(array.ptr(), &buffer, PyBUF_SIMPLE) != 0)
    throw nb::python_error();
  const auto *data = static_cast<const uint8_t *>(buffer.buf);
  const size_t byte_count = static_cast<size_t>(buffer.len);
  if (shape.empty()) {
    core::runtime::RawByteBuffer owned(data, data + byte_count);
    PyBuffer_Release(&buffer);
    return Tensor::FromRawBytes(name, data_type, std::move(shape), std::move(owned));
  }
  PyBuffer_Release(&buffer);
  owners.push_back(std::move(array));
  return Tensor::Borrow(name, data_type, std::move(shape), data, byte_count);
}

void PutMapFromDict(RuntimeContext &rt, const std::string &name, nb::dict dictionary) {
  int64_t size = static_cast<int64_t>(dictionary.size());
  if (size == 0) {
    rt.PutMap(name, Map(name, Tensor{}, Tensor{}));
    return;
  }
  auto iterator = dictionary.begin();
  nb::handle first_key = (*iterator).first;
  nb::handle first_value = (*iterator).second;
  bool string_keys = nb::isinstance<nb::str>(first_key);
  bool integer_keys = nb::isinstance<nb::int_>(first_key) &&
                      !nb::isinstance<nb::bool_>(first_key) &&
                      !nb::isinstance<nb::float_>(first_key);
  bool string_values = nb::isinstance<nb::str>(first_value);
  bool float_values = nb::isinstance<nb::float_>(first_value);
  bool integer_values = nb::isinstance<nb::int_>(first_value) &&
                        !nb::isinstance<nb::bool_>(first_value) &&
                        !nb::isinstance<nb::float_>(first_value);
  if (!string_keys && !integer_keys) {
    throw std::invalid_argument("put_map: unsupported key type in dict - keys must be int or str.");
  }
  if (!string_values && !float_values && !integer_values) {
    throw std::invalid_argument(
        "put_map: unsupported value type in dict - values must be int, float, or str.");
  }

  std::vector<std::string> string_key_buffer, string_value_buffer;
  std::vector<int64_t> integer_key_buffer, integer_value_buffer;
  std::vector<float> float_value_buffer;
  const auto reserve_size = static_cast<size_t>(size);
  if (string_keys)
    string_key_buffer.reserve(reserve_size);
  else
    integer_key_buffer.reserve(reserve_size);
  if (string_values)
    string_value_buffer.reserve(reserve_size);
  else if (float_values)
    float_value_buffer.reserve(reserve_size);
  else
    integer_value_buffer.reserve(reserve_size);
  for (auto [key, value] : dictionary) {
    if (string_keys)
      string_key_buffer.push_back(nb::cast<std::string>(key));
    else
      integer_key_buffer.push_back(nb::cast<int64_t>(key));
    if (string_values)
      string_value_buffer.push_back(nb::cast<std::string>(value));
    else if (float_values)
      float_value_buffer.push_back(nb::cast<float>(value));
    else
      integer_value_buffer.push_back(nb::cast<int64_t>(value));
  }
  Tensor keys = string_keys ? Tensor::FromStrings(name, {size}, string_key_buffer)
                            : Tensor::FromInt64(name, {size}, integer_key_buffer);
  Tensor values = string_values  ? Tensor::FromStrings(name, {size}, string_value_buffer)
                  : float_values ? Tensor::FromFloat(name, {size}, float_value_buffer)
                                 : Tensor::FromInt64(name, {size}, integer_value_buffer);
  rt.PutMap(name, Map(name, std::move(keys), std::move(values)));
}

nb::dlpack::dtype DtypeForTensor(int32_t data_type) {
  using Code = nb::dlpack::dtype_code;
  const auto make = [](Code code, uint8_t bits) {
    return nb::dlpack::dtype{static_cast<uint8_t>(code), bits, 1};
  };
  switch (static_cast<TensorProto::DataType>(data_type)) {
  case TensorProto::FLOAT:
    return make(Code::Float, 32);
  case TensorProto::DOUBLE:
    return make(Code::Float, 64);
  case TensorProto::FLOAT16:
    return make(Code::Float, 16);
  case TensorProto::INT8:
    return make(Code::Int, 8);
  case TensorProto::INT16:
    return make(Code::Int, 16);
  case TensorProto::INT32:
    return make(Code::Int, 32);
  case TensorProto::INT64:
    return make(Code::Int, 64);
  case TensorProto::UINT8:
    return make(Code::UInt, 8);
  case TensorProto::UINT16:
    return make(Code::UInt, 16);
  case TensorProto::UINT32:
    return make(Code::UInt, 32);
  case TensorProto::UINT64:
    return make(Code::UInt, 64);
  case TensorProto::BOOL:
    return make(Code::Bool, 8);
  default:
    return {};
  }
}

nb::object ExoticDtype(int32_t data_type) {
  nb::module_ ml_dtypes = nb::module_::import_("ml_dtypes");
  switch (static_cast<TensorProto::DataType>(data_type)) {
  case TensorProto::BFLOAT16:
    return ml_dtypes.attr("bfloat16");
  case TensorProto::FLOAT8E4M3FN:
    return ml_dtypes.attr("float8_e4m3fn");
  case TensorProto::FLOAT8E4M3FNUZ:
    return ml_dtypes.attr("float8_e4m3fnuz");
  case TensorProto::FLOAT8E5M2:
    return ml_dtypes.attr("float8_e5m2");
  case TensorProto::FLOAT8E5M2FNUZ:
    return ml_dtypes.attr("float8_e5m2fnuz");
  default:
    return nb::none();
  }
}

nb::object TensorToNumpy(Tensor &tensor, RuntimeContext &rt) {
  if (static_cast<TensorProto::DataType>(tensor.data_type) == TensorProto::STRING ||
      DtypeForTensor(tensor.data_type).bits == 0 && ExoticDtype(tensor.data_type).is_none()) {
    TensorProto proto;
    proto.set_data_type(tensor.data_type);
    for (int64_t dimension : tensor.shape)
      proto.add_dims(static_cast<uint64_t>(dimension));
    if (static_cast<TensorProto::DataType>(tensor.data_type) == TensorProto::STRING) {
      for (const std::string &value : tensor.AsStrings())
        proto.add_string_data(utils::String(value));
    } else {
      proto.ref_raw_data().assign(tensor.bytes(), tensor.size_bytes());
    }
    return nb::module_::import_("onnx_light.onnx_lib.numpy_helper")
        .attr("to_array")(nb::cast(std::move(proto)));
  }

  std::vector<size_t> shape;
  shape.reserve(tensor.shape.size());
  for (int64_t dimension : tensor.shape)
    shape.push_back(static_cast<size_t>(dimension));

  const size_t byte_count = tensor.size_bytes();
  nb::object owner;
  const uint8_t *data = tensor.bytes();
  if (!tensor.has_allocation() && data == tensor.data.data()) {
    auto *owned = new core::runtime::RawByteBuffer(tensor.data.release());
    data = owned->data();
    owner = nb::capsule(owned, [](void *pointer) noexcept {
      delete static_cast<core::runtime::RawByteBuffer *>(pointer);
    });
  } else {
    owner = nb::cast(&rt, nb::rv_policy::reference);
  }

  nb::dlpack::dtype dtype = DtypeForTensor(tensor.data_type);
  if (dtype.bits != 0) {
    nb::ndarray<nb::numpy, nb::ro> array(data, shape.size(), shape.data(), owner, nullptr, dtype);
    return nb::cast(std::move(array));
  }

  nb::ndarray<nb::numpy, const uint8_t> raw(data, {byte_count}, owner);
  nb::object array = nb::cast(std::move(raw)).attr("view")(ExoticDtype(tensor.data_type));
  return array.attr("reshape")(nb::cast(shape));
}

class ReferenceEvaluatorRunner {
public:
  ReferenceEvaluatorRunner(const GraphProto &graph, std::vector<std::string> input_names,
                           std::unordered_set<std::string> map_inputs,
                           std::unordered_set<std::string> sequence_inputs,
                           std::vector<std::string> output_names)
      : graph_(&graph), input_names_(std::move(input_names)), map_inputs_(std::move(map_inputs)),
        sequence_inputs_(std::move(sequence_inputs)), output_names_(std::move(output_names)),
        output_names_set_(output_names_.begin(), output_names_.end()) {}

  ReferenceEvaluatorRunner(const FunctionProto &function, std::vector<std::string> input_names,
                           std::unordered_set<std::string> map_inputs,
                           std::unordered_set<std::string> sequence_inputs,
                           std::vector<std::string> output_names)
      : function_(&function), input_names_(std::move(input_names)),
        map_inputs_(std::move(map_inputs)), sequence_inputs_(std::move(sequence_inputs)),
        output_names_(std::move(output_names)),
        output_names_set_(output_names_.begin(), output_names_.end()) {}

  void Reset() {
    session_.reset();
    plan_.reset();
  }

  nb::list Run(RuntimeContext &rt, nb::object requested_outputs, nb::dict feeds,
               bool release_intermediates) {
    std::vector<std::string> outputs = requested_outputs.is_none()
                                           ? output_names_
                                           : nb::cast<std::vector<std::string>>(requested_outputs);
    for (const std::string &name : input_names_) {
      if (!feeds.contains(name.c_str())) {
        throw std::invalid_argument("Missing input '" + name + "' for ReferenceEvaluator.run.");
      }
    }

    bool outputs_are_declared =
        std::all_of(outputs.begin(), outputs.end(),
                    [this](const std::string &name) { return output_names_set_.contains(name); });
    rt.Clear();
    rt.set_release_intermediates(release_intermediates && outputs_are_declared);
    std::vector<nb::object> input_owners;

    for (auto [key, value] : feeds) {
      std::string name = nb::cast<std::string>(key);
      if (map_inputs_.contains(name)) {
        nb::object map_value = nb::borrow<nb::object>(value);
        if (!nb::isinstance<nb::dict>(map_value) && nb::hasattr(map_value, "dtype") &&
            nb::cast<size_t>(map_value.attr("size")) == 1) {
          map_value = map_value.attr("item")();
        }
        if (!nb::isinstance<nb::dict>(map_value))
          throw nb::type_error(("Map input '" + name + "' must be a Python dict.").c_str());
        PutMapFromDict(rt, name, nb::cast<nb::dict>(map_value));
      } else if (sequence_inputs_.contains(name)) {
        if (!nb::isinstance<nb::list>(value) && !nb::isinstance<nb::tuple>(value))
          throw nb::type_error(("Sequence input '" + name + "' must be a list or tuple.").c_str());
        Tensors tensors;
        for (nb::handle element : nb::borrow<nb::sequence>(value))
          tensors.push_back(TensorFromNumpy(name, element, input_owners));
        int32_t element_type =
            tensors.empty() ? 0 : static_cast<int32_t>(tensors.front().data_type);
        rt.PutSequence(name, Sequence(name, element_type, std::move(tensors)));
      } else {
        rt.Set(name, TensorFromNumpy(name, value, input_owners));
      }
    }

    if (graph_ != nullptr) {
      for (const TensorProto &initializer : graph_->initializer()) {
        if (!rt.Has(initializer.name()))
          rt.Set(initializer.name().value(), core::runtime::TensorFromProto(initializer),
                 core::runtime::RuntimeEventKind::kInitializer);
      }
    }
    EnsureSession();
    session_->Run(rt);

    nb::list results;
    for (const std::string &name : outputs) {
      if (rt.Has(name)) {
        results.append(TensorToNumpy(rt.Get(name), rt));
      } else if (rt.HasSequence(name)) {
        nb::list sequence;
        for (Tensor &tensor : rt.GetSequence(name).values)
          sequence.append(TensorToNumpy(tensor, rt));
        results.append(std::move(sequence));
      } else {
        throw std::runtime_error("Output '" + name + "' was not produced by the graph.");
      }
    }
    return results;
  }

private:
  void EnsureSession() {
    if (session_)
      return;
    if (graph_ != nullptr) {
      plan_ = std::make_unique<ExecutionPlan>(*graph_);
      session_ = std::make_unique<RuntimeSession>(*plan_);
      session_->SetDeclaredShapes(*graph_);
    } else {
      plan_ = std::make_unique<ExecutionPlan>(*function_);
      session_ = std::make_unique<RuntimeSession>(*plan_);
    }
  }

  const GraphProto *graph_ = nullptr;
  const FunctionProto *function_ = nullptr;
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> map_inputs_;
  std::unordered_set<std::string> sequence_inputs_;
  std::vector<std::string> output_names_;
  std::unordered_set<std::string> output_names_set_;
  std::unique_ptr<ExecutionPlan> plan_;
  std::unique_ptr<RuntimeSession> session_;
};

} // namespace

NB_MODULE(_onnxpykernels, m) {
  m.doc() = "onnx_light kernels bindings: deterministic pseudo-random helpers "
            "backing _onnxpybackend_test, plus the RunNode dispatcher "
            "(and the RuntimeSession every node list, including a whole model, is run "
            "through) and its supporting RuntimeContext/KernelContext types.";

  // The ``runtime`` submodule exposes :cpp:func:`RunNode` and
  // :cpp:class:`RuntimeSession` (used to run any node list: a bare graph, a
  // function body, or a whole model's graph, paired with
  // :cpp:func:`RegisterModelFunctions` for model-local functions). These
  // take/return ``Tensor`` and proto types whose
  // ``nb::class_`` bindings live in sibling extensions. Importing those
  // extensions here guarantees the cross-module typeid registry has the
  // necessary entries by the time we register the ``runtime`` callables and
  // makes the runtime submodule usable even when consumers import
  // ``_onnxpykernels`` directly (bypassing the ``_onnxpy.py`` shim).
  nb::module_::import_("onnx_light.onnx_py._onnxpyprotoop");
  nb::module_::import_("onnx_light.onnx_py._onnxpybackend");

  AddOnnxPyKernels(m);
  AddOnnxPyRuntime(m);

  // Register all built-in onnx_kernels operator factories with the
  // core::runtime dispatch table so that RunNode/RuntimeSession work as
  // soon as the module is imported.
  onnx_kernels::RegisterKernelFunctions();
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
      "next_uint64", [](uint64_t state) { return core::runtime::NextUint64(state); },
      nb::arg("state"), "Returns ``(next_state, value)`` for the SplitMix64 generator.");

  backend_mod.def(
      "rand",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return core::runtime::Rand(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic uniform values in ``[0, 1)`` as a flat list.");

  backend_mod.def(
      "randint",
      [](int64_t low, int64_t high, const std::vector<int64_t> &shape,
         std::optional<uint64_t> seed) { return core::runtime::RandInt(low, high, shape, seed); },
      nb::arg("low"), nb::arg("high"), nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` deterministic integers in ``[low, high)`` as a flat list.");

  backend_mod.def(
      "randn",
      [](const std::vector<int64_t> &shape, std::optional<uint64_t> seed) {
        return core::runtime::Randn(shape, seed);
      },
      nb::arg("shape"), nb::arg("seed") = nb::none(),
      "Returns ``prod(shape)`` approximately normal-distributed values (Irwin-Hall) "
      "as a flat list.");
}

void AddOnnxPyRuntime(nb::module_ &m) {
  // -----------------------------------------------------------------------
  // Submodule `runtime`
  // Exposes the C++ RunNode dispatcher and RuntimeSession (used to run any
  // node list, including a whole model's graph paired with
  // RegisterModelFunctions) together with their supporting RuntimeContext /
  // KernelContext / OpsetId types and the TensorFromProto helper.
  // -----------------------------------------------------------------------
  auto rt_mod = m.def_submodule("runtime");
  rt_mod.doc() = "C++ kernel dispatcher exposed to Python. RunNode and "
                 "RuntimeSession evaluate one or more nodes through the static "
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
      "default_opset", [](int64_t version) { return core::runtime::DefaultOpset(version); },
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

  // RuntimeEventAction — enum classifying a RuntimeEvent record.
  // Mirrors :cpp:enum:`core::runtime::RuntimeEventAction`.
  nb::enum_<core::runtime::RuntimeEventAction>(rt_mod, "RuntimeEventAction", nb::is_arithmetic(),
                                               "Action kind recorded in a :class:`RuntimeEvent`. "
                                               "``kAdd`` / ``kReplace`` / ``kRemove`` mark tensor "
                                               "map mutations; ``kRunNode`` marks the dispatch of "
                                               "a single kernel.")
      .value("kAdd", core::runtime::RuntimeEventAction::kAdd,
             "A new tensor was inserted into the runtime tensor map.")
      .value("kReplace", core::runtime::RuntimeEventAction::kReplace,
             "An existing tensor in the runtime map was replaced.")
      .value("kRemove", core::runtime::RuntimeEventAction::kRemove,
             "A tensor was removed from the runtime tensor map.")
      .value("kRunNode", core::runtime::RuntimeEventAction::kRunNode,
             "A kernel was dispatched for a node.");

  // RuntimeEvent — append-only log entry for a single tensor map mutation.
  // Mirrors :cpp:class:`core::runtime::RuntimeEvent`; ``values`` / ``string_values``
  // expose the populated prefix of the underlying fixed-size buffer as Python
  // lists of length ``value_count``.
  nb::class_<core::runtime::RuntimeEvent>(
      rt_mod, "RuntimeEvent",
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
          "action", [](const core::runtime::RuntimeEvent &ev) { return ev.action; },
          ":class:`RuntimeEventAction` member describing the mutation kind: "
          "``kAdd``, ``kReplace``, ``kRemove`` or ``kRunNode``.")
      .def_prop_ro(
          "kind",
          [](const core::runtime::RuntimeEvent &ev) {
            return std::string(core::runtime::RuntimeEventKindName(ev.kind));
          },
          "Tensor role: ``\"unknown\"``, ``\"initializer\"``, ``\"input\"``, "
          "``\"intermediate\"`` or ``\"output\"``.")
      .def_ro("timestamp_ns", &core::runtime::RuntimeEvent::timestamp_ns,
              "Nanoseconds since the Unix epoch (``std::chrono::system_clock``).")
      .def_ro("name", &core::runtime::RuntimeEvent::name, "Tensor name targeted by the mutation.")
      .def_ro("data_type", &core::runtime::RuntimeEvent::data_type,
              "``TensorProto.DataType`` integer of the tensor, or ``-1`` when "
              "the payload was truncated (more than 8 elements).")
      .def_ro("shape", &core::runtime::RuntimeEvent::shape,
              "Tensor shape, or empty list when truncated / for ``remove`` events.")
      .def_ro("value_count", &core::runtime::RuntimeEvent::value_count,
              "Number of populated entries in :attr:`values` / :attr:`string_values` "
              "(``min(element_count, 8)``, ``0`` for ``remove`` and ``run_node`` events).")
      .def_ro("op_domain", &core::runtime::RuntimeEvent::op_domain,
              "For ``run_node`` events: normalised ONNX op domain of the dispatched "
              "node (default domain reported as ``\"ai.onnx\"``). Empty for other "
              "event actions.")
      .def_ro("op_type", &core::runtime::RuntimeEvent::op_type,
              "For ``run_node`` events: ONNX ``op_type`` of the dispatched node. "
              "Empty for other event actions.")
      .def_ro("inputs", &core::runtime::RuntimeEvent::inputs,
              "For ``run_node`` events: ordered list of input names consumed by the "
              "node (matching ``NodeProto.input``). Empty for other event actions.")
      .def_ro("duration_ns", &core::runtime::RuntimeEvent::duration_ns,
              "For ``run_node`` events: wall-clock duration of the kernel dispatch in "
              "nanoseconds (``std::chrono::steady_clock``). ``0`` for other event "
              "actions.")
      .def_ro("node_index", &core::runtime::RuntimeEvent::node_index,
              "Index of the node this event is associated with: ``-1`` for graph "
              "inputs, ``-2`` for initializers, and the ``>= 0`` position of the "
              "producing / dispatched node otherwise.")
      .def_ro("device", &core::runtime::RuntimeEvent::device,
              "Device the tensor lives on: ``-1`` for the CPU and ``0``–``8192`` for "
              "a GPU device index. The CPU reference runtime always reports ``-1``.")
      .def_ro("subgraph_node_index", &core::runtime::RuntimeEvent::subgraph_node_index,
              "Index of the control-flow node in the parent graph whose attribute subgraph "
              "produced this event. ``-1`` for top-level-graph events.")
      .def_ro("subgraph_attr_name", &core::runtime::RuntimeEvent::subgraph_attr_name,
              "Attribute name of the subgraph within the owning control-flow node "
              "(``\"body\"``, ``\"then_branch\"``, ``\"else_branch\"``, etc.). "
              "Empty for top-level-graph events.")
      .def_ro("allocated_bytes", &core::runtime::RuntimeEvent::allocated_bytes,
              "Total number of bytes held by every live buffer in the "
              ":class:`RuntimeContext`'s allocator right after the action that "
              "produced this event (the runtime's live memory footprint). ``0`` "
              "when no allocator is attached to the context.")
      .def_ro("peak_bytes", &core::runtime::RuntimeEvent::peak_bytes,
              "Peak value ever reached by :attr:`allocated_bytes` up to the moment "
              "this event was recorded. ``0`` when no allocator is attached.")
      .def_prop_ro(
          "values",
          [](const core::runtime::RuntimeEvent &ev) {
            nb::list out;
            if (static_cast<core::runtime::DataType>(ev.data_type) !=
                core::runtime::DataType::STRING) {
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
          [](const core::runtime::RuntimeEvent &ev) {
            nb::list out;
            if (static_cast<core::runtime::DataType>(ev.data_type) ==
                core::runtime::DataType::STRING) {
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
          [](const core::runtime::RuntimeEvent &ev) {
            nb::dict d;
            d["action"] = std::string(core::runtime::RuntimeEventActionName(ev.action));
            d["kind"] = std::string(core::runtime::RuntimeEventKindName(ev.kind));
            d["timestamp_ns"] = ev.timestamp_ns;
            d["name"] = ev.name;
            d["data_type"] = ev.data_type;
            d["shape"] = ev.shape;
            d["value_count"] = ev.value_count;
            d["op_domain"] = ev.op_domain;
            d["op_type"] = ev.op_type;
            d["inputs"] = ev.inputs;
            d["duration_ns"] = ev.duration_ns;
            d["node_index"] = ev.node_index;
            d["device"] = ev.device;
            d["subgraph_node_index"] = ev.subgraph_node_index;
            d["subgraph_attr_name"] = ev.subgraph_attr_name;
            d["allocated_bytes"] = ev.allocated_bytes;
            d["peak_bytes"] = ev.peak_bytes;
            const int32_t n = ev.value_count;
            if (static_cast<core::runtime::DataType>(ev.data_type) ==
                core::runtime::DataType::STRING) {
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
      .def("summary", &core::runtime::RuntimeEvent::summary,
           "Returns a concise, human-readable one-line summary of the event: the "
           "action / kind, the tensor name (or ``op_type(inputs)`` for ``run_node`` "
           "events), the node index, the dispatch duration (for ``run_node`` events) "
           "and the allocator's live (``allocated_bytes``) and peak (``peak_bytes``) "
           "memory.")
      .def("__repr__", [](const core::runtime::RuntimeEvent &ev) {
        return std::string("RuntimeEvent(action='") +
               core::runtime::RuntimeEventActionName(ev.action) + "', kind='" +
               core::runtime::RuntimeEventKindName(ev.kind) + "', name='" + ev.name +
               "', data_type=" + std::to_string(ev.data_type) +
               ", value_count=" + std::to_string(ev.value_count) +
               ", node_index=" + std::to_string(ev.node_index) +
               ", device=" + std::to_string(ev.device) +
               ", subgraph_node_index=" + std::to_string(ev.subgraph_node_index) +
               ", subgraph_attr_name='" + ev.subgraph_attr_name +
               "', allocated_bytes=" + std::to_string(ev.allocated_bytes) +
               ", peak_bytes=" + std::to_string(ev.peak_bytes) + ")";
      });

  // ExecuteActionKind — kind of a single ExecuteAction.
  nb::enum_<ExecuteActionKind>(rt_mod, "ExecuteActionKind", nb::is_arithmetic(),
                               "Kind of a single :class:`ExecuteAction` scheduled by an "
                               ":class:`ExecutionPlan`.")
      .value("kLockInitializer", ExecuteActionKind::kLockInitializer,
             "Locks an initializer so it stays alive while still referenced.")
      .value("kUnlockInitializer", ExecuteActionKind::kUnlockInitializer,
             "Unlocks an initializer once no remaining node references it.")
      .value("kLockInput", ExecuteActionKind::kLockInput,
             "Locks an input so it stays alive while still referenced.")
      .value("kUnlockInput", ExecuteActionKind::kUnlockInput,
             "Unlocks an input once no remaining node references it.")
      .value("kAllocateBuffer", ExecuteActionKind::kAllocateBuffer,
             "Allocates a buffer for a named result.")
      .value("kDeleteBuffer", ExecuteActionKind::kDeleteBuffer,
             "Deletes a named result (frees its buffer).")
      .value("kTransfer", ExecuteActionKind::kTransfer,
             "Transfers a named result to another named result.")
      .value("kExecuteNode", ExecuteActionKind::kExecuteNode, "Executes a node.")
      .value("kCreateShape", ExecuteActionKind::kCreateShape,
             "Creates the shape of a named result.")
      .value("kDeleteShape", ExecuteActionKind::kDeleteShape,
             "Deletes the shape of a named result.")
      .value("kAllocateTemporaryBuffer", ExecuteActionKind::kAllocateTemporaryBuffer,
             "Allocates a temporary buffer to handle a kernel memory peak.")
      .value("kDeleteTemporaryBuffer", ExecuteActionKind::kDeleteTemporaryBuffer,
             "Deallocates a temporary buffer once the kernel(s) are done.")
      .value("kDeleteSequence", ExecuteActionKind::kDeleteSequence,
             "Deletes a named sequence result (frees the sequence it holds).")
      .value("kDeleteMap", ExecuteActionKind::kDeleteMap,
             "Deletes a named map result (frees the map it holds).");

  // ExecuteAction — single step of an ExecutionPlan.
  nb::class_<ExecuteAction>(
      rt_mod, "ExecuteAction",
      "Single step of an :class:`ExecutionPlan`: one memory-management or "
      "node-execution operation (lock / unlock an input or initializer, "
      "allocate / delete a named result or temporary buffer, create / delete a "
      "shape, transfer a named result, or execute a node).")
      .def(nb::init<>())
      .def(nb::init<ExecuteActionKind, std::string>(), nb::arg("kind"), nb::arg("name"),
           "Builds an action of ``kind`` targeting ``name``.")
      .def_prop_ro(
          "kind", [](const ExecuteAction &a) { return a.kind(); },
          "Kind of the action (:class:`ExecuteActionKind`).")
      .def_prop_ro(
          "kind_name", [](const ExecuteAction &a) { return std::string(a.kind_name()); },
          "Stable, human-readable name of :attr:`kind`.")
      .def_prop_ro(
          "name", [](const ExecuteAction &a) { return a.name(); },
          "Primary named result / input / initializer the action operates on.")
      .def_prop_ro(
          "target", [](const ExecuteAction &a) { return a.target(); },
          "Destination named result of a ``kTransfer`` action, or the input "
          "buffer reused by an in-place ``kAllocateBuffer`` (empty otherwise).")
      .def_prop_ro(
          "node_index", [](const ExecuteAction &a) { return a.node_index(); },
          "Index of the node for a ``kExecuteNode`` action (``0`` otherwise).")
      .def_prop_ro(
          "size", [](const ExecuteAction &a) { return a.size(); },
          "Number of bytes for buffer allocations (``0`` when unknown).")
      .def_prop_ro(
          "is_inplace", [](const ExecuteAction &a) { return a.is_inplace(); },
          "Whether a ``kAllocateBuffer`` action reuses an input buffer in place "
          "instead of allocating fresh memory.")
      .def_prop_ro(
          "inplace_output_index", [](const ExecuteAction &a) { return a.inplace().output_index; },
          "Output index of the in-place reuse decision (negative when the action "
          "is not an in-place allocation).")
      .def_prop_ro(
          "inplace_input_index", [](const ExecuteAction &a) { return a.inplace().input_index; },
          "Input index reused in place (negative when the action is not an "
          "in-place allocation).")
      .def(
          "summary", [](const ExecuteAction &a) { return a.summary(); },
          "Returns a concise, human-readable one-line summary of the action, "
          "including only the fields relevant to its :attr:`kind`.")
      .def("__repr__", [](const ExecuteAction &a) {
        return std::string("ExecuteAction(kind='") + a.kind_name() + "', name='" + a.name() +
               "', target='" + a.target() + "', node_index=" + std::to_string(a.node_index()) +
               ", size=" + std::to_string(a.size()) +
               ", is_inplace=" + (a.is_inplace() ? "True" : "False") + ")";
      });

  // ExecutionPlan — precomputed per-graph execution / release schedule.
  nb::class_<ExecutionPlan>(
      rt_mod, "ExecutionPlan",
      "Precomputed per-graph execution schedule used by :class:`RuntimeSession` "
      "when :attr:`RuntimeContext.release_intermediates` is enabled. Captures the "
      "structural set of names that must never be released (:func:`keep`) and "
      "the ordered list of :class:`ExecuteAction` steps (:func:`actions`) "
      "derived from the in-place / lifetime metadata written to each node by "
      "the in-place reuse pass. Depends only on graph topology / metadata, so a "
      "single plan can be reused across every invocation of the same model.")
      .def(nb::init<>())
      .def(nb::init<const GraphProto &>(), nb::arg("graph"), nb::keep_alive<1, 2>(),
           "Builds the plan for ``graph``. ``keep`` is seeded with the graph's "
           "declared inputs, initializers and declared outputs. This binding keeps "
           "``graph`` alive for at least as long as the plan, since the plan's node "
           "list holds non-owning pointers into it.")
      .def(nb::init<const FunctionProto &>(), nb::arg("func"), nb::keep_alive<1, 2>(),
           "Builds the plan for ``func``. ``keep`` is seeded with the function's "
           "declared inputs and outputs. This binding keeps ``func`` alive for at "
           "least as long as the plan, since the plan's node list holds non-owning "
           "pointers into it.")
      .def(
          "keep", [](const ExecutionPlan &plan) { return plan.keep(); },
          "Returns the structural set of names that must never be released.")
      .def_prop_ro(
          "num_nodes", [](const ExecutionPlan &plan) { return plan.num_nodes(); },
          "Number of nodes covered by this plan.")
      .def(
          "actions", [](const ExecutionPlan &plan) { return plan.actions(); },
          "Returns the ordered list of :class:`ExecuteAction` steps the runtime "
          "performs while executing the underlying node sequence.")
      .def(
          "release_after",
          [](const ExecutionPlan &plan, const NodeProto &node, RuntimeContext &rt) {
            plan.ReleaseAfter(node, rt);
          },
          nb::arg("node"), nb::arg("rt"),
          "Releases from ``rt`` every intermediate whose last use falls at "
          "``node``. ``node`` must be one of the nodes the plan was built from "
          "(lookup is by identity); otherwise this is a no-op.");

  // RuntimeSessionOptions — bundled construction knobs for RuntimeSession.
  nb::class_<RuntimeSessionOptions>(
      rt_mod, "RuntimeSessionOptions",
      "Bundled construction options for :class:`RuntimeSession`, grouping the "
      "per-session knobs (:attr:`parameters`, :attr:`verbose`, :attr:`check_shapes`, "
      ":attr:`allow_external_output_allocators`) so they can be passed as a single "
      "``options`` argument instead of individual keyword arguments.")
      .def(
          "__init__",
          [](RuntimeSessionOptions *self, nb::object parameters_obj, int verbose, bool check_shapes,
             bool allow_external_output_allocators) {
            RuntimeParameters parameters = parameters_obj.is_none()
                                               ? RuntimeParameters()
                                               : nb::cast<RuntimeParameters>(parameters_obj);
            new (self) RuntimeSessionOptions{
                .parameters = std::move(parameters),
                .verbose = verbose,
                .check_shapes = check_shapes,
                .allow_external_output_allocators = allow_external_output_allocators,
            };
          },
          nb::kw_only(), nb::arg("parameters").none() = nb::none(), nb::arg("verbose") = 0,
          nb::arg("check_shapes") = false, nb::arg("allow_external_output_allocators") = false,
          "Builds the options bundle. All fields are keyword-only and optional; each "
          "defaults to the same value :class:`RuntimeSession` uses when the field is "
          "omitted.")
      .def_rw("parameters", &RuntimeSessionOptions::parameters,
              "Model-independent execution parameters applied to the nodes the session runs.")
      .def_rw("verbose", &RuntimeSessionOptions::verbose,
              "Verbosity level requested for :func:`RuntimeSession.run`.")
      .def_rw("check_shapes", &RuntimeSessionOptions::check_shapes,
              "When ``True``, :func:`RuntimeSession.run` validates concrete tensor shapes "
              "against the declared (possibly symbolic) shapes.")
      .def_rw("allow_external_output_allocators",
              &RuntimeSessionOptions::allow_external_output_allocators,
              "When ``True``, :func:`RuntimeSession.run` does not require a node's "
              "allocator-backed outputs to be owned by the session's unique allocator.");

  // RuntimeSession — reusable execution session binding an ExecutionPlan.
  // Every node list the runtime executes (a graph, a function body, a
  // subgraph, ...) is run through one of these sessions; there is no
  // standalone "run this node list" function.
  nb::class_<RuntimeSession>(
      rt_mod, "RuntimeSession",
      "Reusable execution session binding a precomputed :class:`ExecutionPlan`. "
      "The first call to :func:`run` resolves, builds, and caches the kernel "
      "instance for every scheduled node against the supplied "
      ":class:`RuntimeContext`; subsequent calls reuse those cached instances. When "
      ":attr:`RuntimeContext.release_intermediates` is enabled, :func:`run` also "
      "frees each intermediate whose last reference has been reached.")
      .def(
          "__init__",
          [](RuntimeSession *self, const ModelProto &model, nb::object parameters_obj, int verbose,
             bool check_shapes, bool allow_external_output_allocators) {
            RuntimeParameters parameters = parameters_obj.is_none()
                                               ? RuntimeParameters()
                                               : nb::cast<RuntimeParameters>(parameters_obj);
            new (self) RuntimeSession(
                model, core::runtime::RuntimeSessionOptions{
                           .parameters = std::move(parameters),
                           .verbose = verbose,
                           .check_shapes = check_shapes,
                           .allow_external_output_allocators = allow_external_output_allocators,
                       });
          },
          nb::arg("model"), nb::kw_only(), nb::arg("parameters").none() = nb::none(),
          nb::arg("verbose") = 0, nb::arg("check_shapes") = false,
          nb::arg("allow_external_output_allocators") = false, nb::keep_alive<1, 2>(),
          "Builds a session over an :class:`ExecutionPlan` the session owns, "
          "built from ``model.graph``. Use this when no precomputed plan is "
          "available: ``model`` (and the graph it owns) must outlive the session; "
          "this binding keeps ``model`` alive for at least as long as the session.")
      .def(
          "__init__",
          [](RuntimeSession *self, const ExecutionPlan &plan, nb::object parameters_obj,
             int verbose, bool check_shapes, bool allow_external_output_allocators) {
            RuntimeParameters parameters = parameters_obj.is_none()
                                               ? RuntimeParameters()
                                               : nb::cast<RuntimeParameters>(parameters_obj);
            new (self) RuntimeSession(
                plan, core::runtime::RuntimeSessionOptions{
                          .parameters = std::move(parameters),
                          .verbose = verbose,
                          .check_shapes = check_shapes,
                          .allow_external_output_allocators = allow_external_output_allocators,
                      });
          },
          nb::arg("plan"), nb::kw_only(), nb::arg("parameters").none() = nb::none(),
          nb::arg("verbose") = 0, nb::arg("check_shapes") = false,
          nb::arg("allow_external_output_allocators") = false, nb::keep_alive<1, 2>(),
          "Builds a session over ``plan``. ``plan`` (and the graph / function it "
          "was built from) must outlive the session; this binding keeps ``plan`` "
          "alive for at least as long as the session.")
      .def(
          "__init__",
          [](RuntimeSession *self, const ModelProto &model, const RuntimeSessionOptions &options) {
            new (self) RuntimeSession(model, options);
          },
          nb::arg("model"), nb::arg("options"), nb::keep_alive<1, 2>(),
          "Builds a session over an :class:`ExecutionPlan` the session owns, built "
          "from ``model.graph``, using a :class:`RuntimeSessionOptions` bundle. "
          "``model`` must outlive the session; this binding keeps it alive.")
      .def(
          "__init__",
          [](RuntimeSession *self, const ExecutionPlan &plan,
             const RuntimeSessionOptions &options) { new (self) RuntimeSession(plan, options); },
          nb::arg("plan"), nb::arg("options"), nb::keep_alive<1, 2>(),
          "Builds a session over ``plan`` using a :class:`RuntimeSessionOptions` "
          "bundle. ``plan`` must outlive the session; this binding keeps it alive.")
      .def("run", &RuntimeSession::Run, nb::arg("rt"),
           "Executes the plan once against ``rt``, resolving, building, and caching "
           "kernel instances on the first call. Safe to call repeatedly on the same session.")
      .def_prop_ro("parameters", &RuntimeSession::parameters,
                   "Model-independent execution parameters applied to the nodes this "
                   "session runs.")
      .def_prop_ro("check_shapes", &RuntimeSession::check_shapes,
                   "When ``True``, :func:`run` validates that the concrete shape of every "
                   "tensor carrying a declared (possibly symbolic) shape — the graph "
                   "inputs, outputs and ``value_info`` — is consistent with that "
                   "declaration: concrete dimensions must match exactly and every symbolic "
                   "``dim_param`` must resolve to the same value everywhere it appears "
                   "during a single run. Declared shapes are populated automatically when "
                   "the session is built from a model; a session built from an "
                   ":class:`ExecutionPlan` must call :func:`set_declared_shapes` first. "
                   "Disabled by default.")
      .def_prop_ro("allow_external_output_allocators",
                   &RuntimeSession::allow_external_output_allocators,
                   "When ``True``, :func:`run` does not require a node's allocator-backed "
                   "outputs to be owned by the session's unique allocator, so a kernel may "
                   "return an output allocated outside the common allocator. When ``False`` "
                   "(the default), such an output is rejected.")
      .def("set_declared_shapes", &RuntimeSession::SetDeclaredShapes, nb::arg("graph"),
           "Records the declared (possibly symbolic) shapes carried by ``graph``'s "
           "inputs, outputs and ``value_info`` so that, when :attr:`check_shapes` is "
           "enabled, :func:`run` can validate concrete tensor shapes against them.")
      .def_prop_ro("verbose", &RuntimeSession::verbose,
                   "Verbosity level requested for :func:`run`. When non-zero, it overrides "
                   "the :class:`RuntimeContext` verbosity for this session's progress "
                   "lines without mutating the context itself.")
      .def_prop_ro("required_inputs", &RuntimeSession::required_inputs,
                   "Returns the external input names the scheduled nodes read. Populated "
                   "during kernel initialization; empty until the first :func:`run`.");

  nb::class_<ReferenceEvaluatorRunner>(
      rt_mod, "ReferenceEvaluatorRunner",
      "Native hot path used by ReferenceEvaluator.run to adapt Python values, execute a "
      "cached RuntimeSession, and expose its outputs in one Python-to-C++ call.")
      .def(
          "__init__",
          [](ReferenceEvaluatorRunner *self, const GraphProto &graph,
             std::vector<std::string> input_names, std::unordered_set<std::string> map_inputs,
             std::unordered_set<std::string> sequence_inputs,
             std::vector<std::string> output_names) {
            new (self)
                ReferenceEvaluatorRunner(graph, std::move(input_names), std::move(map_inputs),
                                         std::move(sequence_inputs), std::move(output_names));
          },
          nb::arg("graph"), nb::arg("input_names"), nb::arg("map_inputs"),
          nb::arg("sequence_inputs"), nb::arg("output_names"), nb::keep_alive<1, 2>())
      .def(
          "__init__",
          [](ReferenceEvaluatorRunner *self, const FunctionProto &function,
             std::vector<std::string> input_names, std::unordered_set<std::string> map_inputs,
             std::unordered_set<std::string> sequence_inputs,
             std::vector<std::string> output_names) {
            new (self)
                ReferenceEvaluatorRunner(function, std::move(input_names), std::move(map_inputs),
                                         std::move(sequence_inputs), std::move(output_names));
          },
          nb::arg("function"), nb::arg("input_names"), nb::arg("map_inputs"),
          nb::arg("sequence_inputs"), nb::arg("output_names"), nb::keep_alive<1, 2>())
      .def("run", &ReferenceEvaluatorRunner::Run, nb::arg("runtime_context"),
           nb::arg("output_names").none() = nb::none(), nb::arg("feed_inputs"),
           nb::arg("release_intermediates") = true,
           "Runs one inference entirely through the native input/execution/output path.")
      .def("reset", &ReferenceEvaluatorRunner::Reset,
           "Drops the cached execution plan and RuntimeSession so kernels resolve again.");

  // RuntimeParameters — model-independent execution knobs (parallelism, ...).
  nb::class_<RuntimeParameters>(
      rt_mod, "RuntimeParameters",
      "Model-independent execution settings shared across the nodes of a graph "
      "evaluated through :func:`run_node` / :class:`RuntimeSession`. Currently carries only "
      "the requested degree of parallelism, :attr:`num_threads`.")
      .def(nb::init<>())
      .def(nb::init<int32_t>(), nb::arg("num_threads"))
      .def_rw("num_threads", &RuntimeParameters::num_threads,
              "Number of threads used to parallelize the execution of a graph. ``0`` "
              "(default) uses the number of CPU cores, ``1`` disables parallelization, "
              "``> 1`` uses exactly that many worker threads, and any negative value is "
              "treated like ``0``.")
      .def("effective_num_threads", &RuntimeParameters::EffectiveNumThreads,
           "Resolves :attr:`num_threads` to a concrete count: ``0`` and negative values "
           "become the number of CPU cores (falling back to ``1``); every other value is "
           "returned unchanged. The result is always ``>= 1``.")
      .def("is_parallel", &RuntimeParameters::is_parallel,
           "Returns ``True`` when the graph should run with more than one thread, i.e. "
           "when :meth:`effective_num_threads` is greater than ``1``.");

  // SimpleRawBufferAllocator — fixed-capacity pool allocator exposing live and
  // peak memory so RuntimeEvent.allocated_bytes / peak_bytes become non-zero.
  nb::class_<SimpleRawBufferAllocator>(
      rt_mod, "SimpleRawBufferAllocator",
      "Fixed-capacity pool allocator for the runtime's raw buffers. Attach one "
      "to a :class:`RuntimeContext` via the :class:`RuntimeContext` constructor so "
      "the runtime routes tensor storage through it; every recorded "
      ":class:`RuntimeEvent` then carries the allocator's live "
      "(:attr:`RuntimeEvent.allocated_bytes`) and peak "
      "(:attr:`RuntimeEvent.peak_bytes`) memory. ``capacity`` is the maximum "
      "number of buffers that may be alive at the same time.")
      .def(nb::init<size_t>(), nb::arg("capacity"))
      .def_prop_ro("total_allocated_size", &SimpleRawBufferAllocator::TotalAllocatedSize,
                   "Total number of bytes across all currently live buffers.")
      .def_prop_ro("peak_allocated_size", &SimpleRawBufferAllocator::PeakAllocatedSize,
                   "Maximum value ever reached by :attr:`total_allocated_size` since "
                   "construction or the last :meth:`reset_peak`.")
      .def_prop_ro("allocated_count", &SimpleRawBufferAllocator::allocated_count,
                   "Number of buffer slots currently in use.")
      .def_prop_ro("capacity", &SimpleRawBufferAllocator::capacity,
                   "Maximum number of buffers that may be alive at the same time.")
      .def("reset_peak", &SimpleRawBufferAllocator::ResetPeak,
           "Resets the memory peak to the current :attr:`total_allocated_size`.");

  // RuntimeContext — name-keyed tensor map + kernel context + function registry.
  nb::class_<RuntimeContext>(
      rt_mod, "RuntimeContext",
      "Per-invocation runtime state passed to :func:`RunNode` / "
      ":class:`RuntimeSession`. Owns the name-keyed "
      "tensor map carrying graph inputs/initializers and every intermediate value "
      "produced by previously executed nodes.")
      .def(nb::init<>())
      .def(
          "__init__",
          [](RuntimeContext *self, KernelContext kernel_ctx, bool events_enabled, int verbose,
             bool release_intermediates, SimpleRawBufferAllocator *allocator) {
            new (self) RuntimeContext(std::move(kernel_ctx),
                                      core::runtime::RuntimeContextOptions{
                                          .allocator = allocator,
                                          .events_enabled = events_enabled,
                                          .verbose = verbose,
                                          .release_intermediates = release_intermediates,
                                      });
          },
          nb::arg("kernel_ctx"), nb::kw_only(), nb::arg("events_enabled") = false,
          nb::arg("verbose") = 0, nb::arg("release_intermediates") = false,
          nb::arg("allocator").none() = nullptr, nb::keep_alive<1, 6>())
      .def_prop_ro(
          "events_enabled", [](const RuntimeContext &rt) { return rt.events_enabled(); },
          "When ``True``, :func:`set` / :func:`put` / :func:`remove` and "
          ":func:`run_node` record events (incl. clock reads and value decoding). "
          "Default is ``False`` for maximum throughput; enable only when profiling "
          "is required.")
      .def_prop_ro(
          "verbose", [](const RuntimeContext &rt) { return rt.verbose(); },
          "Verbosity level used by :func:`run_node` to print execution progress to "
          "``stdout`` while the graph runs. ``0`` disables printing.")
      .def_prop_rw(
          "release_intermediates",
          [](const RuntimeContext &rt) { return rt.release_intermediates(); },
          [](RuntimeContext &rt, bool v) { rt.set_release_intermediates(v); },
          "When ``True``, :class:`RuntimeSession` (used to run any node list, including "
          "a whole model's graph) remove an intermediate tensor (or sequence) from the runtime "
          "context as soon as the last node that references it has finished — emitting a "
          "``kRemove`` event when :attr:`events_enabled` is ``True``. Graph / function "
          "outputs and names already present in the context before the run are always "
          "preserved. Default is ``False`` so that intermediate values stay observable "
          "after the run.")
      .def_prop_ro(
          "kernel_ctx", [](RuntimeContext &rt) -> KernelContext & { return rt.kernel_ctx(); },
          nb::rv_policy::reference_internal, "Kernel construction context (opset).")
      .def("has", &RuntimeContext::Has, nb::arg("name"),
           "Returns ``True`` if a tensor named ``name`` is currently held.")
      .def("remove", &RuntimeContext::Remove, nb::arg("name"),
           "Removes the tensor stored under ``name`` if present. Returns ``True`` if "
           "an entry was erased.")
      .def(
          "set",
          [](RuntimeContext &rt, const std::string &name, Tensor tensor, const std::string &kind) {
            rt.Set(name, std::move(tensor), ParseRuntimeEventKind(kind));
          },
          nb::arg("name"), nb::arg("tensor"), nb::arg("kind") = "input",
          "Inserts ``tensor`` under ``name``. Raises if ``name`` already exists. "
          "Records an ``add`` event in :func:`events` with the supplied ``kind`` "
          "(default ``\"input\"``).")
      .def(
          "put",
          [](RuntimeContext &rt, const std::string &name, Tensor tensor, const std::string &kind) {
            rt.Put(name, std::move(tensor), ParseRuntimeEventKind(kind));
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
           "Returns ``True`` if a sequence named ``name`` is currently held by the context.")
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
            const Tensors &values = rt.GetSequence(name).values;
            return std::vector<Tensor>(values.begin(), values.end());
          },
          nb::arg("name"),
          "Returns the tensors in the sequence stored under ``name`` as a list of "
          ":class:`Tensor` objects. Raises ``std::out_of_range`` if absent.")
      .def(
          "put_sequence",
          [](RuntimeContext &rt, const std::string &name, std::vector<Tensor> values) {
            // The element type is shared by every tensor of a sequence; infer it
            // from the first element and fall back to ``UNDEFINED`` (``0``) for an
            // empty sequence.
            int32_t elem_type = values.empty() ? 0 : static_cast<int32_t>(values.front().data_type);
            rt.PutSequence(name, Sequence(name, elem_type, std::move(values)));
          },
          nb::arg("name"), nb::arg("values"),
          "Inserts or overwrites the sequence stored under ``name`` from a list of "
          ":class:`Tensor` objects. The sequence element type is inferred from the "
          "first tensor (``UNDEFINED`` when the list is empty).")
      .def("has_map", &RuntimeContext::HasMap, nb::arg("name"),
           "Returns ``True`` if a map named ``name`` is currently held by the context.")
      .def(
          "map_names",
          [](const RuntimeContext &rt) {
            std::vector<std::string> out;
            out.reserve(rt.maps().size());
            for (const auto &kv : rt.maps()) {
              out.push_back(kv.first);
            }
            return out;
          },
          "Returns the list of map names currently held by the context.")
      .def(
          "put_map",
          [](RuntimeContext &rt, const std::string &name, nb::dict d) {
            int64_t n = static_cast<int64_t>(d.size());
            if (n == 0) {
              // Empty dict: store an empty map with default-constructed tensors.
              rt.PutMap(name, Map(name, Tensor{}, Tensor{}));
              return;
            }
            auto it = d.begin();
            nb::handle first_key = (*it).first;
            nb::handle first_val = (*it).second;

            // Detect key and value scalar types from the first entry.
            // In Python, bool is a subclass of int, so exclude it explicitly.
            bool str_keys = nb::isinstance<nb::str>(first_key);
            bool int_keys = nb::isinstance<nb::int_>(first_key) &&
                            !nb::isinstance<nb::bool_>(first_key) &&
                            !nb::isinstance<nb::float_>(first_key);
            bool str_vals = nb::isinstance<nb::str>(first_val);
            bool float_vals = nb::isinstance<nb::float_>(first_val);
            bool int_vals = nb::isinstance<nb::int_>(first_val) &&
                            !nb::isinstance<nb::bool_>(first_val) &&
                            !nb::isinstance<nb::float_>(first_val);

            if (!str_keys && !int_keys) {
              throw std::invalid_argument(
                  "put_map: unsupported key type in dict - keys must be int or str scalars.");
            }
            if (!str_vals && !float_vals && !int_vals) {
              throw std::invalid_argument(
                  "put_map: unsupported value type in dict - values must be "
                  "int, float, or str scalars (all entries must share the same type).");
            }

            // Allocate storage and iterate once to populate both keys and values.
            std::vector<std::string> str_key_buf, str_val_buf;
            std::vector<int64_t> int_key_buf, int_val_buf;
            std::vector<float> float_val_buf;

            if (str_keys)
              str_key_buf.reserve(static_cast<size_t>(n));
            else
              int_key_buf.reserve(static_cast<size_t>(n));
            if (str_vals)
              str_val_buf.reserve(static_cast<size_t>(n));
            else if (float_vals)
              float_val_buf.reserve(static_cast<size_t>(n));
            else
              int_val_buf.reserve(static_cast<size_t>(n));

            for (auto [k, v] : d) {
              if (str_keys)
                str_key_buf.push_back(nb::cast<std::string>(k));
              else
                int_key_buf.push_back(nb::cast<int64_t>(k));
              if (str_vals)
                str_val_buf.push_back(nb::cast<std::string>(v));
              else if (float_vals)
                float_val_buf.push_back(nb::cast<float>(v));
              else
                int_val_buf.push_back(nb::cast<int64_t>(v));
            }

            Tensor keys_t = str_keys ? Tensor::FromStrings(name, {n}, str_key_buf)
                                     : Tensor::FromInt64(name, {n}, int_key_buf);
            Tensor vals_t = str_vals     ? Tensor::FromStrings(name, {n}, str_val_buf)
                            : float_vals ? Tensor::FromFloat(name, {n}, float_val_buf)
                                         : Tensor::FromInt64(name, {n}, int_val_buf);

            rt.PutMap(name, Map(name, std::move(keys_t), std::move(vals_t)));
          },
          nb::arg("name"), nb::arg("d"),
          "Inserts or overwrites the map stored under ``name`` from a Python ``dict``. "
          "Keys must be ``int`` or ``str`` scalars; values must be ``int``, ``float``, or "
          "``str`` scalars (all entries must share the same key type and value type). "
          "An empty dict stores a map with empty tensors. "
          "The map key/value types are inferred from the first dict entry.")
      .def(
          "events",
          [](const RuntimeContext &rt) {
            // Returns a copy of the append-only log as a list of
            // :class:`RuntimeEvent` instances. Use :meth:`RuntimeEvent.as_dict`
            // to materialise an individual entry as a plain ``dict`` (e.g.
            // for serialisation or tabular rendering).
            return rt.events();
          },
          "Returns the append-only log of tensor map mutations and node dispatches "
          "as a list of :class:`RuntimeEvent` instances. Each entry carries "
          "``action`` (:class:`RuntimeEventAction` enum), ``kind`` "
          "(``\"unknown\"`` / ``\"initializer\"`` / ``\"input\"`` / "
          "``\"intermediate\"`` / ``\"output\"``), ``timestamp_ns`` "
          "(``int`` nanoseconds since the Unix epoch), ``name``, ``data_type`` "
          "(``TensorProto.DataType`` integer), ``shape``, ``value_count``, and "
          "the first ``value_count`` element values from the tensor (``values`` "
          "for numeric dtypes, ``string_values`` for ``STRING``). The value "
          "buffer is fixed-size (capped at 8 entries); for tensors with more "
          "than 8 elements only the first 8 are kept, ``data_type`` is set to "
          "``-1`` and ``shape`` is empty to signal the truncated payload. Call "
          ":meth:`RuntimeEvent.as_dict` to convert an individual entry to a "
          "plain Python ``dict``.")
      .def("clear_events", &RuntimeContext::ClearEvents,
           "Empties the event log without otherwise touching the tensor map.")
      .def("clear", &RuntimeContext::Clear,
           "Resets the per-invocation state so the context can be reused for a fresh "
           "run: clears the tensor map, the sequence map and the event log, and resets "
           "the current node index. The kernel context, registered model-local "
           "functions and custom kernels, the cached execution plans and the "
           ":attr:`events_enabled` / :attr:`release_intermediates` settings are "
           "preserved, so the execution-plan cache is amortised across repeated runs "
           "of the same model.")
      .def(
          "register_custom_kernel",
          [](RuntimeContext &rt, const std::string &domain, const std::string &op_type,
             nb::callable fn) {
            // Wrap the Python callable in a CustomKernelFn. We capture
            // the callable in a ``nb::callable`` which keeps a Python
            // reference alive until the registration is replaced or the
            // RuntimeContext is destroyed. The GIL is reacquired before
            // invoking the callable so that it is safe to call from the
            // RunNode dispatcher (which may be invoked without the GIL
            // held in the future).
            rt.RegisterCustomKernel(domain, op_type,
                                    [fn](const NodeProto &node, RuntimeContext &ctx) {
                                      nb::gil_scoped_acquire gil;
                                      fn(nb::cast(&node, nb::rv_policy::reference),
                                         nb::cast(&ctx, nb::rv_policy::reference));
                                    });
          },
          nb::arg("domain"), nb::arg("op_type"), nb::arg("fn"),
          "Registers a custom kernel for ``(domain, op_type)``. The empty domain "
          "is normalised to ``ai.onnx``. ``fn`` is a Python callable invoked as "
          "``fn(node, ctx)`` where ``node`` is the :class:`NodeProto` being "
          "dispatched and ``ctx`` is this :class:`RuntimeContext`. The callable "
          "must read its inputs from ``ctx`` (via :meth:`get` / :meth:`get_sequence`) "
          "and write its outputs back into ``ctx`` (via :meth:`put` / "
          ":meth:`put_sequence`) under the names declared by ``node.output``. "
          "Custom kernels override any built-in entry with the same key, but "
          "model-local functions and the built-in control-flow operators "
          "(``If``, ``Loop``, ``Scan``, ``SequenceMap``) still take precedence.")
      .def("unregister_custom_kernel", &RuntimeContext::UnregisterCustomKernel, nb::arg("domain"),
           nb::arg("op_type"),
           "Removes a custom kernel registration for ``(domain, op_type)``. "
           "The empty domain is normalised to ``ai.onnx``. Returns ``True`` "
           "when an entry was removed.")
      .def("clear_custom_kernels", &RuntimeContext::ClearCustomKernels,
           "Removes every custom kernel registration from the runtime context.")
      .def(
          "get_execution_plan",
          [](RuntimeContext &rt, const GraphProto &graph) -> const ExecutionPlan & {
            return rt.GetExecutionPlan(graph);
          },
          nb::arg("graph"), nb::rv_policy::reference_internal, nb::keep_alive<1, 2>(),
          "Returns the cached :class:`ExecutionPlan` for ``graph``, building it on "
          "first use. The plan is keyed by the identity of ``graph`` and reused "
          "across subsequent runs of the same model, so the analysis is paid only "
          "once for the lifetime of this :class:`RuntimeContext`. This binding keeps "
          "``graph`` alive for at least as long as ``self``, since the cached plan "
          "holds non-owning pointers into it beyond the lifetime of the returned "
          "wrapper.")
      .def(
          "get_execution_plan",
          [](RuntimeContext &rt, const FunctionProto &func) -> const ExecutionPlan & {
            return rt.GetExecutionPlan(func);
          },
          nb::arg("func"), nb::rv_policy::reference_internal, nb::keep_alive<1, 2>(),
          "Returns the cached :class:`ExecutionPlan` for ``func``, building it on "
          "first use. Same caching semantics as the ``GraphProto`` overload — the "
          "structural keep set consists of the function's declared inputs and "
          "outputs. This binding keeps ``func`` alive for at least as long as "
          "``self``, for the same reason as the ``graph`` overload.")
      .def("clear_execution_plans", &RuntimeContext::ClearExecutionPlans,
           "Clears every cached :class:`ExecutionPlan`. Useful when the owning model "
           "has been mutated in place (rare).")
      .def_static(
          "collect_external_inputs",
          [](nb::handle nodes) {
            return WithNodeList(nodes, [](const utils::RepeatedProtoField<NodeProto> &typed_nodes) {
              return core::runtime::RuntimeSession::CollectExternalInputs(typed_nodes);
            });
          },
          nb::arg("nodes"),
          "Returns the list of input names referenced by ``nodes`` that are not "
          "produced as outputs by any node in the same list. Names captured by "
          "subgraph attributes (``GRAPH`` / ``GRAPHS``) are inspected recursively: "
          "for every subgraph, names read by its nodes that are neither produced "
          "inside the subgraph (formal inputs, initializers, intermediate node "
          "outputs) nor produced by the outer ``nodes`` are appended. The returned "
          "list preserves first-seen order and contains no duplicates; empty input "
          "names (optional inputs left unbound) are skipped.");

  // Top-level run helper. Only single-node dispatch (``run_node``) is
  // exposed as a standalone helper; running any node list (a bare graph, a
  // function body, or a whole model's graph) is done by building an
  // :class:`ExecutionPlan` (via :func:`RuntimeContext.get_execution_plan`)
  // and driving it through a :class:`RuntimeSession`. For a ``ModelProto``,
  // call :func:`register_model_functions` first so nodes referring to
  // model-local functions resolve correctly.
  rt_mod.def(
      "run_node",
      [](const NodeProto &node, RuntimeContext &rt) { core::runtime::RunNode(node, rt); },
      nb::arg("node"), nb::arg("rt"),
      "Runs the kernel registered for ``node`` and stores its outputs in ``rt``. "
      "``rt`` must already contain entries for every input referenced by ``node``; "
      "on return it also contains entries for every output declared by ``node``.");

  rt_mod.def(
      "register_model_functions",
      [](const ModelProto &model, RuntimeContext &rt) {
        core::runtime::RegisterModelFunctions(model, rt);
      },
      nb::arg("model"), nb::arg("rt"), nb::keep_alive<2, 1>(),
      "Registers every ``FunctionProto`` in ``model.functions`` in ``rt``'s "
      "function registry so nodes referring to model-local functions by "
      "``(domain, op_type, overload)`` are dispatched transparently. Call this "
      "once before building ``model.graph``'s :class:`ExecutionPlan` and driving "
      "it through a :class:`RuntimeSession`. Keeps ``model`` alive for at least "
      "as long as ``rt``, since ``rt`` stores non-owning pointers into it.");

  rt_mod.def(
      "tensor_from_proto",
      [](const TensorProto &tp) {
        // ``TensorFromProto`` returns a borrowed (zero-copy) view into
        // ``tp.raw_data()`` when raw data is used. That view becomes
        // dangling as soon as ``tp`` is garbage-collected on the Python
        // side, so we materialize an owned ``Tensor`` here.
        Tensor t = core::runtime::TensorFromProto(tp);
        if (t.data.empty() && t.size_bytes() > 0) {
          core::runtime::RawByteBuffer owned(t.bytes(), t.bytes() + t.size_bytes());
          return Tensor::FromRawBytes(t.name, t.data_type, t.shape, std::move(owned));
        }
        return t;
      },
      nb::arg("tp"),
      "Converts a ``TensorProto`` to a :class:`Tensor`. Supports all numeric data "
      "types stored either in the typed repeated fields or in the raw little-endian "
      "``raw_data`` field; ``STRING`` tensors are read from ``string_data``. The "
      "returned tensor owns its bytes, so it remains valid after ``tp`` is "
      "garbage-collected.");

  rt_mod.def(
      "tensor_to_proto",
      [](const Tensor &t) {
        // Materializes a ``TensorProto`` from a runtime ``Tensor``. For the
        // raw-data path the proto's ``raw_data`` borrows the tensor's byte
        // buffer (zero-copy) instead of copying it; ``nb::keep_alive<0, 1>``
        // ties the source tensor's lifetime to the returned proto so the
        // borrowed view never dangles.
        TensorProto tp;
        if (!t.name.empty())
          tp.set_name(t.name);
        tp.set_data_type(t.data_type);
        tp.ref_dims().reserve(t.shape.size());
        for (int64_t d : t.shape)
          tp.ref_dims().push_back(static_cast<uint64_t>(d));
        if (static_cast<TensorProto::DataType>(t.data_type) == TensorProto::DataType::STRING) {
          tp.ref_string_data().reserve(t.string_data.size());
          for (const std::string &s : t.string_data)
            tp.add_string_data(utils::String(s));
        } else {
          // ``assign_borrowed`` stores a non-owning view over the tensor's
          // bytes; ``nb::keep_alive<0, 1>`` (below) keeps the source tensor
          // alive for the proto's lifetime so the view never dangles.
          tp.ref_raw_data().assign_borrowed(t.bytes(), t.size_bytes());
        }
        return tp;
      },
      nb::keep_alive<0, 1>(), nb::arg("t"),
      "Converts a runtime :class:`Tensor` to a ``TensorProto``. For non-``STRING`` "
      "tensors the proto's ``raw_data`` borrows the tensor's byte buffer (zero-copy); "
      "the source tensor is kept alive for the lifetime of the returned proto so the "
      "borrowed view never dangles. ``STRING`` tensors are written to ``string_data``.");

  rt_mod.def(
      "tensor_to_numpy",
      [](nb::handle t_obj, bool steal) {
        // Builds a 1-D ``uint8`` NumPy array over the tensor's raw byte buffer.
        //
        // By default (``steal=False``) the array borrows the bytes (no copy)
        // and keeps the source tensor (``t_obj``) alive for as long as the view
        // lives, so the borrowed span never dangles.
        //
        // When ``steal=True`` *and* the tensor owns its bytes inline (an
        // canonical raw-byte buffer that is neither allocator-backed nor a
        // borrowed span) the buffer's ownership is transferred to NumPy: the
        // byte vector is moved into a capsule that frees it when the array (and
        // any array derived from it) is garbage-collected. This DLPack-style
        // hand-off lets a runtime graph producing owned outputs be released
        // while the arrays live on, at the cost of emptying the source tensor.
        // Stealing is skipped for allocator-backed tensors (their bytes belong
        // to the allocator pool and must be returned to it) and for borrowed
        // views (they reference external memory), which fall back to borrowing.
        Tensor &t = nb::cast<Tensor &>(t_obj);
        EXT_ENFORCE_INVALID(
            !(static_cast<TensorProto::DataType>(t.data_type) == TensorProto::DataType::STRING),
            "tensor_to_numpy: STRING tensors have no raw byte buffer; use "
            "tensor_to_proto instead.");
        const size_t n = t.size_bytes();
        // Inline-owned when there is no allocator backing and ``bytes()``
        // resolves to the inline ``data`` buffer (borrowed tensors keep
        // ``data`` empty and read from an external span instead).
        if (steal && !t.has_allocation() && n > 0 && t.bytes() == t.data.data()) {
          auto *owned = new core::runtime::RawByteBuffer(t.data.release());
          nb::capsule owner(owned, [](void *p) noexcept {
            delete static_cast<core::runtime::RawByteBuffer *>(p);
          });
          return nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>>(owned->data(), {n}, owner);
        }
        return nb::ndarray<nb::numpy, const uint8_t, nb::ndim<1>>(t.bytes(), {n}, t_obj);
      },
      nb::arg("t"), nb::arg("steal") = false,
      "Returns a 1-D ``uint8`` NumPy array over the runtime tensor's raw "
      "little-endian byte buffer. By default the array is a zero-copy view whose "
      "source tensor is kept alive for the lifetime of the returned array (it is "
      "the array's ``base``) so the borrowed view never dangles. When ``steal`` is "
      "``True`` and the tensor owns its bytes inline (not allocator-backed and not "
      "a borrowed view), the buffer's ownership is transferred to NumPy "
      "(DLPack-style) — the array owns the bytes through a capsule, the source "
      "tensor is emptied, and it is no longer kept alive. Callers reinterpret the "
      "bytes via ``ndarray.view(dtype).reshape(shape)``. ``STRING`` tensors have no "
      "raw buffer and must go through :func:`tensor_to_proto`.");

  rt_mod.def(
      "tensor_from_numpy",
      [](const std::string &name, int32_t data_type, std::vector<int64_t> shape,
         nb::ndarray<const uint8_t, nb::ndim<1>, nb::c_contig> raw, bool copy) {
        // Converts a contiguous NumPy uint8 buffer into a runtime tensor.
        // ``copy=True`` preserves the existing behaviour: bytes are copied
        // into owned tensor storage. ``copy=False`` builds a borrowed tensor
        // view over the NumPy memory (zero-copy); ``keep_alive<0, 4>`` below
        // guarantees the source array outlives the returned tensor.
        const uint8_t *ptr = raw.data();
        size_t n = raw.shape(0);
        if (!copy) {
          return Tensor::Borrow(std::string(name), data_type, std::move(shape), ptr, n);
        }
        core::runtime::RawByteBuffer owned(ptr, ptr + n);
        return Tensor::FromRawBytes(std::string(name), data_type, std::move(shape),
                                    std::move(owned));
      },
      nb::keep_alive<0, 4>(), nb::arg("name"), nb::arg("data_type"), nb::arg("shape"),
      nb::arg("raw"), nb::arg("copy") = true,
      "Constructs a :class:`Tensor` from raw element bytes in a contiguous 1-D "
      "``uint8`` NumPy array with an explicit ONNX data type. When ``copy=True`` "
      "(default), bytes are copied into owned tensor storage. When ``copy=False``, "
      "the tensor borrows the NumPy buffer (zero-copy). Use this overload for "
      "bfloat16, float8, and sub-byte packed types.");

  // --- global (process-wide) custom kernels -------------------------------
  // Module-level counterparts of RuntimeContext.register_custom_kernel that
  // install a kernel once for *every* RuntimeContext instead of a single one.
  rt_mod.def(
      "register_custom_kernel",
      [](const std::string &domain, const std::string &op_type, nb::callable fn) {
        // Same GIL-safe wrapping as the RuntimeContext binding: the callable is
        // captured in an nb::callable (keeping a Python reference alive until the
        // registration is replaced or cleared) and the GIL is reacquired before
        // it is invoked from the RunNode dispatcher.
        core::runtime::RegisterGlobalCustomKernel(domain, op_type,
                                                  [fn](const NodeProto &node, RuntimeContext &ctx) {
                                                    nb::gil_scoped_acquire gil;
                                                    fn(nb::cast(&node, nb::rv_policy::reference),
                                                       nb::cast(&ctx, nb::rv_policy::reference));
                                                  });
      },
      nb::arg("domain"), nb::arg("op_type"), nb::arg("fn"),
      "Registers a process-wide (global) custom kernel for ``(domain, op_type)``. "
      "Unlike :meth:`RuntimeContext.register_custom_kernel`, which affects a single "
      "context, a global kernel is picked up by every :class:`RuntimeContext`. The "
      "empty domain is normalised to ``ai.onnx``. ``fn`` is a Python callable "
      "invoked as ``fn(node, ctx)`` (see "
      ":meth:`RuntimeContext.register_custom_kernel`). A per-context registration "
      "for the same key overrides the global one; both override any built-in entry, "
      "but model-local functions and the built-in control-flow operators (``If``, "
      "``Loop``, ``Scan``, ``SequenceMap``) still take precedence.");
  rt_mod.def("unregister_custom_kernel", &core::runtime::UnregisterGlobalCustomKernel,
             nb::arg("domain"), nb::arg("op_type"),
             "Removes a process-wide custom kernel registration for "
             "``(domain, op_type)``. The empty domain is normalised to ``ai.onnx``. "
             "Returns ``True`` when an entry was removed.");
  rt_mod.def("clear_custom_kernels", &core::runtime::ClearGlobalCustomKernels,
             "Removes every process-wide custom kernel registration.");
}

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "../onnx_proto/_onnxpy.h"
#include "nnef/exporter.h"
#include "nnef/tensor_io.h"
#include "onnx.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using nnef::AttributeValue;
using nnef::ExportContext;
using nnef::NNEFExportError;
using nnef::NNEFGraph;
using nnef::NNEFTensor;

namespace {

// Map a numpy dtype description to NNEF (item_type, bits).
bool DtypeFromNumpy(nb::dlpack::dtype dt, int &item_type, int &bits) {
  bits = dt.bits;
  if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Float)) {
    if (bits == 16 || bits == 32 || bits == 64) {
      item_type = nnef::kItemTypeFloat;
      return true;
    }
  } else if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Int)) {
    if (bits == 8 || bits == 16 || bits == 32 || bits == 64) {
      item_type = nnef::kItemTypeSigned;
      return true;
    }
  } else if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::UInt)) {
    if (bits == 8 || bits == 16 || bits == 32 || bits == 64) {
      item_type = nnef::kItemTypeUnsigned;
      return true;
    }
  } else if (dt.code == static_cast<uint8_t>(nb::dlpack::dtype_code::Bool)) {
    item_type = nnef::kItemTypeBool;
    bits = 8;
    return true;
  }
  return false;
}

NNEFTensor NumpyToNNEFTensor(nb::ndarray<nb::ro, nb::c_contig> arr) {
  NNEFTensor t;
  int item_type, bits;
  if (!DtypeFromNumpy(arr.dtype(), item_type, bits)) {
    throw std::invalid_argument("Unsupported numpy dtype for NNEF");
  }
  t.item_type = item_type;
  t.bits = bits;
  for (size_t i = 0; i < arr.ndim(); ++i) {
    t.shape.push_back(static_cast<int64_t>(arr.shape(i)));
  }
  size_t nbytes = 1;
  for (size_t i = 0; i < arr.ndim(); ++i)
    nbytes *= arr.shape(i);
  if (arr.ndim() == 0)
    nbytes = 1;
  if (item_type == nnef::kItemTypeBool) {
    // Each element is a single byte (numpy bool is uint8 in memory)
    t.data.resize(nbytes);
    const uint8_t *src = reinterpret_cast<const uint8_t *>(arr.data());
    for (size_t i = 0; i < nbytes; ++i) {
      t.data[i] = src[i] ? 1u : 0u;
    }
  } else {
    size_t byte_size = nbytes * static_cast<size_t>(bits / 8);
    t.data.assign(reinterpret_cast<const uint8_t *>(arr.data()),
                  reinterpret_cast<const uint8_t *>(arr.data()) + byte_size);
  }
  return t;
}

// Convert NNEFTensor to numpy ndarray (owning copy).
nb::object NNEFTensorToNumpy(const NNEFTensor &t) {
  nb::dlpack::dtype dt;
  dt.lanes = 1;
  dt.bits = static_cast<uint8_t>(t.bits);
  switch (t.item_type) {
  case nnef::kItemTypeFloat:
    dt.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Float);
    break;
  case nnef::kItemTypeSigned:
    dt.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Int);
    break;
  case nnef::kItemTypeUnsigned:
    dt.code = static_cast<uint8_t>(nb::dlpack::dtype_code::UInt);
    break;
  case nnef::kItemTypeBool:
    dt.code = static_cast<uint8_t>(nb::dlpack::dtype_code::Bool);
    dt.bits = 8;
    break;
  default:
    throw std::runtime_error("Unsupported NNEF item type");
  }
  // Allocate owning buffer via a shared_ptr-like deleter mechanism.
  auto *buf = new std::vector<uint8_t>(t.data);
  nb::capsule owner(buf, [](void *p) noexcept { delete static_cast<std::vector<uint8_t> *>(p); });

  std::vector<size_t> shape;
  for (auto d : t.shape)
    shape.push_back(static_cast<size_t>(d));

  nb::ndarray<nb::numpy> arr(buf->data(), shape.size(), shape.empty() ? nullptr : shape.data(),
                             owner,
                             /*strides*/ nullptr, dt);
  return nb::cast(arr, nb::rv_policy::move);
}

// Convert a Python value (from a registered Python converter's attrs dict) into AttributeValue.
// Not needed for the test, but provide it for completeness.
AttributeValue PyToAttributeValue(nb::handle h) {
  if (nb::isinstance<nb::bool_>(h))
    return static_cast<int64_t>(nb::cast<bool>(h) ? 1 : 0);
  if (nb::isinstance<nb::int_>(h))
    return nb::cast<int64_t>(h);
  if (nb::isinstance<nb::float_>(h))
    return nb::cast<double>(h);
  if (nb::isinstance<nb::str>(h))
    return nb::cast<std::string>(h);
  if (nb::isinstance<nb::list>(h) || nb::isinstance<nb::tuple>(h)) {
    nb::sequence seq = nb::cast<nb::sequence>(h);
    // Inspect first element.
    bool all_int = true, all_float = true, all_str = true, any = false;
    for (auto item : seq) {
      any = true;
      if (!nb::isinstance<nb::int_>(item))
        all_int = false;
      if (!nb::isinstance<nb::float_>(item) && !nb::isinstance<nb::int_>(item))
        all_float = false;
      if (!nb::isinstance<nb::str>(item))
        all_str = false;
    }
    if (!any)
      return std::vector<int64_t>{};
    if (all_int) {
      std::vector<int64_t> v;
      for (auto item : seq)
        v.push_back(nb::cast<int64_t>(item));
      return v;
    }
    if (all_str) {
      std::vector<std::string> v;
      for (auto item : seq)
        v.push_back(nb::cast<std::string>(item));
      return v;
    }
    if (all_float) {
      std::vector<double> v;
      for (auto item : seq)
        v.push_back(nb::cast<double>(item));
      return v;
    }
  }
  throw std::runtime_error("Cannot convert Python value to NNEF AttributeValue");
}

nb::object AttributeValueToPy(const AttributeValue &v) {
  return std::visit(
      [](auto &&x) -> nb::object {
        using T = std::decay_t<decltype(x)>;
        if constexpr (std::is_same_v<T, std::monostate>) {
          return nb::none();
        } else if constexpr (std::is_same_v<T, int64_t>) {
          return nb::cast(x);
        } else if constexpr (std::is_same_v<T, double>) {
          return nb::cast(x);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return nb::cast(x);
        } else if constexpr (std::is_same_v<T, std::vector<int64_t>>) {
          return nb::cast(x);
        } else if constexpr (std::is_same_v<T, std::vector<double>>) {
          return nb::cast(x);
        } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
          return nb::cast(x);
        } else if constexpr (std::is_same_v<T, NNEFTensor>) {
          return NNEFTensorToNumpy(x);
        } else {
          return nb::none();
        }
      },
      v);
}

// Parse a Python-provided model. Accepts onnx_light.ModelProto by reference,
// or any object implementing SerializeToString() (e.g. upstream onnx.ModelProto),
// or raw bytes.
std::unique_ptr<ModelProto> ParseModelArg(nb::object model_obj, const ModelProto *&out_ref) {
  out_ref = nullptr;
  // Direct cast (onnx_light ModelProto).
  if (nb::try_cast<ModelProto *>(model_obj, const_cast<ModelProto *&>(out_ref))) {
    return nullptr;
  }
  ModelProto *mp_ptr = nullptr;
  if (nb::try_cast<ModelProto *>(model_obj, mp_ptr)) {
    out_ref = mp_ptr;
    return nullptr;
  }
  // Bytes.
  std::string serialized;
  if (nb::isinstance<nb::bytes>(model_obj)) {
    nb::bytes b = nb::cast<nb::bytes>(model_obj);
    serialized.assign(reinterpret_cast<const char *>(b.data()), b.size());
  } else if (nb::hasattr(model_obj, "SerializeToString")) {
    nb::object s = model_obj.attr("SerializeToString")();
    if (nb::isinstance<nb::bytes>(s)) {
      nb::bytes b = nb::cast<nb::bytes>(s);
      serialized.assign(reinterpret_cast<const char *>(b.data()), b.size());
    } else {
      serialized = nb::cast<std::string>(s);
    }
  } else {
    throw std::runtime_error("export_to_nnef: expected ModelProto, bytes, or an object with "
                             "SerializeToString()");
  }
  auto owned = std::make_unique<ModelProto>();
  owned->ParseFromString(serialized);
  out_ref = owned.get();
  return owned;
}

} // namespace

void AddOnnxPyNnef(nb::module_ &m) {
  // Ensure NodeProto / ModelProto class objects are registered so we can pass
  // them to Python-defined converter callbacks.
  try {
    nb::module_::import_("onnx_light.onnx_py._onnxpyprotoop");
  } catch (...) {
    // Best-effort; bindings may already be loaded by the test runner.
  }

  auto nnef_mod = m.def_submodule("nnef");
  nnef_mod.doc() = "ONNX -> NNEF exporter (C++ implementation).";

  // Exception.
  nb::exception<NNEFExportError>(nnef_mod, "NNEFExportError");

  // ----- ExportContext --------------------------------------------------
  nb::class_<ExportContext>(nnef_mod, "ExportContext",
                            "Mutable state threaded through op converter invocations.")
      .def(
          "add_statement", [](ExportContext &self, const std::string &s) { self.AddStatement(s); },
          nb::arg("stmt"))
      .def(
          "make_temp",
          [](ExportContext &self, const std::string &base) { return self.MakeTemp(base); },
          nb::arg("base") = "t")
      .def(
          "map_name",
          [](ExportContext &self, const std::string &name) { return self.MapName(name); },
          nb::arg("onnx_name"))
      .def(
          "get_initializer",
          [](ExportContext &self, const std::string &name) -> nb::object {
            const NNEFTensor *t = self.GetInitializer(name);
            if (t == nullptr)
              return nb::none();
            return NNEFTensorToNumpy(*t);
          },
          nb::arg("onnx_name"));

  // ----- NNEFGraph ------------------------------------------------------
  nb::class_<NNEFGraph>(nnef_mod, "NNEFGraph", "In-memory representation of a NNEF graph.")
      .def_prop_ro("name", [](const NNEFGraph &g) { return g.name; })
      .def_prop_ro("inputs", [](const NNEFGraph &g) { return g.inputs; })
      .def_prop_ro("outputs", [](const NNEFGraph &g) { return g.outputs; })
      .def_prop_ro("statements", [](const NNEFGraph &g) { return g.statements; })
      .def_prop_ro(
          "version",
          [](const NNEFGraph &g) { return nb::make_tuple(g.version_major, g.version_minor); })
      .def_prop_ro("extensions", [](const NNEFGraph &g) { return g.extensions; })
      .def_prop_ro("initializers",
                   [](const NNEFGraph &g) {
                     nb::dict d;
                     for (const auto &p : g.initializers) {
                       d[nb::cast(p.first)] = NNEFTensorToNumpy(p.second);
                     }
                     return d;
                   })
      .def("to_text", [](const NNEFGraph &g) { return g.ToText(); });

  // ----- export_to_nnef / to_nnef_text / save_nnef ---------------------
  nnef_mod.def(
      "export_to_nnef",
      [](nb::object model_obj, nb::object graph_name) {
        const ModelProto *mp = nullptr;
        auto owned = ParseModelArg(model_obj, mp);
        std::string gn;
        if (!graph_name.is_none())
          gn = nb::cast<std::string>(graph_name);
        return nnef::ExportToNNEF(*mp, gn);
      },
      nb::arg("model"), nb::arg("graph_name") = nb::none());

  nnef_mod.def(
      "to_nnef_text",
      [](nb::object model_obj, nb::object graph_name) {
        const ModelProto *mp = nullptr;
        auto owned = ParseModelArg(model_obj, mp);
        std::string gn;
        if (!graph_name.is_none())
          gn = nb::cast<std::string>(graph_name);
        return nnef::ToNNEFText(*mp, gn);
      },
      nb::arg("model"), nb::arg("graph_name") = nb::none());

  nnef_mod.def(
      "save_nnef",
      [](nb::object model_obj, const std::string &out_dir, nb::object graph_name, bool overwrite) {
        const ModelProto *mp = nullptr;
        auto owned = ParseModelArg(model_obj, mp);
        std::string gn;
        if (!graph_name.is_none())
          gn = nb::cast<std::string>(graph_name);
        try {
          return nnef::SaveNNEF(*mp, out_dir, gn, overwrite);
        } catch (const std::runtime_error &e) {
          std::string msg = e.what();
          if (msg.rfind("FileExistsError:", 0) == 0) {
            PyErr_SetString(PyExc_FileExistsError, msg.c_str() + 17);
            throw nb::python_error();
          }
          throw;
        }
      },
      nb::arg("model"), nb::arg("out_dir"), nb::arg("graph_name") = nb::none(),
      nb::arg("overwrite") = true);

  nnef_mod.def("supported_ops", []() { return nnef::SupportedOps(); });

  // ----- register_op_converter -----------------------------------------
  nnef_mod.def(
      "register_op_converter",
      [](const std::string &op_type, nb::object callable) {
        nb::object holder = callable; // keep reference
        nnef::RegisterOpConverter(
            op_type, [holder](ExportContext &ctx, const NodeProto &node,
                              const std::map<std::string, AttributeValue> &attrs,
                              const std::vector<std::string> &inputs,
                              const std::vector<std::string> &outputs) {
              nb::gil_scoped_acquire gil;
              nb::dict pyattrs;
              for (const auto &kv : attrs) {
                pyattrs[nb::cast(kv.first)] = AttributeValueToPy(kv.second);
              }
              // Pass the NodeProto by reference if registered; otherwise as None.
              nb::object py_node;
              try {
                py_node = nb::cast(node, nb::rv_policy::reference);
              } catch (...) {
                py_node = nb::none();
              }
              holder(nb::cast(&ctx, nb::rv_policy::reference), py_node, pyattrs, inputs, outputs);
            });
      },
      nb::arg("op_type"), nb::arg("converter"));

  // ----- _ConverterRegistryView (dict-like proxy for ``_CONVERTERS``) ---
  struct ConverterRegistryView {};
  nb::class_<ConverterRegistryView>(nnef_mod, "_ConverterRegistryView",
                                    "Dict-like view of the C++ converter registry.")
      .def(nb::init<>())
      .def("__contains__",
           [](ConverterRegistryView &, const std::string &k) { return nnef::HasOpConverter(k); })
      .def("__getitem__",
           [](ConverterRegistryView &, const std::string &k) -> nb::object {
             if (!nnef::HasOpConverter(k))
               throw nb::key_error(k.c_str());
             // Return a tiny placeholder so callers can detect presence.
             return nb::cast(true);
           })
      .def("__delitem__",
           [](ConverterRegistryView &, const std::string &k) {
             if (!nnef::UnregisterOpConverter(k))
               throw nb::key_error(k.c_str());
           })
      .def("__len__", []() { return nnef::SupportedOps().size(); })
      .def(
          "__iter__",
          [](ConverterRegistryView &) {
            auto ops = nnef::SupportedOps();
            return nb::iter(nb::cast(ops));
          },
          nb::keep_alive<0, 1>())
      .def("keys", []() { return nnef::SupportedOps(); })
      .def(
          "pop",
          [](ConverterRegistryView &, const std::string &k,
             nb::object default_value) -> nb::object {
            bool had = nnef::HasOpConverter(k);
            if (had) {
              nnef::UnregisterOpConverter(k);
              return nb::cast(true);
            }
            if (default_value.is_none())
              throw nb::key_error(k.c_str());
            return default_value;
          },
          nb::arg("key"), nb::arg("default") = nb::none());

  // ----- tensor_io ------------------------------------------------------
  nnef_mod.def(
      "write_nnef_tensor",
      [](const std::string &path, nb::ndarray<nb::ro, nb::c_contig> arr) {
        nnef::WriteNNEFTensor(path, NumpyToNNEFTensor(arr));
      },
      nb::arg("path"), nb::arg("array"));

  nnef_mod.def(
      "read_nnef_tensor",
      [](const std::string &path) -> nb::object {
        NNEFTensor t = nnef::ReadNNEFTensor(path);
        return NNEFTensorToNumpy(t);
      },
      nb::arg("path"));
}

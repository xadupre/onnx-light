// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <array>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::CompressedTensorsQuantizationArgs;
using core::runtime::CompressedTensorsWeights;
using core::runtime::Shape;
using core::runtime::Tensor;
using ONNX_LIGHT_NAMESPACE::EncodedValueProto;

namespace {

constexpr const char *kRevision = "eb058d630f7344099459b6795b1e3eeffdda4b40";
constexpr size_t kDefaultMaxBytes = size_t{1} << 30;

struct PythonCompressedTensorsWeights {
  std::string compression_format;
  nb::dict quantization_args;
  nb::dict tensors;
  std::string upstream_revision = kRevision;
};

nb::object DictValue(const nb::dict &dict, const char *name) {
  return dict.contains(name) ? nb::borrow<nb::object>(dict[name]) : nb::none();
}

bool ExactBool(nb::handle value) { return nb::isinstance<nb::bool_>(value); }

bool ExactInt(nb::handle value) {
  return nb::isinstance<nb::int_>(value) && !nb::isinstance<nb::bool_>(value);
}

std::string DictString(const nb::dict &dict, const char *name, bool required = false) {
  nb::object value = DictValue(dict, name);
  if (value.is_none()) {
    EXT_ENFORCE_INVALID(!required, "compressed-tensors: ", name, " must be a string.");
    return {};
  }
  EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(value), "compressed-tensors: ", name,
                      " must be a string.");
  return nb::cast<std::string>(value);
}

CompressedTensorsQuantizationArgs ParseArguments(const nb::dict &args) {
  static const std::unordered_set<std::string> allowed{
      "num_bits", "type",     "symmetric",   "strategy", "group_size", "block_structure",
      "dynamic",  "actorder", "scale_dtype", "zp_dtype", "observer",   "observer_kwargs"};
  static const std::array<const char *, 4> required{"num_bits", "type", "symmetric", "strategy"};
  for (const char *name : required)
    EXT_ENFORCE_INVALID(args.contains(name), "compressed-tensors: missing ", name, ".");
  for (auto item : args) {
    EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(item.first),
                        "compressed-tensors: argument names must be strings.");
    const std::string name = nb::cast<std::string>(item.first);
    EXT_ENFORCE_INVALID(allowed.contains(name),
                        "compressed-tensors: unsupported QuantizationArgs field ", name, ".");
  }
  CompressedTensorsQuantizationArgs result;
  nb::object bits = DictValue(args, "num_bits");
  EXT_ENFORCE_INVALID(ExactInt(bits), "compressed-tensors: num_bits must be an integer.");
  const int64_t bit_value = nb::cast<int64_t>(bits);
  EXT_ENFORCE_INVALID(bit_value >= 1 && bit_value <= 8,
                      "compressed-tensors: num_bits must be between 1 and 8.");
  result.num_bits = static_cast<uint32_t>(bit_value);
  result.type = DictString(args, "type", true);
  nb::object symmetric = DictValue(args, "symmetric");
  EXT_ENFORCE_INVALID(ExactBool(symmetric), "compressed-tensors: symmetric must be a boolean.");
  result.symmetric = nb::cast<bool>(symmetric);
  result.strategy = DictString(args, "strategy", true);
  nb::object group_size = DictValue(args, "group_size");
  if (!group_size.is_none()) {
    EXT_ENFORCE_INVALID(ExactInt(group_size),
                        "compressed-tensors: group_size must be an integer or null.");
    result.group_size = nb::cast<int64_t>(group_size);
  }
  for (const auto &[name, value] :
       {std::pair<const char *, std::optional<std::string> *>("scale_dtype", &result.scale_dtype),
        {"zp_dtype", &result.zp_dtype}}) {
    nb::object dtype = DictValue(args, name);
    if (!dtype.is_none()) {
      EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(dtype), "compressed-tensors: ", name,
                          " must be a string or null.");
      *value = nb::cast<std::string>(dtype);
    }
  }
  nb::object dynamic = DictValue(args, "dynamic");
  if (!dynamic.is_none()) {
    EXT_ENFORCE_INVALID(ExactBool(dynamic), "compressed-tensors: dynamic must be a boolean.");
    result.dynamic = nb::cast<bool>(dynamic);
  }
  nb::object actorder = DictValue(args, "actorder");
  result.actorder = !actorder.is_none() && !(ExactBool(actorder) && !nb::cast<bool>(actorder));
  result.block_structure = !DictValue(args, "block_structure").is_none();
  return result;
}

int32_t NumpyType(nb::handle array) {
  nb::object dtype = array.attr("dtype");
  const std::string name = nb::cast<std::string>(nb::str(dtype));
  if (name == "int8")
    return TensorProto::INT8;
  if (name == "uint8")
    return TensorProto::UINT8;
  if (name == "int32")
    return TensorProto::INT32;
  if (name == "int64")
    return TensorProto::INT64;
  if (name == "float32")
    return TensorProto::FLOAT;
  if (name == "float8_e4m3fn")
    return TensorProto::FLOAT8E4M3FN;
  EXT_THROW_INVALID("compressed-tensors: unsupported NumPy dtype ", name, ".");
}

Tensor TensorFromArray(const std::string &name, nb::handle array, size_t max_bytes) {
  nb::module_ numpy = nb::module_::import_("numpy");
  EXT_ENFORCE_INVALID(nb::isinstance(array, numpy.attr("ndarray")), "compressed-tensors: ", name,
                      " must be a loaded NumPy array.");
  nb::object value = nb::borrow<nb::object>(array);
  const size_t nbytes = nb::cast<size_t>(value.attr("nbytes"));
  EXT_ENFORCE_INVALID(nbytes <= max_bytes, "compressed-tensors: checkpoint exceeds max_bytes.");
  const int32_t type = NumpyType(value);
  Shape shape(nb::cast<std::vector<int64_t>>(value.attr("shape")));
  nb::bytes raw = nb::cast<nb::bytes>(value.attr("tobytes")("C"));
  std::vector<uint8_t> bytes(raw.size());
  if (!bytes.empty())
    std::memcpy(bytes.data(), raw.data(), bytes.size());
  return Tensor(name, type, std::move(shape), std::move(bytes));
}

CompressedTensorsWeights ToNative(const PythonCompressedTensorsWeights &checkpoint,
                                  size_t max_bytes) {
  EXT_ENFORCE_INVALID(checkpoint.tensors.size() > 0,
                      "compressed-tensors: tensors must be a nonempty mapping.");
  CompressedTensorsWeights result;
  result.compression_format = checkpoint.compression_format;
  result.quantization_args = ParseArguments(checkpoint.quantization_args);
  result.upstream_revision = checkpoint.upstream_revision;
  size_t total = 0;
  for (auto item : checkpoint.tensors) {
    EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(item.first),
                        "compressed-tensors: tensor names must be strings.");
    const std::string name = nb::cast<std::string>(item.first);
    Tensor tensor = TensorFromArray(name, item.second, max_bytes - total);
    EXT_ENFORCE_INVALID(tensor.size_bytes() <= max_bytes - total,
                        "compressed-tensors: checkpoint exceeds max_bytes.");
    total += tensor.size_bytes();
    result.tensors.emplace(name, std::move(tensor));
  }
  return result;
}

nb::object ArrayFromTensor(const Tensor &tensor) {
  nb::module_ numpy = nb::module_::import_("numpy");
  const char *dtype = nullptr;
  switch (static_cast<TensorProto::DataType>(tensor.data_type)) {
  case TensorProto::INT8:
    dtype = "int8";
    break;
  case TensorProto::UINT8:
    dtype = "uint8";
    break;
  case TensorProto::INT32:
    dtype = "int32";
    break;
  case TensorProto::INT64:
    dtype = "int64";
    break;
  case TensorProto::FLOAT:
    dtype = "float32";
    break;
  case TensorProto::FLOAT8E4M3FN:
    dtype = "float8_e4m3fn";
    break;
  default:
    EXT_THROW_INVALID("compressed-tensors: cannot export an unsupported array dtype.");
  }
  nb::bytes raw(reinterpret_cast<const char *>(tensor.bytes()), tensor.size_bytes());
  nb::object array = numpy.attr("frombuffer")(raw, dtype);
  const std::vector<int64_t> shape(tensor.shape.begin(), tensor.shape.end());
  array = array.attr("reshape")(shape);
  return array.attr("copy")();
}

nb::dict TensorDictionary(const CompressedTensorsWeights &checkpoint) {
  nb::dict result;
  for (const auto &[name, tensor] : checkpoint.tensors)
    result[nb::str(name.c_str())] = ArrayFromTensor(tensor);
  return result;
}

} // namespace

void AddOnnxPyCompressedTensors(nb::module_ &runtime) {
  nb::class_<PythonCompressedTensorsWeights>(runtime, "CompressedTensorsWeights")
      .def(
          "__init__",
          [](PythonCompressedTensorsWeights *self, std::string compression_format,
             nb::dict quantization_args, nb::dict tensors, std::string upstream_revision) {
            new (self) PythonCompressedTensorsWeights{
                std::move(compression_format), std::move(quantization_args), std::move(tensors),
                std::move(upstream_revision)};
          },
          nb::arg("compression_format"), nb::arg("quantization_args"), nb::arg("tensors"),
          nb::arg("upstream_revision") = kRevision)
      .def_rw("compression_format", &PythonCompressedTensorsWeights::compression_format)
      .def_rw("quantization_args", &PythonCompressedTensorsWeights::quantization_args)
      .def_rw("tensors", &PythonCompressedTensorsWeights::tensors)
      .def_rw("upstream_revision", &PythonCompressedTensorsWeights::upstream_revision);
  runtime.attr("compressed_tensors_upstream_revision") = kRevision;
  runtime.def(
      "import_compressed_tensors_weights",
      [](const PythonCompressedTensorsWeights &checkpoint, size_t max_bytes) {
        return core::runtime::ImportCompressedTensorsWeights(ToNative(checkpoint, max_bytes),
                                                             max_bytes);
      },
      nb::arg("checkpoint"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_compressed_tensors_weights",
      [](const EncodedValueProto &value, const std::string &compression_format,
         const nb::dict &quantization_args, const std::string &upstream_revision,
         size_t max_bytes) {
        auto exported = core::runtime::ExportCompressedTensorsWeights(
            value, compression_format, ParseArguments(quantization_args), upstream_revision,
            max_bytes);
        PythonCompressedTensorsWeights result;
        result.compression_format = exported.compression_format;
        result.quantization_args = nb::dict(quantization_args);
        result.tensors = TensorDictionary(exported);
        result.upstream_revision = exported.upstream_revision;
        return result;
      },
      nb::arg("value"), nb::arg("compression_format"), nb::arg("quantization_args"),
      nb::arg("upstream_revision") = kRevision, nb::arg("max_bytes") = kDefaultMaxBytes);
}

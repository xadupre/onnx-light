// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/quantization.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/unordered_set.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <limits>
#include <optional>
#include <utility>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using core::runtime::AwqWeights;
using core::runtime::BitsAndBytesWeights;
using core::runtime::CompressedTensorsQuantizationArgs;
using core::runtime::CompressedTensorsWeights;
using core::runtime::Exl3Weights;
using core::runtime::GptqWeights;
using core::runtime::Shape;
using core::runtime::Tensor;
using ONNX_LIGHT_NAMESPACE::EncodedValueProto;

namespace {

constexpr const char *kRevision = "eb058d630f7344099459b6795b1e3eeffdda4b40";
constexpr const char *kExl3Revision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
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
  if (name == "int16")
    return TensorProto::INT16;
  if (name == "uint8")
    return TensorProto::UINT8;
  if (name == "int32")
    return TensorProto::INT32;
  if (name == "int64")
    return TensorProto::INT64;
  if (name == "float32")
    return TensorProto::FLOAT;
  if (name == "float16")
    return TensorProto::FLOAT16;
  if (name == "bfloat16")
    return TensorProto::BFLOAT16;
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
  EXT_ENFORCE_INVALID(raw.size() == nbytes && raw.size() <= max_bytes,
                      "compressed-tensors: inconsistent array byte extent.");
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
  case TensorProto::INT16:
    dtype = "int16";
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
  case TensorProto::FLOAT16:
    dtype = "float16";
    break;
  case TensorProto::BFLOAT16:
    nb::module_::import_("ml_dtypes");
    dtype = "bfloat16";
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

GptqWeights GptqMetadata(const std::string &format, int64_t in_features, int64_t out_features,
                         int64_t group_size, uint32_t bits, int64_t group_axis, int32_t word_dtype,
                         const std::string &revision, size_t max_bytes) {
  EXT_ENFORCE_INVALID(revision == kExl3Revision, "GPTQModel unsupported upstream revision.");
  EXT_ENFORCE_INVALID(format == "gptq" || format == "gptq_v2" || format == "gptq_p",
                      "GPTQModel unsupported format.");
  EXT_ENFORCE_INVALID(max_bytes > 0, "GPTQModel max_bytes must be positive.");
  GptqWeights result;
  result.format = format;
  result.in_features = in_features;
  result.out_features = out_features;
  result.group_size = group_size;
  result.bits = bits;
  result.group_axis = group_axis;
  result.word_dtype = word_dtype;
  result.upstream_revision = revision;
  return result;
}

Tensor BorrowGptqArray(const std::string &name, nb::handle value, size_t &total, size_t max_bytes) {
  nb::module_ numpy = nb::module_::import_("numpy");
  EXT_ENFORCE_INVALID(nb::isinstance(value, numpy.attr("ndarray")),
                      "GPTQModel tensors must be loaded NumPy arrays.");
  EXT_ENFORCE_INVALID(nb::cast<bool>(value.attr("flags").attr("c_contiguous")) &&
                          nb::cast<bool>(value.attr("dtype").attr("isnative")),
                      "GPTQModel arrays must be C-contiguous with native byte order.");
  const int32_t type = NumpyType(value);
  nb::object storage = nb::borrow<nb::object>(value);
  if (type == TensorProto::BFLOAT16)
    storage = storage.attr("view")(numpy.attr("uint16"));
  const auto array =
      nb::cast<nb::ndarray<nb::numpy, nb::ro, nb::c_contig, nb::device::cpu>>(storage);
  EXT_ENFORCE_INVALID(total <= max_bytes && array.nbytes() <= max_bytes - total,
                      "GPTQModel tensors exceed max_bytes.");
  Shape shape;
  for (size_t i = 0; i < array.ndim(); ++i) {
    EXT_ENFORCE_INVALID(array.shape(i) <=
                            static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                        "GPTQModel dimension size overflow.");
    shape.push_back(static_cast<int64_t>(array.shape(i)));
  }
  total += array.nbytes();
  return Tensor::Borrow(name, type, std::move(shape), static_cast<const uint8_t *>(array.data()),
                        array.nbytes());
}

void ConvertNativeLittleEndianWords(Tensor &tensor) {
  if constexpr (std::endian::native == std::endian::big) {
    const size_t width = tensor.element_size();
    if (width == 1)
      return;
    for (size_t offset = 0; offset < tensor.size_bytes(); offset += width)
      std::reverse(tensor.mutable_bytes() + offset, tensor.mutable_bytes() + offset + width);
  }
}

Tensor BitsAndBytesArray(const std::string &name, nb::handle value, size_t &total,
                         size_t max_bytes) {
  Tensor tensor = BorrowGptqArray(name, value, total, max_bytes);
  if constexpr (std::endian::native == std::endian::big) {
    if (tensor.element_size() > 1) {
      tensor = tensor.ToOwned();
      ConvertNativeLittleEndianWords(tensor);
    }
  }
  return tensor;
}

nb::object ArrayFromLittleEndianTensor(Tensor tensor) {
  ConvertNativeLittleEndianWords(tensor);
  return ArrayFromTensor(tensor);
}

} // namespace

void AddOnnxPyCompressedTensors(nb::module_ &runtime) {
  runtime.attr("bitsandbytes_upstream_revision") = kExl3Revision;
  runtime.def(
      "import_bitsandbytes_weights",
      [](const nb::dict &tensors, const std::string &revision, size_t max_bytes) {
        EXT_ENFORCE_INVALID(revision == kExl3Revision,
                            "bitsandbytes unsupported upstream revision.");
        EXT_ENFORCE_INVALID(tensors.size() >= 2 && tensors.size() <= 6,
                            "bitsandbytes requires a single module's weight and state tensors.");
        BitsAndBytesWeights weights;
        weights.upstream_revision = revision;
        size_t total = 0;
        for (auto item : tensors) {
          EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(item.first),
                              "bitsandbytes tensor names must be strings.");
          const std::string name = nb::cast<std::string>(item.first);
          EXT_ENFORCE_INVALID(name == "weight" || name == "weight_scb" || name == "weight_absmax" ||
                                  name == "weight_quant_map" || name == "weight_nested_absmax" ||
                                  name == "weight_nested_quant_map" || name == "weight_quant_state",
                              "bitsandbytes unexpected side tensor.");
          weights.tensors.emplace(name, BitsAndBytesArray(name, item.second, total, max_bytes));
        }
        return core::runtime::ImportBitsAndBytesWeights(weights, max_bytes);
      },
      nb::arg("tensors"), nb::arg("upstream_revision") = kExl3Revision,
      nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_bitsandbytes_weights",
      [](const EncodedValueProto &value, size_t max_bytes) {
        auto weights = core::runtime::ExportBitsAndBytesWeights(value, max_bytes);
        nb::dict result;
        for (auto &[name, tensor] : weights.tensors)
          result[nb::str(name.c_str())] = ArrayFromLittleEndianTensor(std::move(tensor));
        return result;
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "unpack_bitsandbytes_codes",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromLittleEndianTensor(
            core::runtime::UnpackBitsAndBytesCodes(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "reconstruct_bitsandbytes_weights",
      [](const EncodedValueProto &value, size_t max_bytes, int32_t reconstruction_dtype) {
        return ArrayFromLittleEndianTensor(core::runtime::ReconstructBitsAndBytesWeights(
            value, max_bytes, {}, reconstruction_dtype));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes,
      nb::arg("reconstruction_dtype") = static_cast<int32_t>(TensorProto::FLOAT));
  runtime.attr("gptqmodel_upstream_revision") = kExl3Revision;
  runtime.def(
      "import_gptq_weights",
      [](const nb::dict &tensors, const std::string &format, int64_t in_features,
         int64_t out_features, int64_t group_size, uint32_t bits, int64_t group_axis,
         int32_t word_dtype, const std::string &revision, size_t max_bytes) {
        auto weights = GptqMetadata(format, in_features, out_features, group_size, bits, group_axis,
                                    word_dtype, revision, max_bytes);
        EXT_ENFORCE_INVALID(tensors.size() == 4,
                            "GPTQModel requires qweight, qzeros, scales and g_idx.");
        size_t total = 0;
        for (auto item : tensors) {
          EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(item.first),
                              "GPTQModel tensor names must be strings.");
          const std::string name = nb::cast<std::string>(item.first);
          EXT_ENFORCE_INVALID(name == "qweight" || name == "qzeros" || name == "scales" ||
                                  name == "g_idx",
                              "GPTQModel unexpected side tensor.");
          weights.tensors.emplace(name, BorrowGptqArray(name, item.second, total, max_bytes));
        }
        return core::runtime::ImportGptqWeights(weights, max_bytes);
      },
      nb::arg("tensors"), nb::arg("format"), nb::arg("in_features"), nb::arg("out_features"),
      nb::arg("group_size") = 128, nb::arg("bits") = 4, nb::arg("group_axis") = 1,
      nb::arg("word_dtype") = static_cast<int32_t>(TensorProto::INT32),
      nb::arg("upstream_revision") = kExl3Revision, nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "pack_gptq_weights",
      [](nb::handle codes, nb::handle zeros, nb::handle scales, nb::handle g_idx,
         const std::string &format, int64_t in_features, int64_t out_features, int64_t group_size,
         uint32_t bits, int64_t group_axis, int32_t word_dtype, const std::string &revision,
         size_t max_bytes) {
        const auto metadata = GptqMetadata(format, in_features, out_features, group_size, bits,
                                           group_axis, word_dtype, revision, max_bytes);
        size_t total = 0;
        const auto c = BorrowGptqArray("codes", codes, total, max_bytes);
        const auto z = BorrowGptqArray("zeros", zeros, total, max_bytes);
        const auto s = BorrowGptqArray("scales", scales, total, max_bytes);
        const auto g = BorrowGptqArray("g_idx", g_idx, total, max_bytes);
        return core::runtime::PackGptqWeights(metadata, c, z, s, g, max_bytes);
      },
      nb::arg("codes"), nb::arg("zeros"), nb::arg("scales"), nb::arg("g_idx"), nb::arg("format"),
      nb::arg("in_features"), nb::arg("out_features"), nb::arg("group_size") = 128,
      nb::arg("bits") = 4, nb::arg("group_axis") = 1,
      nb::arg("word_dtype") = static_cast<int32_t>(TensorProto::INT32),
      nb::arg("upstream_revision") = kExl3Revision, nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_gptq_weights",
      [](const EncodedValueProto &value, size_t max_bytes) {
        auto weights = core::runtime::ExportGptqWeights(value, max_bytes);
        nb::dict tensors;
        for (const auto &[name, tensor] : weights.tensors)
          tensors[nb::str(name.c_str())] = ArrayFromTensor(tensor);
        nb::dict result;
        result["format"] = weights.format;
        result["in_features"] = weights.in_features;
        result["out_features"] = weights.out_features;
        result["group_size"] = weights.group_size;
        result["bits"] = weights.bits;
        result["group_axis"] = weights.group_axis;
        result["word_dtype"] = weights.word_dtype;
        result["upstream_revision"] = weights.upstream_revision;
        result["tensors"] = std::move(tensors);
        return result;
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "unpack_gptq_codes",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromTensor(core::runtime::UnpackGptqCodes(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "unpack_gptq_zeros",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromTensor(core::runtime::UnpackGptqZeros(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "reconstruct_gptq_weights",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromTensor(core::runtime::ReconstructGptqWeights(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.attr("awq_upstream_revision") = kExl3Revision;
  runtime.def(
      "import_awq_weights",
      [](const nb::dict &tensors, const std::string &variant, int64_t in_features,
         int64_t out_features, int64_t group_size, uint32_t bits, int64_t group_axis,
         const std::string &revision, size_t max_bytes) {
        EXT_ENFORCE_INVALID(revision == kExl3Revision, "AWQ unsupported upstream revision.");
        EXT_ENFORCE_INVALID(variant == "gemm" || variant == "gemv" || variant == "gemv_fast" ||
                                variant == "llm-awq",
                            "AWQ unsupported variant.");
        EXT_ENFORCE_INVALID(bits == 4 && group_axis == 1 && in_features > 0 && out_features > 0 &&
                                (group_size > 0 || group_size == -1) && max_bytes > 0,
                            "AWQ invalid bits, axis, dimensions, group_size or max_bytes.");
        EXT_ENFORCE_INVALID(tensors.size() == 3, "AWQ requires exactly three native tensors.");
        AwqWeights weights;
        weights.variant = variant;
        weights.in_features = in_features;
        weights.out_features = out_features;
        weights.group_size = group_size;
        weights.bits = bits;
        weights.group_axis = group_axis;
        weights.upstream_revision = revision;
        nb::module_ numpy = nb::module_::import_("numpy");
        size_t total = 0;
        for (auto item : tensors) {
          EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(item.first),
                              "AWQ tensor names must be strings.");
          const std::string name = nb::cast<std::string>(item.first);
          EXT_ENFORCE_INVALID(name == "qweight" || name == "qzeros" || name == "scales" ||
                                  name == "scaled_zeros",
                              "AWQ unexpected side tensor.");
          EXT_ENFORCE_INVALID(nb::isinstance(item.second, numpy.attr("ndarray")),
                              "AWQ tensors must be loaded NumPy arrays.");
          EXT_ENFORCE_INVALID(nb::cast<bool>(item.second.attr("flags").attr("c_contiguous")) &&
                                  nb::cast<bool>(item.second.attr("dtype").attr("isnative")),
                              "AWQ arrays must be C-contiguous with native byte order.");
          const int32_t type = NumpyType(item.second);
          nb::object storage = nb::borrow<nb::object>(item.second);
          if (type == TensorProto::BFLOAT16)
            storage = storage.attr("view")(numpy.attr("uint16"));
          const auto array =
              nb::cast<nb::ndarray<nb::numpy, nb::ro, nb::c_contig, nb::device::cpu>>(storage);
          EXT_ENFORCE_INVALID(array.nbytes() <= max_bytes - total,
                              "AWQ checkpoint exceeds max_bytes.");
          Shape shape;
          for (size_t i = 0; i < array.ndim(); ++i) {
            EXT_ENFORCE_INVALID(array.shape(i) <=
                                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                                "AWQ dimension size overflow.");
            shape.push_back(static_cast<int64_t>(array.shape(i)));
          }
          total += array.nbytes();
          weights.tensors.emplace(name, Tensor::Borrow(name, type, std::move(shape),
                                                       static_cast<const uint8_t *>(array.data()),
                                                       array.nbytes()));
        }
        return core::runtime::ImportAwqWeights(weights, max_bytes);
      },
      nb::arg("tensors"), nb::arg("variant"), nb::arg("in_features"), nb::arg("out_features"),
      nb::arg("group_size") = 128, nb::arg("bits") = 4, nb::arg("group_axis") = 1,
      nb::arg("upstream_revision") = kExl3Revision, nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_awq_weights",
      [](const EncodedValueProto &value, size_t max_bytes) {
        auto weights = core::runtime::ExportAwqWeights(value, max_bytes);
        nb::dict tensors;
        for (const auto &[name, tensor] : weights.tensors)
          tensors[nb::str(name.c_str())] = ArrayFromTensor(tensor);
        nb::dict result;
        result["variant"] = weights.variant;
        result["in_features"] = weights.in_features;
        result["out_features"] = weights.out_features;
        result["group_size"] = weights.group_size;
        result["bits"] = weights.bits;
        result["group_axis"] = weights.group_axis;
        result["upstream_revision"] = weights.upstream_revision;
        result["tensors"] = std::move(tensors);
        return result;
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "unpack_awq_codes",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromTensor(core::runtime::UnpackAwqCodes(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "reconstruct_awq_weights",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromTensor(core::runtime::ReconstructAwqWeights(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.attr("exl3_upstream_revision") = kExl3Revision;
  runtime.def(
      "import_exl3_weights",
      [](const nb::dict &tensors, const std::string &revision, size_t max_bytes) {
        EXT_ENFORCE_INVALID(revision == kExl3Revision, "EXL3 unsupported upstream revision.");
        EXT_ENFORCE_INVALID(tensors.size() >= 3 && tensors.size() <= 4,
                            "EXL3 requires trellis, suh, svh and at most one codebook marker.");
        Exl3Weights weights;
        weights.upstream_revision = revision;
        size_t total = 0;
        for (auto item : tensors) {
          EXT_ENFORCE_INVALID(nb::isinstance<nb::str>(item.first),
                              "EXL3 tensor names must be strings.");
          const std::string name = nb::cast<std::string>(item.first);
          EXT_ENFORCE_INVALID(name == "trellis" || name == "suh" || name == "svh" ||
                                  name == "mcg" || name == "mul1",
                              "EXL3 unexpected side tensor.");
          Tensor tensor = TensorFromArray(name, item.second, max_bytes - total);
          total += tensor.size_bytes();
          weights.tensors.emplace(name, std::move(tensor));
        }
        return core::runtime::ImportExl3Weights(weights, max_bytes);
      },
      nb::arg("tensors"), nb::arg("upstream_revision") = kExl3Revision,
      nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_exl3_weights",
      [](const EncodedValueProto &value, size_t max_bytes) {
        auto weights = core::runtime::ExportExl3Weights(value, max_bytes);
        nb::dict result;
        for (const auto &[name, tensor] : weights.tensors)
          result[nb::str(name.c_str())] = ArrayFromTensor(tensor);
        return result;
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "unpack_exl3_states",
      [](const EncodedValueProto &value, size_t max_bytes) {
        return ArrayFromTensor(core::runtime::UnpackExl3States(value, max_bytes));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "reconstruct_exl3_weights",
      [](const EncodedValueProto &value, size_t max_bytes, int32_t reconstruction_dtype) {
        return ArrayFromTensor(
            core::runtime::ReconstructExl3Weights(value, max_bytes, {}, reconstruction_dtype));
      },
      nb::arg("value"), nb::arg("max_bytes") = kDefaultMaxBytes,
      nb::arg("reconstruction_dtype") = static_cast<int32_t>(TensorProto::FLOAT));
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

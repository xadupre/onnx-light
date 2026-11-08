// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/gguf.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr size_t kDefaultMaxBytes = size_t{1} << 30;

StructTypeCatalogue Catalogue(const ModelProto *model) {
  StructTypeCatalogue catalogue;
  if (model)
    catalogue.Build(*model);
  return catalogue;
}

struct PythonGptqModelGgufWeights {
  nb::object qweight;
  std::vector<int64_t> shape;
  std::string subtype;
  std::string layout_version = std::string(kGgufLayoutVersion);
  std::string upstream_revision = std::string(kGgufUpstreamRevision);
};

GptqModelGgufWeights ToNative(const PythonGptqModelGgufWeights &weights, size_t max_bytes,
                              nb::object &owner) {
  nb::module_ numpy = nb::module_::import_("numpy");
  EXT_ENFORCE_INVALID(nb::isinstance(weights.qweight, numpy.attr("ndarray")),
                      "GGUF: qweight must be a NumPy array.");
  const nb::object dtype = weights.qweight.attr("dtype");
  EXT_ENFORCE_INVALID(nb::cast<std::string>(nb::str(dtype)) == "uint8",
                      "GGUF: qweight must have UINT8 dtype.");
  const size_t nbytes = nb::cast<size_t>(weights.qweight.attr("nbytes"));
  EXT_ENFORCE_INVALID(max_bytes > 0 && nbytes <= max_bytes, "GGUF: qweight exceeds max_bytes.");
  const auto packed_shape = nb::cast<std::vector<int64_t>>(weights.qweight.attr("shape"));
  owner = numpy.attr("ascontiguousarray")(weights.qweight);
  const auto array = nb::cast<nb::ndarray<nb::numpy, const uint8_t, nb::c_contig>>(owner);
  GptqModelGgufWeights result;
  result.qweight =
      Tensor::Borrow("qweight", TensorProto::UINT8, Shape(packed_shape), array.data(), nbytes);
  result.shape = Shape(weights.shape);
  result.subtype = weights.subtype;
  result.layout_version = weights.layout_version;
  result.upstream_revision = weights.upstream_revision;
  return result;
}

PythonGptqModelGgufWeights FromNative(const GptqModelGgufWeights &weights) {
  nb::module_ numpy = nb::module_::import_("numpy");
  nb::bytes raw(reinterpret_cast<const char *>(weights.qweight.bytes()),
                weights.qweight.size_bytes());
  const std::vector<int64_t> packed_shape(weights.qweight.shape.begin(),
                                          weights.qweight.shape.end());
  PythonGptqModelGgufWeights result;
  result.qweight =
      numpy.attr("frombuffer")(raw, "uint8").attr("reshape")(packed_shape).attr("copy")();
  result.shape.assign(weights.shape.begin(), weights.shape.end());
  result.subtype = weights.subtype;
  result.layout_version = weights.layout_version;
  result.upstream_revision = weights.upstream_revision;
  return result;
}

} // namespace

void AddOnnxPyGguf(nb::module_ &runtime) {
  runtime.attr("gguf_layout_version") = nb::str(kGgufLayoutVersion);
  runtime.attr("gguf_upstream_revision") = nb::str(kGgufUpstreamRevision);
  runtime.attr("gguf_reference_revision") = nb::str(kGgufReferenceRevision);
  nb::class_<GgufBlockLayout>(runtime, "GgufBlockLayout")
      .def_ro("subtype", &GgufBlockLayout::subtype)
      .def_ro("block_size", &GgufBlockLayout::block_size)
      .def_ro("type_size", &GgufBlockLayout::type_size);
  nb::class_<GgufBlockParameters>(runtime, "GgufBlockParameters")
      .def_ro("scales", &GgufBlockParameters::scales)
      .def_ro("minima", &GgufBlockParameters::minima);
  runtime.def("gguf_block_layouts", &GgufBlockLayouts);
  runtime.def(
      "import_gguf_tensor_payload",
      [](nb::bytes data, const std::vector<int64_t> &shape, std::string subtype,
         std::string layout_version, std::string upstream_revision, size_t max_bytes) {
        return ImportGgufTensorPayload({data.c_str(), data.size()}, Shape(shape),
                                       std::move(subtype), std::move(layout_version),
                                       std::move(upstream_revision), max_bytes);
      },
      nb::arg("data"), nb::arg("shape"), nb::arg("subtype"),
      nb::arg("layout_version") = std::string(kGgufLayoutVersion),
      nb::arg("upstream_revision") = std::string(kGgufUpstreamRevision),
      nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_gguf_tensor_payload",
      [](const EncodedValueProto &payload, size_t max_bytes, const ModelProto *model) {
        const auto data = ExportGgufTensorPayload(payload, max_bytes, Catalogue(model));
        return nb::bytes(data.data(), data.size());
      },
      nb::arg("payload"), nb::arg("max_bytes") = kDefaultMaxBytes,
      nb::arg("model").none() = nullptr);
  runtime.def(
      "dequantize_gguf_tensor_payload",
      [](const EncodedValueProto &payload, size_t max_bytes, const ModelProto *model) {
        return DequantizeGgufTensorPayload(payload, max_bytes, Catalogue(model));
      },
      nb::arg("payload"), nb::arg("max_bytes") = kDefaultMaxBytes,
      nb::arg("model").none() = nullptr);
  runtime.def(
      "read_gguf_block_parameters",
      [](const EncodedValueProto &payload, size_t block_index, size_t max_bytes,
         const ModelProto *model) {
        return ReadGgufBlockParameters(payload, block_index, max_bytes, Catalogue(model));
      },
      nb::arg("payload"), nb::arg("block_index"), nb::arg("max_bytes") = kDefaultMaxBytes,
      nb::arg("model").none() = nullptr);
  nb::class_<PythonGptqModelGgufWeights>(runtime, "GptqModelGgufWeights")
      .def(
          "__init__",
          [](PythonGptqModelGgufWeights *self, nb::object qweight, std::vector<int64_t> shape,
             std::string subtype, std::string layout_version, std::string upstream_revision) {
            new (self)
                PythonGptqModelGgufWeights{std::move(qweight), std::move(shape), std::move(subtype),
                                           std::move(layout_version), std::move(upstream_revision)};
          },
          nb::arg("qweight"), nb::arg("shape"), nb::arg("subtype"),
          nb::arg("layout_version") = std::string(kGgufLayoutVersion),
          nb::arg("upstream_revision") = std::string(kGgufUpstreamRevision))
      .def_rw("qweight", &PythonGptqModelGgufWeights::qweight)
      .def_rw("shape", &PythonGptqModelGgufWeights::shape)
      .def_rw("subtype", &PythonGptqModelGgufWeights::subtype)
      .def_rw("layout_version", &PythonGptqModelGgufWeights::layout_version)
      .def_rw("upstream_revision", &PythonGptqModelGgufWeights::upstream_revision);
  runtime.def(
      "import_gptqmodel_gguf_weights",
      [](const PythonGptqModelGgufWeights &weights, size_t max_bytes) {
        nb::object owner;
        return ImportGptqModelGgufWeights(ToNative(weights, max_bytes, owner), max_bytes);
      },
      nb::arg("weights"), nb::arg("max_bytes") = kDefaultMaxBytes);
  runtime.def(
      "export_gptqmodel_gguf_weights",
      [](const EncodedValueProto &payload, size_t max_bytes, const ModelProto *model) {
        return FromNative(ExportGptqModelGgufWeights(payload, max_bytes, Catalogue(model)));
      },
      nb::arg("payload"), nb::arg("max_bytes") = kDefaultMaxBytes,
      nb::arg("model").none() = nullptr);
}

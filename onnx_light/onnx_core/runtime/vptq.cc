// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_core/runtime/quantization.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {
namespace {
constexpr const char *kName = "onnx_light.vptq_weights.v1";
constexpr const char *kRevision = "c1976918cfe3703fbcf6d1ce2440a82b26394375";
constexpr int kMetadata = 26;

size_t Mul(size_t a, size_t b) {
  EXT_ENFORCE_INVALID(!b || a <= std::numeric_limits<size_t>::max() / b, "VPTQ size overflow.");
  return a * b;
}
size_t Ceil(size_t a, size_t b) { return a / b + (a % b != 0); }
size_t Positive(int64_t value) {
  EXT_ENFORCE_INVALID(value > 0 && uint64_t(value) <= std::numeric_limits<size_t>::max(),
                      "VPTQ invalid positive dimension.");
  return static_cast<size_t>(value);
}
uint32_t Read(const uint8_t *p, size_t width) {
  uint32_t result = 0;
  for (size_t i = 0; i < width; ++i)
    result |= uint32_t(p[i]) << (8 * i);
  return result;
}
void Copy(uint8_t *dst, const uint8_t *src, size_t count, size_t width) {
  std::memcpy(dst, src, count);
  if constexpr (std::endian::native == std::endian::big)
    for (size_t i = 0; i < count; i += width)
      std::reverse(dst + i, dst + i + width);
}
struct Spec {
  std::string name;
  int32_t dtype;
  Shape shape;
  size_t offset, bytes, width;
};
struct Layout {
  size_t k, n, groups, columns, outliers, vectors, words, total = 0, count;
  unsigned main_bits, residual_bits;
  int32_t dtype;
  std::vector<Spec> specs;
  Layout(const VptqWeights &w, int32_t type, bool bias, size_t limit) : dtype(type) {
    EXT_ENFORCE_INVALID(limit > 0 && w.upstream_revision == kRevision && w.schema_version == 1,
                        "VPTQ invalid limit, schema version or upstream revision.");
    k = Positive(w.in_features);
    n = Positive(w.out_features);
    count = Mul(k, n);
    EXT_ENFORCE_INVALID(count <= limit / 4, "VPTQ FLOAT output exceeds max_bytes.");
    EXT_ENFORCE_INVALID(!w.enable_perm || k <= size_t(std::numeric_limits<int32_t>::max()) + 1,
                        "VPTQ permutation exceeds INT32 channel range.");
    EXT_ENFORCE_INVALID(w.bits >= 1 && w.bits <= 8 && w.vector_lens[1] > 0 &&
                            w.num_centroids[1] > 1 && w.num_res_centroids[1] != 1 &&
                            w.num_res_centroids[0] <= 1,
                        "VPTQ unsupported codebook or outlier residual configuration.");
    EXT_ENFORCE_INVALID(type == TensorProto::FLOAT || type == TensorProto::FLOAT16 ||
                            type == TensorProto::BFLOAT16,
                        "VPTQ unsupported codebook dtype.");
    EXT_ENFORCE_INVALID(std::isfinite(w.npercent) && w.npercent >= 0 && w.npercent < 100 &&
                            std::isfinite(w.kmeans_tol) && std::isfinite(w.percdamp) &&
                            w.kmeans_mode.size() <= std::min(limit, size_t{4096}) &&
                            (w.norm_dim == 0 || w.norm_dim == 1),
                        "VPTQ invalid constructor metadata.");
    const bool outlier = w.npercent > 0 && w.vector_lens[0] > 0 && w.num_centroids[0] > 1;
    EXT_ENFORCE_INVALID(w.npercent == 0 || outlier, "VPTQ unusable outlier codebook.");
    const double initial = std::ceil((w.npercent / 100.0) * double(k));
    EXT_ENFORCE_INVALID(initial >= 0 && initial < double(k), "VPTQ empty grouped layout.");
    outliers = outlier ? static_cast<size_t>(initial) : 0;
    EXT_ENFORCE_INVALID((w.group_size == -1 || w.group_size > 0) &&
                            (w.group_num == -1 || w.group_num > 0) &&
                            (w.group_size == -1 || w.group_num == -1),
                        "VPTQ grouping knobs must be mutually exclusive.");
    if (w.group_num != -1) {
      groups = Positive(w.group_num);
      EXT_ENFORCE_INVALID(groups <= k - outliers, "VPTQ empty groups.");
      if (outlier)
        outliers += (k - outliers) % groups;
      columns = outlier ? (k - outliers) / groups : Ceil(k, groups);
    } else {
      columns = w.group_size == -1 ? k - outliers : Positive(w.group_size);
      groups = Ceil(k - outliers, columns);
    }
    EXT_ENFORCE_INVALID(groups && columns && Mul(groups, columns) >= k - outliers,
                        "VPTQ incomplete grouped layout.");
    const auto bits = [](int64_t entries) {
      return static_cast<unsigned>(std::bit_width(uint64_t(entries - 1)));
    };
    main_bits = bits(w.num_centroids[1]);
    residual_bits = w.num_res_centroids[1] > 1 ? bits(w.num_res_centroids[1]) : 0;
    EXT_ENFORCE_INVALID(main_bits + residual_bits <= 32, "VPTQ index width exceeds INT32.");
    vectors = Ceil(n, Positive(w.vector_lens[1]));
    words = Ceil(Mul(columns, main_bits + residual_bits), 32);
    const auto dim = [](size_t v) {
      EXT_ENFORCE_INVALID(v <= size_t(std::numeric_limits<int64_t>::max()),
                          "VPTQ tensor dimension overflow.");
      return static_cast<int64_t>(v);
    };
    auto add = [&](const char *name, int32_t dt, Shape shape) {
      const size_t width = dt == TensorProto::INT32 || dt == TensorProto::FLOAT ? 4 : 2;
      size_t bytes = width;
      for (int64_t d : shape)
        bytes = Mul(bytes, Positive(d));
      EXT_ENFORCE_INVALID(total <= limit && bytes <= limit - total,
                          "VPTQ checkpoint exceeds max_bytes.");
      specs.push_back({name, dt, std::move(shape), total, bytes, width});
      total += bytes;
    };
    add("indices", TensorProto::INT32, {dim(groups), dim(vectors), dim(words)});
    add("centroids.weight", type,
        {dim(groups), dim(Mul(Positive(w.num_centroids[1]), Positive(w.vector_lens[1])))});
    if (residual_bits)
      add("res_centroids.weight", type,
          {dim(groups), dim(Mul(Positive(w.num_res_centroids[1]), Positive(w.vector_lens[1])))});
    if (outlier) {
      add("outlier_centroids.weight", type,
          {1, dim(Mul(Positive(w.num_centroids[0]), Positive(w.vector_lens[0])))});
      add("outlier_indices", TensorProto::INT32,
          {1, dim(Ceil(n, Positive(w.vector_lens[0]))), dim(outliers)});
    }
    if (w.enable_norm) {
      add("weight_scale", type, {dim(w.norm_dim == 0 ? k : n)});
      add("weight_bias", type, {dim(w.norm_dim == 0 ? k : n)});
    }
    if (w.enable_perm)
      add("perm", TensorProto::INT32, {dim(k)});
    if (bias)
      add("bias", type, {dim(n)});
  }
  const Spec &Get(const char *name) const {
    for (const auto &s : specs)
      if (s.name == name)
        return s;
    EXT_THROW_INVALID("VPTQ missing tensor ", name, ".");
  }
};

StructTypeProto Schema(const VptqWeights &w, const Layout &g, bool bias) {
  StructTypeProto result;
  result.set_name(kName);
  auto constant = [&](const char *name, int32_t dtype) {
    auto *field = result.mutable_structure()->add_field();
    field->set_name(name);
    field->mutable_constant()->set_data_type(dtype);
    return field->mutable_constant();
  };
  auto integer = [&](const char *name, int64_t n) {
    constant(name, TensorProto::INT64)->add_int64_data(n);
  };
  auto text = [&](const char *name, const std::string &v) {
    auto *t = constant(name, TensorProto::UINT8);
    t->add_dims(v.size());
    t->set_raw_data(v);
  };
  auto real = [&](const char *name, double v) {
    auto *t = constant(name, TensorProto::DOUBLE);
    uint64_t raw = std::bit_cast<uint64_t>(v);
    std::string bytes(8, '\0');
    for (size_t i = 0; i < 8; ++i)
      bytes[i] = static_cast<char>(raw >> (8 * i));
    t->set_raw_data(bytes);
  };
  text("gptqmodel_revision", w.upstream_revision);
  integer("in_features", w.in_features);
  integer("out_features", w.out_features);
  integer("bits", w.bits);
  integer("outlier_vector_len", w.vector_lens[0]);
  integer("vector_len", w.vector_lens[1]);
  integer("outlier_num_centroids", w.num_centroids[0]);
  integer("num_centroids", w.num_centroids[1]);
  integer("outlier_num_res_centroids", w.num_res_centroids[0]);
  integer("num_res_centroids", w.num_res_centroids[1]);
  real("npercent", w.npercent);
  integer("group_size", w.group_size);
  integer("group_num", w.group_num);
  integer("enable_norm", w.enable_norm);
  integer("norm_dim", w.norm_dim);
  integer("enable_perm", w.enable_perm);
  text("kmeans_mode", w.kmeans_mode);
  integer("kmeans_iters", w.kmeans_iters);
  real("kmeans_tol", w.kmeans_tol);
  integer("kmeans_seed", w.kmeans_seed);
  integer("block_size", w.block_size);
  real("percdamp", w.percdamp);
  integer("dtype", g.dtype);
  integer("has_bias", bias);
  integer("packed", 1);
  integer("pack_dtype", TensorProto::INT32);
  for (const auto &s : g.specs) {
    auto *f = result.mutable_structure()->add_field();
    f->set_name(s.name);
    auto *t = f->mutable_type()->mutable_tensor_type();
    t->set_elem_type(s.dtype);
    t->mutable_shape();
    for (int64_t d : s.shape)
      t->mutable_shape()->add_dim()->set_dim_value(d);
  }
  return result;
}

uint32_t Index(const uint8_t *data, const Layout &g, size_t group, size_t row, size_t column) {
  const size_t bit = column * (g.main_bits + g.residual_bits);
  const size_t base = (group * g.vectors + row) * g.words * 4;
  uint64_t raw = Read(data + base + (bit / 32) * 4, 4);
  if (bit % 32 + g.main_bits + g.residual_bits > 32)
    raw |= uint64_t(Read(data + base + (bit / 32 + 1) * 4, 4)) << 32;
  const uint64_t mask = (uint64_t{1} << (g.main_bits + g.residual_bits)) - 1;
  return static_cast<uint32_t>((raw >> (bit % 32)) & mask);
}
void CheckData(const VptqWeights &w, const Layout &g, const uint8_t *data) {
  const uint32_t mask = static_cast<uint32_t>((uint64_t{1} << g.main_bits) - 1);
  for (size_t group = 0; group < g.groups; ++group)
    for (size_t row = 0; row < g.vectors; ++row)
      for (size_t col = 0; col < g.columns; ++col) {
        const uint32_t index = Index(data, g, group, row, col);
        EXT_ENFORCE_INVALID(
            (index & mask) < uint64_t(w.num_centroids[1]) &&
                (!g.residual_bits || (index >> g.main_bits) < uint64_t(w.num_res_centroids[1])),
            "VPTQ index exceeds codebook.");
      }
  if (g.outliers) {
    const auto &s = g.Get("outlier_indices");
    for (size_t i = 0; i < s.bytes; i += 4) {
      const uint32_t index = Read(data + s.offset + i, 4);
      EXT_ENFORCE_INVALID(index <= uint32_t(std::numeric_limits<int32_t>::max()) &&
                              index < uint64_t(w.num_centroids[0]),
                          "VPTQ outlier index exceeds codebook.");
    }
  }
  if (w.enable_perm) {
    const auto &p = g.Get("perm");
    // The logical FLOAT bound also bounds the permutation-validation workspace.
    std::vector<uint8_t> seen(g.k, 0);
    for (size_t i = 0; i < g.k; ++i) {
      const uint32_t index = Read(data + p.offset + i * 4, 4);
      EXT_ENFORCE_INVALID(index <= uint32_t(std::numeric_limits<int32_t>::max()) && index < g.k &&
                              !seen[index],
                          "VPTQ invalid permutation.");
      seen[index] = 1;
    }
  }
}

struct Decoded {
  VptqWeights weights;
  Layout layout;
};
Decoded Validate(const EncodedValueProto &v, size_t limit, const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(v.has_struct_type() && !v.has_parameter_ref() && v.has_logical_type() &&
                          v.logical_type().has_tensor_type() && v.has_raw_data() &&
                          v.raw_data().size() <= limit,
                      "VPTQ requires a loaded independent encoded tensor.");
  const auto &type = catalogue.Resolve(v.struct_type());
  EXT_ENFORCE_INVALID(type.name().value() == kName && type.has_structure() &&
                          type.structure().field_size() >= kMetadata &&
                          type.structure().field_size() <= kMetadata + 9,
                      "VPTQ unsupported schema/version.");
  auto field = [&](int i) -> const TensorProto & {
    EXT_ENFORCE_INVALID(type.structure().field(i).has_constant(), "VPTQ missing metadata.");
    return type.structure().field(i).constant();
  };
  auto integer = [&](int i) {
    const auto &f = field(i);
    EXT_ENFORCE_INVALID(f.data_type() == TensorProto::INT64 && f.int64_data_size() == 1,
                        "VPTQ invalid integer metadata.");
    return f.int64_data(0);
  };
  auto boolean = [&](int i) {
    const int64_t value = integer(i);
    EXT_ENFORCE_INVALID(value == 0 || value == 1, "VPTQ invalid boolean metadata.");
    return value != 0;
  };
  auto text = [&](int i) {
    const auto &f = field(i);
    EXT_ENFORCE_INVALID(f.raw_data().size() <= 4096, "VPTQ excessive text metadata.");
    return f.raw_data();
  };
  auto real = [&](int i) {
    const auto &f = field(i);
    EXT_ENFORCE_INVALID(f.data_type() == TensorProto::DOUBLE && f.raw_data().size() == 8,
                        "VPTQ invalid floating metadata.");
    const auto *p = reinterpret_cast<const uint8_t *>(f.raw_data().data());
    return std::bit_cast<double>(uint64_t(Read(p, 4)) | (uint64_t(Read(p + 4, 4)) << 32));
  };
  VptqWeights w;
  w.upstream_revision = text(0);
  w.in_features = integer(1);
  w.out_features = integer(2);
  w.bits = integer(3);
  w.vector_lens = {integer(4), integer(5)};
  w.num_centroids = {integer(6), integer(7)};
  w.num_res_centroids = {integer(8), integer(9)};
  w.npercent = real(10);
  w.group_size = integer(11);
  w.group_num = integer(12);
  w.enable_norm = boolean(13);
  w.norm_dim = integer(14);
  w.enable_perm = boolean(15);
  w.kmeans_mode = text(16);
  w.kmeans_iters = integer(17);
  w.kmeans_tol = real(18);
  w.kmeans_seed = integer(19);
  w.block_size = integer(20);
  w.percdamp = real(21);
  const int64_t dtype = integer(22);
  EXT_ENFORCE_INVALID(dtype == TensorProto::FLOAT || dtype == TensorProto::FLOAT16 ||
                          dtype == TensorProto::BFLOAT16,
                      "VPTQ invalid dtype metadata.");
  const bool bias = boolean(23);
  Layout g(w, static_cast<int32_t>(dtype), bias, limit);
  auto expected = Schema(w, g, bias);
  if (type.has_type_id())
    expected.set_type_id(type.type_id());
  std::string difference;
  EXT_ENFORCE_INVALID(type.Equals(expected, &difference), "VPTQ noncanonical schema: ", difference);
  const auto &logical = v.logical_type().tensor_type();
  EXT_ENFORCE_INVALID(
      logical.elem_type() == TensorProto::FLOAT && logical.has_shape() &&
          logical.shape().dim_size() == 2 && logical.shape().dim(0).has_dim_value() &&
          logical.shape().dim(1).has_dim_value() &&
          logical.shape().dim(0).dim_value() == w.out_features &&
          logical.shape().dim(1).dim_value() == w.in_features && v.raw_data().size() == g.total,
      "VPTQ invalid logical shape or payload extent.");
  const auto storage = catalogue.ValidateEncodedValue(v);
  EXT_ENFORCE_INVALID(!storage.external && storage.record_count == 1,
                      "VPTQ requires one inline loaded record.");
  CheckData(w, g, reinterpret_cast<const uint8_t *>(v.raw_data().data()));
  return {std::move(w), std::move(g)};
}
float Load(const uint8_t *p, int32_t type, size_t i) {
  if (type == TensorProto::FLOAT)
    return std::bit_cast<float>(Read(p + i * 4, 4));
  const uint16_t bits = static_cast<uint16_t>(Read(p + i * 2, 2));
  return type == TensorProto::FLOAT16 ? Float16BitsToFloat(bits) : Bfloat16BitsToFloat(bits);
}
float Round(float value, int32_t type) {
  volatile float rounded = value;
  if (type == TensorProto::FLOAT16)
    return Float16BitsToFloat(FloatToFloat16Bits(rounded));
  if (type == TensorProto::BFLOAT16)
    return Bfloat16BitsToFloat(FloatToBfloat16Bits(rounded));
  return rounded;
}
} // namespace

EncodedValueProto ImportVptqWeights(const VptqWeights &w, size_t max_bytes) {
  const auto found = w.tensors.find("centroids.weight");
  EXT_ENFORCE_INVALID(found != w.tensors.end(), "VPTQ missing centroids.weight.");
  const bool bias = w.tensors.contains("bias");
  const Layout g(w, found->second.data_type, bias, max_bytes);
  EXT_ENFORCE_INVALID(w.tensors.size() == g.specs.size(), "VPTQ unexpected or missing tensors.");
  for (const auto &s : g.specs) {
    const auto it = w.tensors.find(s.name);
    EXT_ENFORCE_INVALID(it != w.tensors.end(), "VPTQ missing tensor ", s.name, ".");
    const auto &t = it->second;
    EXT_ENFORCE_INVALID(t.data_type == s.dtype && t.shape == s.shape && t.size_bytes() == s.bytes &&
                            t.bytes(),
                        "VPTQ invalid tensor type, shape or bytes: ", s.name, ".");
  }
  EncodedValueProto result;
  *result.mutable_struct_type() = Schema(w, g, bias);
  auto *logical = result.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(TensorProto::FLOAT);
  logical->mutable_shape()->add_dim()->set_dim_value(w.out_features);
  logical->mutable_shape()->add_dim()->set_dim_value(w.in_features);
  auto *raw = result.mutable_raw_data();
  raw->resize(g.total);
  for (const auto &s : g.specs)
    Copy(reinterpret_cast<uint8_t *>(raw->data()) + s.offset, w.tensors.at(s.name).bytes(), s.bytes,
         s.width);
  CheckData(w, g, reinterpret_cast<const uint8_t *>(raw->data()));
  StructTypeCatalogue{}.ValidateEncodedValue(result);
  return result;
}

VptqWeights ExportVptqWeights(const EncodedValueProto &v, size_t max_bytes,
                              const StructTypeCatalogue &catalogue) {
  auto decoded = Validate(v, max_bytes, catalogue);
  for (const auto &s : decoded.layout.specs) {
    std::vector<uint8_t> raw(s.bytes);
    Copy(raw.data(), reinterpret_cast<const uint8_t *>(v.raw_data().data()) + s.offset, s.bytes,
         s.width);
    decoded.weights.tensors.emplace(s.name, Tensor(s.name, s.dtype, s.shape, std::move(raw)));
  }
  return std::move(decoded.weights);
}

Tensor ReconstructVptqWeights(const EncodedValueProto &v, size_t max_bytes,
                              const StructTypeCatalogue &catalogue) {
  const auto decoded = Validate(v, max_bytes, catalogue);
  const auto &w = decoded.weights;
  const auto &g = decoded.layout;
  const auto *data = reinterpret_cast<const uint8_t *>(v.raw_data().data());
  Tensor result(std::string(v.name().value()), TensorProto::FLOAT, {w.out_features, w.in_features},
                std::vector<uint8_t>(g.count * 4));
  auto *output = reinterpret_cast<float *>(result.mutable_bytes());
  const auto *centroids = data + g.Get("centroids.weight").offset;
  const uint32_t mask = static_cast<uint32_t>((uint64_t{1} << g.main_bits) - 1);
  for (size_t row = 0; row < g.n; ++row)
    for (size_t col = 0; col < g.k; ++col) {
      float value;
      if (col < g.outliers) {
        const size_t len = static_cast<size_t>(w.vector_lens[0]);
        const size_t index =
            Read(data + g.Get("outlier_indices").offset + ((row / len) * g.outliers + col) * 4, 4);
        value =
            Load(data + g.Get("outlier_centroids.weight").offset, g.dtype, index * len + row % len);
      } else {
        const size_t group = (col - g.outliers) / g.columns;
        const size_t column = (col - g.outliers) % g.columns;
        const size_t len = static_cast<size_t>(w.vector_lens[1]);
        const uint32_t index = Index(data, g, group, row / len, column);
        value = Load(centroids, g.dtype,
                     (group * size_t(w.num_centroids[1]) + (index & mask)) * len + row % len);
        if (g.residual_bits)
          value = Round(
              value + Load(data + g.Get("res_centroids.weight").offset, g.dtype,
                           (group * size_t(w.num_res_centroids[1]) + (index >> g.main_bits)) * len +
                               row % len),
              g.dtype);
      }
      const size_t destination =
          w.enable_perm ? Read(data + g.Get("perm").offset + col * 4, 4) : col;
      if (w.enable_norm) {
        const size_t index = w.norm_dim == 0 ? destination : row;
        value = Round(value * Load(data + g.Get("weight_scale").offset, g.dtype, index), g.dtype);
        value = Round(value + Load(data + g.Get("weight_bias").offset, g.dtype, index), g.dtype);
      }
      output[row * g.k + destination] = value;
    }
  return result;
}
} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

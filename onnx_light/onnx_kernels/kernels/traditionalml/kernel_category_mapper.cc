// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

namespace {

template <typename InT, typename OutT>
void ValidateInputs(const Tensor &x, const std::vector<std::string> &cats_strings,
                    const std::vector<int64_t> &cats_int64s) {
  EXT_ENFORCE_INVALID(
      cats_strings.size() == cats_int64s.size(),
      "kernel::CategoryMapper requires cats_strings and cats_int64s to have the same length.");
  if constexpr (std::is_same_v<InT, std::string>) {
    static_assert(std::is_same_v<OutT, int64_t>,
                  "kernel::CategoryMapper: string input must map to int64 output.");
    EXT_ENFORCE_INVALID(x.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::CategoryMapper input data_type does not match the requested "
                        "InT (expected STRING).");
    EXT_ENFORCE_INVALID(
        static_cast<int64_t>(x.string_data.size()) == x.element_count(),
        "kernel::CategoryMapper STRING input string_data size does not match its shape.");
  } else {
    static_assert(std::is_same_v<InT, int64_t>,
                  "kernel::CategoryMapper: only string or int64 inputs are supported.");
    static_assert(std::is_same_v<OutT, std::string>,
                  "kernel::CategoryMapper: int64 input must map to string output.");
    EXT_ENFORCE_INVALID(x.data_type == TensorElementType<InT>::value,
                        "kernel::CategoryMapper input data_type does not match the requested "
                        "InT (expected INT64).");
  }
}

// Computes y[i] from x[i] by linear search over the parallel keys/values arrays.
template <typename InT, typename OutT, typename FillOut>
void LookupAndFill(const Tensor &x, const std::vector<std::string> &cats_strings,
                   const std::vector<int64_t> &cats_int64s, OutT default_value, FillOut fill_out) {
  const int64_t n = x.element_count();
  const size_t k = cats_strings.size();
  if constexpr (std::is_same_v<InT, std::string>) {
    const std::vector<std::string> &px = x.AsStrings();
    for (int64_t i = 0; i < n; ++i) {
      OutT mapped = default_value;
      for (size_t j = 0; j < k; ++j) {
        if (px[static_cast<size_t>(i)] == cats_strings[j]) {
          mapped = static_cast<OutT>(cats_int64s[j]);
          break;
        }
      }
      fill_out(i, mapped);
    }
  } else {
    const InT *px = x.As<InT>();
    for (int64_t i = 0; i < n; ++i) {
      OutT mapped = default_value;
      for (size_t j = 0; j < k; ++j) {
        if (px[i] == cats_int64s[j]) {
          mapped = cats_strings[j];
          break;
        }
      }
      fill_out(i, mapped);
    }
  }
}

} // namespace

template <typename InT, typename OutT>
Tensor CategoryMapper::operator()(const Tensor &x, const std::vector<std::string> &cats_strings,
                                  const std::vector<int64_t> &cats_int64s,
                                  OutT default_value) const {
  ValidateInputs<InT, OutT>(x, cats_strings, cats_int64s);
  const int64_t n = x.element_count();
  if constexpr (std::is_same_v<OutT, std::string>) {
    std::vector<std::string> sd(static_cast<size_t>(n));
    LookupAndFill<InT, OutT>(
        x, cats_strings, cats_int64s, default_value,
        [&sd](int64_t i, const std::string &v) { sd[static_cast<size_t>(i)] = v; });
    return Tensor::MakeString("", x.shape, std::move(sd));
  } else {
    std::vector<uint8_t> bytes(static_cast<size_t>(n) * sizeof(OutT));
    Tensor out("", TensorElementType<OutT>::value, x.shape, std::move(bytes));
    OutT *po = reinterpret_cast<OutT *>(out.data.data());
    LookupAndFill<InT, OutT>(x, cats_strings, cats_int64s, default_value,
                             [po](int64_t i, OutT v) { po[i] = v; });
    return out;
  }
}

template <typename InT, typename OutT>
void CategoryMapper::operator()(const Tensor &x, const std::vector<std::string> &cats_strings,
                                const std::vector<int64_t> &cats_int64s, OutT default_value,
                                Tensor &output) const {
  ValidateInputs<InT, OutT>(x, cats_strings, cats_int64s);
  EXT_ENFORCE_INVALID(output.shape == x.shape,
                      "kernel::CategoryMapper preallocated output shape must match the input "
                      "shape.");
  if constexpr (std::is_same_v<OutT, std::string>) {
    EXT_ENFORCE_INVALID(output.data_type == static_cast<int32_t>(DataType::STRING),
                        "kernel::CategoryMapper preallocated output dtype must be STRING for "
                        "int64 → string mapping.");
    output.string_data.assign(static_cast<size_t>(x.element_count()), std::string());
    LookupAndFill<InT, OutT>(x, cats_strings, cats_int64s, default_value,
                             [&output](int64_t i, const std::string &v) {
                               output.string_data[static_cast<size_t>(i)] = v;
                             });
  } else {
    EXT_ENFORCE_INVALID(output.data_type == TensorElementType<OutT>::value,
                        "kernel::CategoryMapper preallocated output dtype must be INT64 for "
                        "string → int64 mapping.");
    EXT_ENFORCE_INVALID(output.data.size() == static_cast<size_t>(x.element_count()) * sizeof(OutT),
                        "kernel::CategoryMapper preallocated output buffer is incorrectly sized.");
    OutT *po = output.As<OutT>();
    LookupAndFill<InT, OutT>(x, cats_strings, cats_int64s, default_value,
                             [po](int64_t i, OutT v) { po[i] = v; });
  }
}

// Explicit instantiations for the two element-type combinations defined by the
// ``ai.onnx.ml::CategoryMapper`` schema.
template Tensor CategoryMapper::operator()<std::string, int64_t>(const Tensor &,
                                                                 const std::vector<std::string> &,
                                                                 const std::vector<int64_t> &,
                                                                 int64_t) const;
template void CategoryMapper::operator()<std::string, int64_t>(const Tensor &,
                                                               const std::vector<std::string> &,
                                                               const std::vector<int64_t> &,
                                                               int64_t, Tensor &) const;
template Tensor CategoryMapper::operator()<int64_t, std::string>(const Tensor &,
                                                                 const std::vector<std::string> &,
                                                                 const std::vector<int64_t> &,
                                                                 std::string) const;
template void CategoryMapper::operator()<int64_t, std::string>(const Tensor &,
                                                               const std::vector<std::string> &,
                                                               const std::vector<int64_t> &,
                                                               std::string, Tensor &) const;

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

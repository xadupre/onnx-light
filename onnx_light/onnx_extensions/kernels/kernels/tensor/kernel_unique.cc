// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_core/runtime/runtime_context.h"
#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {

namespace {

// Normalise ``axis`` into the range [0, rank) and throw if out of range.
int64_t NormaliseUniqueAxis(int64_t axis, int64_t rank) {
  if (axis < 0) {
    axis += rank;
  }
  EXT_ENFORCE_INVALID(axis >= 0 && axis < rank, "kernel::Unique: axis out of range.");
  return axis;
}

// Lexicographic compare of two byte buffers of identical length.
int CompareBytes(const uint8_t *a, const uint8_t *b, std::size_t n) { return std::memcmp(a, b, n); }

// Returns true if ``data`` of the given ``data_type`` admits a "natural"
// (numeric or lexicographic) ordering when ``sorted`` is requested. STRING and
// the common numeric types are supported. The numeric branch uses a typed
// comparison; otherwise the kernel falls back to byte-wise comparison which
// matches numeric ordering for unsigned/Boolean inputs but not for signed or
// floating-point types — those are handled by typed dispatch below.

template <typename T> int CompareTyped(const uint8_t *a, const uint8_t *b) {
  T va;
  T vb;
  std::memcpy(&va, a, sizeof(T));
  std::memcpy(&vb, b, sizeof(T));
  if (va < vb)
    return -1;
  if (va > vb)
    return 1;
  return 0;
}

int CompareElement(DataType dt, const uint8_t *a, const uint8_t *b, std::size_t elem_size) {
  switch (dt) {
  case DataType::FLOAT:
    return CompareTyped<float>(a, b);
  case DataType::DOUBLE:
    return CompareTyped<double>(a, b);
  case DataType::INT8:
    return CompareTyped<int8_t>(a, b);
  case DataType::INT16:
    return CompareTyped<int16_t>(a, b);
  case DataType::INT32:
    return CompareTyped<int32_t>(a, b);
  case DataType::INT64:
    return CompareTyped<int64_t>(a, b);
  case DataType::UINT8:
    return CompareTyped<uint8_t>(a, b);
  case DataType::UINT16:
    return CompareTyped<uint16_t>(a, b);
  case DataType::UINT32:
    return CompareTyped<uint32_t>(a, b);
  case DataType::UINT64:
    return CompareTyped<uint64_t>(a, b);
  case DataType::BOOL:
    return CompareTyped<uint8_t>(a, b);
  default:
    return CompareBytes(a, b, elem_size);
  }
}

// Compare two subtensors of equal byte size, dispatching to the element-wise
// typed comparator. Used for sorting subtensors along ``axis`` lexicographically.
int CompareSubtensor(DataType dt, const uint8_t *a, const uint8_t *b, std::size_t n_elems,
                     std::size_t elem_size) {
  for (std::size_t k = 0; k < n_elems; ++k) {
    const int c = CompareElement(dt, a + k * elem_size, b + k * elem_size, elem_size);
    if (c != 0) {
      return c;
    }
  }
  return 0;
}

int CompareString(const std::string &a, const std::string &b) {
  if (a < b)
    return -1;
  if (a > b)
    return 1;
  return 0;
}

int CompareStringSubtensor(const std::vector<std::string> &strings, std::size_t a_off,
                           std::size_t b_off, std::size_t n) {
  for (std::size_t k = 0; k < n; ++k) {
    const int c = CompareString(strings[a_off + k], strings[b_off + k]);
    if (c != 0)
      return c;
  }
  return 0;
}

// Allocates an INT64 buffer of ``n`` elements. The storage is acquired from
// ``allocator`` when non-null (so no scratch is allocated outside the runtime
// context) and falls back to inline ``std::vector`` storage otherwise.
Tensor MakeInt64Buffer(int64_t n, RawBufferAllocator *allocator) {
  return MakeOutputTensor(static_cast<int32_t>(DataType::INT64), {n},
                          static_cast<std::size_t>(n) * sizeof(int64_t), allocator);
}

// Common driver: given a list of ``count`` "items", a comparator (returning
// negative/zero/positive), build the unique groups (order of first
// occurrence). When ``sorted`` is true, the returned groups are reordered
// lexicographically by the canonical item. Every buffer — the returned outputs
// and the transient working buffers — is acquired from ``allocator`` (or inline
// storage when it is null).
struct UniqueGroups {
  // The "representative" item index in input order for every unique group.
  Tensor indices; // INT64, shape {n_unique}, values in [0, count)
  // For every input item, the index of its group in ``indices``.
  Tensor inverse_indices; // INT64, shape {count}
  // Count of items in every group (same order as ``indices``).
  Tensor counts; // INT64, shape {n_unique}
};

template <typename Cmp>
UniqueGroups ComputeUniqueGroups(int64_t count, const Cmp &cmp, bool sorted,
                                 RawBufferAllocator *allocator) {
  Tensor inverse = MakeInt64Buffer(count, allocator);
  int64_t *inv = (count > 0) ? inverse.As<int64_t>() : nullptr;

  // Working buffers with capacity ``count`` (the maximum possible group count).
  Tensor first_scratch = MakeInt64Buffer(count, allocator);
  Tensor counts_scratch = MakeInt64Buffer(count, allocator);
  int64_t *first = (count > 0) ? first_scratch.As<int64_t>() : nullptr;
  int64_t *cnts = (count > 0) ? counts_scratch.As<int64_t>() : nullptr;

  // Naive O(count * unique) grouping. Sufficient for reference kernel use.
  int64_t n_unique = 0;
  for (int64_t i = 0; i < count; ++i) {
    int64_t found = -1;
    for (int64_t g = 0; g < n_unique; ++g) {
      if (cmp(i, first[g]) == 0) {
        found = g;
        break;
      }
    }
    if (found < 0) {
      inv[i] = n_unique;
      first[n_unique] = i;
      cnts[n_unique] = 1;
      ++n_unique;
    } else {
      inv[i] = found;
      cnts[found] += 1;
    }
  }

  UniqueGroups out;
  out.indices = MakeInt64Buffer(n_unique, allocator);
  out.counts = MakeInt64Buffer(n_unique, allocator);
  int64_t *out_first = (n_unique > 0) ? out.indices.As<int64_t>() : nullptr;
  int64_t *out_counts = (n_unique > 0) ? out.counts.As<int64_t>() : nullptr;

  if (sorted && n_unique > 1) {
    // Build a permutation that sorts the groups by their representative item,
    // then remap ``inverse`` accordingly.
    Tensor perm_scratch = MakeInt64Buffer(n_unique, allocator);
    Tensor remap_scratch = MakeInt64Buffer(n_unique, allocator);
    int64_t *perm = perm_scratch.As<int64_t>();
    int64_t *remap = remap_scratch.As<int64_t>();
    for (int64_t i = 0; i < n_unique; ++i) {
      perm[i] = i;
    }
    std::sort(perm, perm + n_unique,
              [&](int64_t a, int64_t b) { return cmp(first[a], first[b]) < 0; });
    for (int64_t new_idx = 0; new_idx < n_unique; ++new_idx) {
      const int64_t old_idx = perm[new_idx];
      remap[old_idx] = new_idx;
      out_first[new_idx] = first[old_idx];
      out_counts[new_idx] = cnts[old_idx];
    }
    for (int64_t i = 0; i < count; ++i) {
      inv[i] = remap[inv[i]];
    }
  } else {
    for (int64_t g = 0; g < n_unique; ++g) {
      out_first[g] = first[g];
      out_counts[g] = cnts[g];
    }
  }

  out.inverse_indices = std::move(inverse);
  return out;
}

} // namespace

Unique::Outputs Unique::operator()(const Tensor &x) const { return (*this)(x, Attributes{}); }

Unique::Outputs Unique::operator()(const Tensor &x, const Attributes &attrs,
                                   RuntimeContext *rt) const {
  const DataType dt = static_cast<DataType>(x.data_type);
  const bool is_string = (dt == DataType::STRING);
  if (!is_string) {
    // Validate dtype: support the same numeric and BOOL types as NonZero.
    switch (dt) {
    case DataType::FLOAT:
    case DataType::DOUBLE:
    case DataType::INT8:
    case DataType::INT16:
    case DataType::INT32:
    case DataType::INT64:
    case DataType::UINT8:
    case DataType::UINT16:
    case DataType::UINT32:
    case DataType::UINT64:
    case DataType::BOOL:
      break;
    default:
      EXT_ENFORCE_INVALID(false, "kernel::Unique: unsupported input dtype.");
    }
  }

  const std::size_t elem_size = is_string ? 0u : ElementSize(x.data_type);
  const onnx_kernels::Shape &shape = x.shape;
  const int64_t rank = static_cast<int64_t>(shape.size());

  // Resolve axis.
  std::optional<int64_t> axis = attrs.axis;
  if (axis.has_value()) {
    EXT_ENFORCE_INVALID(rank >= 1, "kernel::Unique: axis requires rank >= 1.");
    axis = NormaliseUniqueAxis(*axis, rank);
  }

  // ``count`` is either the flattened element count (no axis) or the size of
  // the axis dimension. ``inner_block_bytes`` is the size in bytes of a single
  // "item" along the chosen dimension (1 element when flattened; otherwise the
  // product of the dimensions other than ``axis`` times ``elem_size``).
  int64_t count = 0;
  int64_t outer = 1;
  int64_t axis_dim = 0;
  int64_t inner_elems = 1;
  if (!axis.has_value()) {
    count = x.element_count();
  } else {
    axis_dim = shape[static_cast<std::size_t>(*axis)];
    count = axis_dim;
    for (int64_t d = 0; d < *axis; ++d) {
      outer *= shape[static_cast<std::size_t>(d)];
    }
    for (int64_t d = *axis + 1; d < rank; ++d) {
      inner_elems *= shape[static_cast<std::size_t>(d)];
    }
  }

  // Build the comparator and the gather helper for both modes (flattened /
  // axis), for STRING and non-STRING tensors.
  UniqueGroups groups;
  if (!axis.has_value()) {
    if (is_string) {
      const auto &strs = x.string_data;
      groups = ComputeUniqueGroups(
          count,
          [&](int64_t a, int64_t b) {
            return CompareString(strs[static_cast<std::size_t>(a)],
                                 strs[static_cast<std::size_t>(b)]);
          },
          attrs.sorted, ctx_.allocator);
    } else {
      const uint8_t *base = x.bytes();
      groups = ComputeUniqueGroups(
          count,
          [&](int64_t a, int64_t b) {
            return CompareElement(dt, base + static_cast<std::size_t>(a) * elem_size,
                                  base + static_cast<std::size_t>(b) * elem_size, elem_size);
          },
          attrs.sorted, ctx_.allocator);
    }
  } else {
    // Axis mode: an "item" is a subtensor spanning all outer x inner positions
    // for a fixed axis index. Two items are equal iff every (outer, inner)
    // element is equal at the corresponding positions. For comparison we
    // walk the (outer, inner) plane.
    if (is_string) {
      const auto &strs = x.string_data;
      // Flat index helper: at (o, k, i) where k is the axis index, the linear
      // string-data offset is o * axis_dim * inner_elems + k * inner_elems + i.
      auto cmp = [&](int64_t ka, int64_t kb) -> int {
        for (int64_t o = 0; o < outer; ++o) {
          const std::size_t base_a =
              static_cast<std::size_t>(o * axis_dim * inner_elems + ka * inner_elems);
          const std::size_t base_b =
              static_cast<std::size_t>(o * axis_dim * inner_elems + kb * inner_elems);
          const int c =
              CompareStringSubtensor(strs, base_a, base_b, static_cast<std::size_t>(inner_elems));
          if (c != 0)
            return c;
        }
        return 0;
      };
      groups = ComputeUniqueGroups(count, cmp, attrs.sorted, ctx_.allocator);
    } else {
      const uint8_t *base = x.bytes();
      const std::size_t block_bytes = static_cast<std::size_t>(inner_elems) * elem_size;
      auto cmp = [&](int64_t ka, int64_t kb) -> int {
        for (int64_t o = 0; o < outer; ++o) {
          const std::size_t off_a =
              (static_cast<std::size_t>(o) * static_cast<std::size_t>(axis_dim) +
               static_cast<std::size_t>(ka)) *
              block_bytes;
          const std::size_t off_b =
              (static_cast<std::size_t>(o) * static_cast<std::size_t>(axis_dim) +
               static_cast<std::size_t>(kb)) *
              block_bytes;
          const int c = CompareSubtensor(dt, base + off_a, base + off_b,
                                         static_cast<std::size_t>(inner_elems), elem_size);
          if (c != 0)
            return c;
        }
        return 0;
      };
      groups = ComputeUniqueGroups(count, cmp, attrs.sorted, ctx_.allocator);
    }
  }

  const int64_t n_unique = groups.indices.element_count();
  const int64_t *first_occ = (n_unique > 0) ? groups.indices.As<int64_t>() : nullptr;

  // Build Y.
  RawBufferAllocator *allocator = rt ? rt->allocator() : nullptr;
  Outputs out;
  if (!axis.has_value()) {
    // Y is 1-D of length n_unique.
    if (is_string) {
      std::vector<std::string> y_strs;
      y_strs.reserve(static_cast<std::size_t>(n_unique));
      for (int64_t g = 0; g < n_unique; ++g) {
        y_strs.push_back(x.string_data[static_cast<std::size_t>(first_occ[g])]);
      }
      out.y = Tensor::FromStrings("", {n_unique}, y_strs);
    } else {
      const size_t y_n_bytes = static_cast<std::size_t>(n_unique) * elem_size;
      Tensor y = MakeOutputTensor(x.data_type, {n_unique}, y_n_bytes, allocator);
      for (int64_t g = 0; g < n_unique; ++g) {
        const std::size_t src_off = static_cast<std::size_t>(first_occ[g]) * elem_size;
        const std::size_t dst_off = static_cast<std::size_t>(g) * elem_size;
        std::memcpy(y.mutable_bytes() + dst_off, x.bytes() + src_off, elem_size);
      }
      out.y = std::move(y);
    }
  } else {
    // Y has the same shape as X, with axis_dim replaced by n_unique.
    onnx_kernels::Shape y_shape = shape;
    y_shape[static_cast<std::size_t>(*axis)] = n_unique;
    int64_t y_total = 1;
    for (int64_t d : y_shape) {
      y_total *= d;
    }
    if (is_string) {
      std::vector<std::string> y_strs(static_cast<std::size_t>(y_total));
      for (int64_t o = 0; o < outer; ++o) {
        for (int64_t g = 0; g < n_unique; ++g) {
          const int64_t k = first_occ[g];
          for (int64_t i = 0; i < inner_elems; ++i) {
            const std::size_t src =
                static_cast<std::size_t>(o * axis_dim * inner_elems + k * inner_elems + i);
            const std::size_t dst =
                static_cast<std::size_t>(o * n_unique * inner_elems + g * inner_elems + i);
            y_strs[dst] = x.string_data[src];
          }
        }
      }
      out.y = Tensor::FromStrings("", y_shape, y_strs);
    } else {
      const std::size_t block_bytes = static_cast<std::size_t>(inner_elems) * elem_size;
      const size_t y_n_bytes = static_cast<std::size_t>(y_total) * elem_size;
      Tensor y = MakeOutputTensor(x.data_type, y_shape, y_n_bytes, allocator);
      for (int64_t o = 0; o < outer; ++o) {
        for (int64_t g = 0; g < n_unique; ++g) {
          const int64_t k = first_occ[g];
          const std::size_t src_off =
              (static_cast<std::size_t>(o) * static_cast<std::size_t>(axis_dim) +
               static_cast<std::size_t>(k)) *
              block_bytes;
          const std::size_t dst_off =
              (static_cast<std::size_t>(o) * static_cast<std::size_t>(n_unique) +
               static_cast<std::size_t>(g)) *
              block_bytes;
          std::memcpy(y.mutable_bytes() + dst_off, x.bytes() + src_off, block_bytes);
        }
      }
      out.y = std::move(y);
    }
  }

  out.indices = std::move(groups.indices);
  out.inverse_indices = std::move(groups.inverse_indices);
  out.counts = std::move(groups.counts);
  return out;
}

void Unique::Run(RuntimeContext &rt) {
  const NodeProto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputRange(node, 1, 4);
  const Tensor &x = GetInput(node, 0, rt.tensors());
  onnx_kernels::kernel::Unique::Attributes attrs;
  attrs.sorted = GetAttributeIntOrDefault(node, "sorted", 1) != 0;
  const AttributeProto *axis_attr = FindAttribute(node, "axis");
  if (axis_attr != nullptr) {
    attrs.axis = axis_attr->i();
  }
  onnx_kernels::kernel::Unique k(rt.kernel_ctx());
  auto out = k(x, attrs);
  SetOutput(node, 0, std::move(out.y), rt);
  if (node.output_size() >= 2) {
    SetOutput(node, 1, std::move(out.indices), rt);
  }
  if (node.output_size() >= 3) {
    SetOutput(node, 2, std::move(out.inverse_indices), rt);
  }
  if (node.output_size() >= 4) {
    SetOutput(node, 3, std::move(out.counts), rt);
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

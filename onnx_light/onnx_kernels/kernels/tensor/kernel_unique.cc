// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

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

// Common driver: given a list of ``count`` "items", a comparator (returning
// negative/zero/positive), build the unique groups (order of first
// occurrence). When ``sorted`` is true, the returned groups are reordered
// lexicographically by the canonical item.
struct UniqueGroups {
  // The "representative" item index in input order for every unique group.
  std::vector<int64_t> first_occurrence; // indices into [0, count)
  // For every input item, the index of its group in ``first_occurrence``.
  std::vector<int64_t> inverse; // size == count
  // Count of items in every group (same order as ``first_occurrence``).
  std::vector<int64_t> counts;
};

template <typename Cmp>
UniqueGroups ComputeUniqueGroups(int64_t count, const Cmp &cmp, bool sorted) {
  UniqueGroups out;
  out.inverse.assign(static_cast<std::size_t>(count), 0);
  // Naive O(count * unique) grouping. Sufficient for reference kernel use.
  for (int64_t i = 0; i < count; ++i) {
    int64_t found = -1;
    for (std::size_t g = 0; g < out.first_occurrence.size(); ++g) {
      if (cmp(i, out.first_occurrence[g]) == 0) {
        found = static_cast<int64_t>(g);
        break;
      }
    }
    if (found < 0) {
      out.inverse[static_cast<std::size_t>(i)] = static_cast<int64_t>(out.first_occurrence.size());
      out.first_occurrence.push_back(i);
      out.counts.push_back(1);
    } else {
      out.inverse[static_cast<std::size_t>(i)] = found;
      out.counts[static_cast<std::size_t>(found)] += 1;
    }
  }

  if (sorted && out.first_occurrence.size() > 1) {
    // Build a permutation that sorts the groups by their representative item.
    std::vector<int64_t> perm(out.first_occurrence.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
      perm[i] = static_cast<int64_t>(i);
    }
    std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
      return cmp(out.first_occurrence[static_cast<std::size_t>(a)],
                 out.first_occurrence[static_cast<std::size_t>(b)]) < 0;
    });

    // Apply the permutation to first_occurrence and counts, and remap
    // inverse[] accordingly.
    std::vector<int64_t> remap(perm.size());
    for (std::size_t new_idx = 0; new_idx < perm.size(); ++new_idx) {
      remap[static_cast<std::size_t>(perm[new_idx])] = static_cast<int64_t>(new_idx);
    }
    std::vector<int64_t> new_first(perm.size());
    std::vector<int64_t> new_counts(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
      new_first[i] = out.first_occurrence[static_cast<std::size_t>(perm[i])];
      new_counts[i] = out.counts[static_cast<std::size_t>(perm[i])];
    }
    out.first_occurrence = std::move(new_first);
    out.counts = std::move(new_counts);
    for (int64_t &v : out.inverse) {
      v = remap[static_cast<std::size_t>(v)];
    }
  }
  return out;
}

} // namespace

Unique::Outputs Unique::operator()(const Tensor &x) const { return (*this)(x, Attributes{}); }

Unique::Outputs Unique::operator()(const Tensor &x, const Attributes &attrs) const {
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
  const std::vector<int64_t> &shape = x.shape;
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
          attrs.sorted);
    } else {
      const uint8_t *base = x.data.data();
      groups = ComputeUniqueGroups(
          count,
          [&](int64_t a, int64_t b) {
            return CompareElement(dt, base + static_cast<std::size_t>(a) * elem_size,
                                  base + static_cast<std::size_t>(b) * elem_size, elem_size);
          },
          attrs.sorted);
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
      groups = ComputeUniqueGroups(count, cmp, attrs.sorted);
    } else {
      const uint8_t *base = x.data.data();
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
      groups = ComputeUniqueGroups(count, cmp, attrs.sorted);
    }
  }

  const int64_t n_unique = static_cast<int64_t>(groups.first_occurrence.size());

  // Build Y.
  Outputs out;
  if (!axis.has_value()) {
    // Y is 1-D of length n_unique.
    if (is_string) {
      std::vector<std::string> y_strs;
      y_strs.reserve(static_cast<std::size_t>(n_unique));
      for (int64_t idx : groups.first_occurrence) {
        y_strs.push_back(x.string_data[static_cast<std::size_t>(idx)]);
      }
      out.y = Tensor::FromStrings("", {n_unique}, y_strs);
    } else {
      Tensor y("", x.data_type, {n_unique},
               std::vector<uint8_t>(static_cast<std::size_t>(n_unique) * elem_size));
      for (int64_t g = 0; g < n_unique; ++g) {
        const std::size_t src_off =
            static_cast<std::size_t>(groups.first_occurrence[static_cast<std::size_t>(g)]) *
            elem_size;
        const std::size_t dst_off = static_cast<std::size_t>(g) * elem_size;
        std::memcpy(y.data.data() + dst_off, x.data.data() + src_off, elem_size);
      }
      out.y = std::move(y);
    }
  } else {
    // Y has the same shape as X, with axis_dim replaced by n_unique.
    std::vector<int64_t> y_shape = shape;
    y_shape[static_cast<std::size_t>(*axis)] = n_unique;
    int64_t y_total = 1;
    for (int64_t d : y_shape) {
      y_total *= d;
    }
    if (is_string) {
      std::vector<std::string> y_strs(static_cast<std::size_t>(y_total));
      for (int64_t o = 0; o < outer; ++o) {
        for (int64_t g = 0; g < n_unique; ++g) {
          const int64_t k = groups.first_occurrence[static_cast<std::size_t>(g)];
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
      Tensor y("", x.data_type, y_shape,
               std::vector<uint8_t>(static_cast<std::size_t>(y_total) * elem_size));
      for (int64_t o = 0; o < outer; ++o) {
        for (int64_t g = 0; g < n_unique; ++g) {
          const int64_t k = groups.first_occurrence[static_cast<std::size_t>(g)];
          const std::size_t src_off =
              (static_cast<std::size_t>(o) * static_cast<std::size_t>(axis_dim) +
               static_cast<std::size_t>(k)) *
              block_bytes;
          const std::size_t dst_off =
              (static_cast<std::size_t>(o) * static_cast<std::size_t>(n_unique) +
               static_cast<std::size_t>(g)) *
              block_bytes;
          std::memcpy(y.data.data() + dst_off, x.data.data() + src_off, block_bytes);
        }
      }
      out.y = std::move(y);
    }
  }

  out.indices = Tensor::FromInt64("", {n_unique}, groups.first_occurrence);
  out.inverse_indices = Tensor::FromInt64("", {count}, groups.inverse);
  out.counts = Tensor::FromInt64("", {n_unique}, groups.counts);
  return out;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

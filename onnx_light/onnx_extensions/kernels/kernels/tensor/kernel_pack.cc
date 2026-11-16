// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/kernels/kernel_run_helpers.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {
namespace {

struct PackingParameters {
  int64_t radix;
  int64_t group_size;
  uint64_t limit;
  int64_t width;
};

PackingParameters Parameters(std::optional<int64_t> bits, std::optional<int64_t> radix,
                             std::optional<int64_t> group_size) {
  if (bits) {
    EXT_ENFORCE_INVALID(!radix && !group_size,
                        "Pack/Unpack: specify either bits or both radix and group_size.");
    EXT_ENFORCE_INVALID(*bits >= 1 && *bits <= 8, "Pack/Unpack: bits must be between 1 and 8.");
    return {int64_t{1} << *bits, 1, uint64_t{1} << *bits, *bits};
  }
  EXT_ENFORCE_INVALID(radix && group_size,
                      "Pack/Unpack: specify either bits or both radix and group_size.");
  EXT_ENFORCE_INVALID(*radix >= 2 && *radix <= 256 && *group_size > 0,
                      "Pack/Unpack: radix must be in [2, 256] and group_size must be positive.");
  uint64_t limit = 1;
  for (int64_t i = 0; i < *group_size; ++i) {
    EXT_ENFORCE_INVALID(limit <= uint64_t(std::numeric_limits<int64_t>::max()) / *radix,
                        "Pack/Unpack: radix^group_size must fit in a positive INT64.");
    limit *= *radix;
  }
  return {*radix, *group_size, limit, static_cast<int64_t>(std::bit_width(limit - 1))};
}

int64_t Axis(const Tensor &x, int64_t axis) {
  const int64_t rank = static_cast<int64_t>(x.shape.size());
  EXT_ENFORCE_INVALID(rank > 0, "Pack/Unpack: X must have rank at least 1.");
  EXT_ENFORCE_INVALID(axis >= -rank && axis < rank, "Pack/Unpack: axis is out of range.");
  return axis < 0 ? axis + rank : axis;
}

size_t ByteSize(const onnx_kernels::Shape &shape, size_t item_size) {
  const uint64_t elements = shape.product(0, shape.size(), "Pack/Unpack");
  EXT_ENFORCE_INVALID(elements <= std::numeric_limits<size_t>::max() / item_size,
                      "Pack/Unpack: tensor byte size overflows.");
  return static_cast<size_t>(elements) * item_size;
}

int64_t Groups(int64_t count, int64_t group_size) {
  return count / group_size + (count % group_size != 0);
}

int64_t Words(int64_t groups, int64_t width, int64_t word_bits) {
  const int64_t quotient = groups / word_bits;
  const int64_t remainder_bits = (groups % word_bits) * width;
  const int64_t extra = remainder_bits / word_bits + (remainder_bits % word_bits != 0);
  EXT_ENFORCE_INVALID(quotient <= (std::numeric_limits<int64_t>::max() - extra) / width,
                      "Pack/Unpack: packed word count overflows INT64.");
  return quotient * width + extra;
}

std::pair<int64_t, int64_t> GroupPosition(int64_t group, int64_t width, int64_t word_bits) {
  const int64_t remainder_bits = (group % word_bits) * width;
  return {(group / word_bits) * width + remainder_bits / word_bits, remainder_bits % word_bits};
}

uint64_t ReadWord(const Tensor &x, int64_t offset, int64_t word_bits) {
  if (word_bits == 8)
    return x.bytes()[offset];
  uint32_t value;
  std::memcpy(&value, x.bytes() + static_cast<size_t>(offset) * sizeof(value), sizeof(value));
  return value;
}

void OrWord(Tensor &y, int64_t offset, int64_t word_bits, uint64_t value) {
  if (word_bits == 8) {
    y.mutable_bytes()[offset] |= static_cast<uint8_t>(value);
  } else {
    uint32_t word = static_cast<uint32_t>(ReadWord(y, offset, word_bits) | value);
    std::memcpy(y.mutable_bytes() + static_cast<size_t>(offset) * sizeof(word), &word,
                sizeof(word));
  }
}

std::optional<int64_t> OptionalInt(const NodeProto &node, const char *name) {
  const auto *attribute = FindAttribute(node, name);
  if (!attribute)
    return std::nullopt;
  EXT_ENFORCE_INVALID(attribute->type() == AttributeProto::INT,
                      "Pack/Unpack: packing attributes must be INT.");
  return attribute->i();
}

} // namespace

Tensor Pack::operator()(const Tensor &x, std::optional<int64_t> bits, std::optional<int64_t> radix,
                        std::optional<int64_t> group_size, int64_t axis, int64_t output_type,
                        RuntimeContext *rt) const {
  const auto p = Parameters(bits, radix, group_size);
  EXT_ENFORCE_INVALID(x.data_type == TensorProto::UINT8, "Pack: X must be a UINT8 tensor.");
  axis = Axis(x, axis);
  EXT_ENFORCE_INVALID(output_type == TensorProto::UINT8 || output_type == TensorProto::UINT32,
                      "Pack: output_type must be UINT8 (2) or UINT32 (12).");
  const size_t input_bytes = ByteSize(x.shape, 1);
  EXT_ENFORCE_INVALID(x.size_bytes() == input_bytes, "Pack: invalid input buffer size.");
  for (size_t i = 0; i < input_bytes; ++i)
    EXT_ENFORCE_INVALID(x.bytes()[i] < p.radix, "Pack: input codes must be representable in radix ",
                        p.radix, ".");
  const int64_t word_bits = output_type == TensorProto::UINT8 ? 8 : 32;
  const int64_t count = x.shape[axis];
  const int64_t groups = Groups(count, p.group_size);
  const int64_t words = Words(groups, p.width, word_bits);
  onnx_kernels::Shape shape = x.shape;
  shape[axis] = words;
  const size_t output_bytes = ByteSize(shape, word_bits / 8);
  Tensor y = rt ? rt->MakeOutputTensor(0, output_type, shape, output_bytes)
                : MakeOutputTensor(output_type, shape, output_bytes, ctx_.allocator);
  if (output_bytes == 0)
    return y;
  std::memset(y.mutable_bytes(), 0, output_bytes);
  const int64_t outer = x.shape.product(0, axis, "Pack");
  const int64_t inner = x.shape.product(axis + 1, x.shape.size(), "Pack");
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      for (int64_t g = 0; g < groups; ++g) {
        uint64_t code = 0, power = 1;
        const int64_t start = g * p.group_size;
        for (int64_t j = 0; j < std::min(p.group_size, count - start); ++j) {
          code += x.bytes()[(o * count + start + j) * inner + i] * power;
          power *= p.radix;
        }
        auto [word, shift] = GroupPosition(g, p.width, word_bits);
        for (int64_t remaining = p.width; remaining > 0; shift = 0, ++word) {
          const int64_t take = std::min(remaining, word_bits - shift);
          const uint64_t mask = (uint64_t{1} << take) - 1;
          OrWord(y, (o * words + word) * inner + i, word_bits, (code & mask) << shift);
          code >>= take;
          remaining -= take;
        }
      }
    }
  }
  return y;
}

Tensor Unpack::operator()(const Tensor &x, const Tensor &count, std::optional<int64_t> bits,
                          std::optional<int64_t> radix, std::optional<int64_t> group_size,
                          int64_t axis, RuntimeContext *rt) const {
  const auto p = Parameters(bits, radix, group_size);
  EXT_ENFORCE_INVALID(x.data_type == TensorProto::UINT8 || x.data_type == TensorProto::UINT32,
                      "Unpack: X must be a UINT8 or UINT32 tensor.");
  axis = Axis(x, axis);
  EXT_ENFORCE_INVALID(count.data_type == TensorProto::INT64 && count.shape.empty() &&
                          count.size_bytes() == sizeof(int64_t),
                      "Unpack: count must be a nonnegative scalar INT64 tensor.");
  int64_t n;
  std::memcpy(&n, count.bytes(), sizeof(n));
  EXT_ENFORCE_INVALID(n >= 0, "Unpack: count must be a nonnegative scalar INT64 tensor.");
  const int64_t word_bits = x.data_type == TensorProto::UINT8 ? 8 : 32;
  EXT_ENFORCE_INVALID(x.size_bytes() == ByteSize(x.shape, word_bits / 8),
                      "Unpack: invalid input buffer size.");
  const int64_t groups = Groups(n, p.group_size);
  const int64_t words = Words(groups, p.width, word_bits);
  EXT_ENFORCE_INVALID(x.shape[axis] == words,
                      "Unpack: packed axis dimension must equal the required word count.");
  onnx_kernels::Shape shape = x.shape;
  shape[axis] = n;
  const size_t output_bytes = ByteSize(shape, 1);
  Tensor y = rt ? rt->MakeOutputTensor(0, TensorProto::UINT8, shape, output_bytes)
                : MakeOutputTensor(TensorProto::UINT8, shape, output_bytes, ctx_.allocator);
  if (output_bytes == 0)
    return y;
  const int64_t outer = shape.product(0, axis, "Unpack");
  const int64_t inner = shape.product(axis + 1, shape.size(), "Unpack");
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      for (int64_t g = 0; g < groups; ++g) {
        uint64_t code = 0;
        auto [word, shift] = GroupPosition(g, p.width, word_bits);
        int64_t written = 0;
        for (int64_t remaining = p.width; remaining > 0; shift = 0, ++word) {
          const int64_t take = std::min(remaining, word_bits - shift);
          const uint64_t mask = (uint64_t{1} << take) - 1;
          code |= ((ReadWord(x, (o * words + word) * inner + i, word_bits) >> shift) & mask)
                  << written;
          written += take;
          remaining -= take;
        }
        EXT_ENFORCE_INVALID(code < p.limit,
                            "Unpack: packed group index must be less than radix^group_size.");
        const int64_t start = g * p.group_size;
        // Digits beyond count and high padding bits are intentionally ignored.
        for (int64_t j = 0; j < std::min(p.group_size, n - start); ++j) {
          y.mutable_bytes()[(o * n + start + j) * inner + i] = code % p.radix;
          code /= p.radix;
        }
      }
    }
  }
  return y;
}

void Pack::Run(RuntimeContext &rt) {
  const auto &node = *node_;
  RequireInputCount(node, 1);
  RequireOutputCount(node, 1);
  SetOutput(node, 0,
            (*this)(GetInput(node, 0, rt), OptionalInt(node, "bits"), OptionalInt(node, "radix"),
                    OptionalInt(node, "group_size"), OptionalInt(node, "axis").value_or(-1),
                    OptionalInt(node, "output_type").value_or(TensorProto::UINT8), &rt),
            rt);
}

void Unpack::Run(RuntimeContext &rt) {
  const auto &node = *node_;
  RequireInputCount(node, 2);
  RequireOutputCount(node, 1);
  SetOutput(node, 0,
            (*this)(GetInput(node, 0, rt), GetInput(node, 1, rt), OptionalInt(node, "bits"),
                    OptionalInt(node, "radix"), OptionalInt(node, "group_size"),
                    OptionalInt(node, "axis").value_or(-1), &rt),
            rt);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

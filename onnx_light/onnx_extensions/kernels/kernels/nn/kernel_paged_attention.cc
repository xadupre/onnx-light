// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/node_helpers.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_set>

namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel {
namespace {

size_t CheckedBytes(int64_t rows, int64_t width, size_t item = sizeof(float)) {
  EXT_ENFORCE_INVALID(
      rows >= 0 && width > 0 &&
          static_cast<uint64_t>(width) <= std::numeric_limits<size_t>::max() / item &&
          static_cast<uint64_t>(rows) <= std::numeric_limits<size_t>::max() / item / width &&
          (rows == 0 || width <= std::numeric_limits<int64_t>::max() / rows),
      "PagedAttention: tensor extent overflow.");
  return static_cast<size_t>(rows) * static_cast<size_t>(width) * item;
}

void AddBytes(uint64_t &total, size_t bytes) {
  EXT_ENFORCE_INVALID(bytes <= std::numeric_limits<uint64_t>::max() - total,
                      "PagedAttention: statistics overflow.");
  total += bytes;
}

void CheckTensor(const Tensor &tensor) {
  EXT_ENFORCE_INVALID(tensor.data_type == DataType::FLOAT && tensor.shape.size() == 4 &&
                          tensor.shape[0] == 1 && tensor.shape[1] == 1,
                      "PagedAttention: only FLOAT [1,1,L,D] tensors are supported.");
  const size_t bytes = CheckedBytes(tensor.shape[2], tensor.shape[3]);
  EXT_ENFORCE_INVALID(tensor.size_bytes() == bytes && (bytes == 0 || tensor.bytes() != nullptr),
                      "PagedAttention: invalid tensor payload extent.");
}

std::pair<int, int> CodeRange(int32_t type) {
  switch (type) {
  case DataType::INT8:
    return {-128, 127};
  case DataType::UINT8:
    return {0, 255};
  case DataType::INT4:
    return {-8, 7};
  case DataType::UINT4:
    return {0, 15};
  default:
    EXT_THROW_INVALID("PagedAttention: unsupported storage type.");
  }
}

bool FourBit(int32_t type) { return type == DataType::INT4 || type == DataType::UINT4; }

void CheckFormat(const PagedAttention::Format &format) {
  EXT_ENFORCE_INVALID(std::isfinite(format.scale) && format.scale > 0,
                      "PagedAttention: scale must be finite and positive.");
  if (format.storage_type == DataType::FLOAT) {
    EXT_ENFORCE_INVALID(format.scale == 1 && format.zero_point == 0,
                        "PagedAttention: dense format requires identity parameters.");
    return;
  }
  const auto [low, high] = CodeRange(format.storage_type);
  EXT_ENFORCE_INVALID(format.zero_point >= low && format.zero_point <= high,
                      "PagedAttention: zero point is outside its storage range.");
}

int ReadCode(const uint8_t *bytes, size_t index, int32_t type) {
  int value = FourBit(type) ? ((bytes[index / 2] >> (4 * (index % 2))) & 15) : bytes[index];
  if (type == DataType::INT4 && value >= 8)
    value -= 16;
  if (type == DataType::INT8 && value >= 128)
    value -= 256;
  return value;
}

float ReadScale(const TensorProto &tensor, size_t index) {
  if (tensor.has_raw_data()) {
    const uint8_t *bytes = tensor.raw_data().data() + index * sizeof(float);
    const uint32_t bits = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
                          (static_cast<uint32_t>(bytes[2]) << 16) |
                          (static_cast<uint32_t>(bytes[3]) << 24);
    float value;
    std::memcpy(&value, &bits, sizeof(float));
    return value;
  }
  return tensor.float_data()[index];
}

int ReadZero(const AffineLayoutProto &affine, size_t index) {
  if (!affine.has_zero_point())
    return 0;
  const auto &tensor = affine.zero_point();
  if (tensor.has_raw_data())
    return ReadCode(tensor.raw_data().data(), index, affine.storage_type());
  if (FourBit(affine.storage_type())) {
    int code = (static_cast<uint32_t>(tensor.int32_data()[index / 8]) >> (4 * (index % 8))) & 15;
    return affine.storage_type() == DataType::INT4 && code >= 8 ? code - 16 : code;
  }
  return tensor.int32_data()[index];
}

const RuntimeValue &Field(const RuntimeValue &value, const char *name) {
  const auto it = value.fields.find(name);
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kStruct && it != value.fields.end(),
                      "PagedAttention: missing cache field ", name, ".");
  return it->second;
}

int64_t Scalar(const RuntimeValue &value) {
  EXT_ENFORCE_INVALID(value.kind == RuntimeValue::Kind::kTensor &&
                          value.tensor.data_type == DataType::INT64 && value.tensor.shape.empty() &&
                          value.tensor.size_bytes() == sizeof(int64_t) &&
                          value.tensor.bytes() != nullptr,
                      "PagedAttention: start/length must be INT64 scalars.");
  int64_t result;
  std::memcpy(&result, value.tensor.bytes(), sizeof(result));
  return result;
}

struct PageView {
  const RuntimeValue *value;
  int64_t capacity, width;
  int axis = -1;
  uint64_t block = 0;

  explicit PageView(const RuntimeValue &source, bool checked = false) : value(&source) {
    if (source.kind == RuntimeValue::Kind::kTensor) {
      CheckTensor(source.tensor);
      capacity = source.tensor.shape[2];
      width = source.tensor.shape[3];
      return;
    }
    EXT_ENFORCE_INVALID(source.kind == RuntimeValue::Kind::kEncoded &&
                            source.Encoded().has_affine(),
                        "PagedAttention: only dense or built-in affine pages are supported.");
    const auto &encoded = source.Encoded();
    EXT_ENFORCE_INVALID(encoded.raw_data().empty() || encoded.raw_data().data() != nullptr,
                        "PagedAttention: null encoded payload.");
    for (const TensorProto *parameter :
         {encoded.affine().has_scale() ? &encoded.affine().scale() : nullptr,
          encoded.affine().has_zero_point() ? &encoded.affine().zero_point() : nullptr})
      if (parameter)
        EXT_ENFORCE_INVALID(parameter->raw_data().empty() ||
                                parameter->raw_data().data() != nullptr,
                            "PagedAttention: null affine parameter payload.");
    if (!checked) {
      const auto layout = StructTypeCatalogue().ValidateEncodedValue(encoded);
      EXT_ENFORCE_INVALID(!layout.external && layout.content_verified,
                          "PagedAttention: pages require verified inline payloads.");
    }
    const auto &type = encoded.logical_type().tensor_type();
    EXT_ENFORCE_INVALID(type.elem_type() == DataType::FLOAT && type.shape().dim_size() == 4 &&
                            type.shape().dim(0).dim_value() == 1 &&
                            type.shape().dim(1).dim_value() == 1,
                        "PagedAttention: affine logical type must be FLOAT [1,1,L,D].");
    capacity = type.shape().dim(2).dim_value();
    width = type.shape().dim(3).dim_value();
    CheckedBytes(capacity, width);
    const auto &affine = encoded.affine();
    CodeRange(affine.storage_type());
    EXT_ENFORCE_INVALID(affine.scale().data_type() == DataType::FLOAT,
                        "PagedAttention: affine scale parameters must be FLOAT.");
    if (affine.has_axis()) {
      axis = static_cast<int>(affine.axis() < 0 ? affine.axis() + 4 : affine.axis());
      block = affine.has_block_size() ? affine.block_size() : 0;
    }
    size_t count = 1;
    for (int i = 0; i < affine.scale().dims_size(); ++i)
      count *= static_cast<size_t>(affine.scale().dims(i));
    for (size_t i = 0; !checked && i < count; ++i) {
      const float scale = ReadScale(affine.scale(), i);
      EXT_ENFORCE_INVALID(std::isfinite(scale) && scale > 0,
                          "PagedAttention: affine scales must be finite and positive.");
      const auto [low, high] = CodeRange(affine.storage_type());
      const int zero = ReadZero(affine, i);
      EXT_ENFORCE_INVALID(zero >= low && zero <= high,
                          "PagedAttention: affine zero point is out of range.");
    }
  }

  double Read(int64_t row, int64_t column, PagedAttention::Statistics &statistics) const {
    const size_t index = static_cast<size_t>(row) * width + column;
    if (value->kind == RuntimeValue::Kind::kTensor) {
      const float result = value->tensor.AsFloat()[index];
      EXT_ENFORCE_INVALID(std::isfinite(result), "PagedAttention: non-finite dense cache value.");
      return result;
    }
    const auto &encoded = value->Encoded();
    const auto &affine = encoded.affine();
    size_t parameter = 0;
    if (axis >= 0) {
      const uint64_t coordinate[4] = {0, 0, static_cast<uint64_t>(row),
                                      static_cast<uint64_t>(column)};
      if (block == 0)
        parameter = coordinate[axis];
      else
        for (int i = 0; i < 4; ++i)
          parameter = parameter * static_cast<size_t>(affine.scale().dims(i)) +
                      (i == axis ? coordinate[i] / block : coordinate[i]);
    }
    AddBytes(statistics.dequantized_bytes, sizeof(float));
    const float scale = ReadScale(affine.scale(), parameter);
    const int zero = ReadZero(affine, parameter);
    const auto [low, high] = CodeRange(affine.storage_type());
    EXT_ENFORCE_INVALID(std::isfinite(scale) && scale > 0 && zero >= low && zero <= high,
                        "PagedAttention: invalid affine parameter value.");
    const double result =
        (ReadCode(encoded.raw_data().data(), index, affine.storage_type()) - zero) *
        static_cast<double>(scale);
    EXT_ENFORCE_INVALID(std::abs(result) <= std::numeric_limits<float>::max(),
                        "PagedAttention: decoded value exceeds FLOAT range.");
    return static_cast<float>(result);
  }
};

Tensor Allocate(RuntimeContext *rt, int slot, int32_t type, const Shape &shape, size_t bytes) {
  return rt ? rt->MakeOutputTensor(slot, type, shape, bytes)
            : MakeOutputTensor(type, shape, bytes, nullptr);
}

RuntimeValue NewPage(const Tensor &input, int64_t begin, int64_t length,
                     const PagedAttention::Format &format, RuntimeContext *rt,
                     PagedAttention::Statistics &statistics) {
  const int64_t width = input.shape[3];
  const size_t count = CheckedBytes(length, width, 1);
  const Shape shape{1, 1, length, width};
  const float *source = input.AsFloat() + static_cast<size_t>(begin) * width;
  if (format.storage_type == DataType::FLOAT) {
    Tensor output = MakeOutputTensor(DataType::FLOAT, shape, CheckedBytes(length, width),
                                     rt ? rt->io_allocator() : nullptr);
    std::memcpy(output.mutable_bytes(), source, output.size_bytes());
    AddBytes(statistics.copied_bytes, output.size_bytes());
    return RuntimeValue(std::move(output)).Retain();
  }
  const size_t bytes = FourBit(format.storage_type) ? count / 2 + count % 2 : count;
  Tensor storage = MakeOutputTensor(DataType::UINT8, {static_cast<int64_t>(bytes)}, bytes,
                                    rt ? rt->io_allocator() : nullptr);
  std::memset(storage.mutable_bytes(), 0, bytes);
  const auto [low, high] = CodeRange(format.storage_type);
  for (size_t i = 0; i < count; ++i) {
    // Clamping before conversion keeps even very small scales within integer bounds.
    const double scaled = std::clamp(static_cast<double>(source[i]) / format.scale,
                                     static_cast<double>(low - format.zero_point),
                                     static_cast<double>(high - format.zero_point));
    const double floor = std::floor(scaled);
    const double fraction = scaled - floor;
    const int base = static_cast<int>(floor);
    const int rounded = base + (fraction > 0.5 || (fraction == 0.5 && base % 2 != 0));
    const int code = std::clamp(rounded + format.zero_point, low, high);
    if (FourBit(format.storage_type))
      storage.mutable_bytes()[i / 2] |= static_cast<uint8_t>((code & 15) << (4 * (i % 2)));
    else
      storage.mutable_bytes()[i] = static_cast<uint8_t>(code);
  }
  storage = std::move(storage).RetainStorage();
  EncodedValueProto encoded;
  auto *logical = encoded.mutable_logical_type()->mutable_tensor_type();
  logical->set_elem_type(DataType::FLOAT);
  for (int64_t d : shape)
    logical->mutable_shape()->add_dim()->set_dim_value(d);
  auto *affine = encoded.mutable_affine();
  affine->set_storage_type(static_cast<TensorProto::DataType>(format.storage_type));
  affine->mutable_scale()->set_data_type(DataType::FLOAT);
  affine->mutable_scale()->add_float_data(format.scale);
  affine->mutable_zero_point()->set_data_type(format.storage_type);
  // Packed raw data also represents a scalar four-bit zero point unambiguously.
  const uint8_t zero =
      static_cast<uint8_t>(format.zero_point) & (FourBit(format.storage_type) ? 15 : 255);
  affine->mutable_zero_point()->set_raw_data(&zero, 1);
  encoded.ref_raw_data().assign_borrowed(storage.bytes(), bytes, storage.borrowed_owner());
  AddBytes(statistics.copied_bytes, bytes);
  return RuntimeValue(std::move(encoded)).Retain();
}

} // namespace

struct PagedAttention::CacheAnalysis {
  struct Summary {
    int64_t first = 0, end = 0, capacity = 0, key_width = 0, value_width = 0;
  };
  core::runtime::RuntimeSequence::Memo<Summary> memo;

  Summary Validate(const core::runtime::RuntimeSequence &pages, Statistics &statistics) {
    return pages.Fold(
        memo,
        [&](const RuntimeValue &page) {
          ++statistics.validated_pages;
          EXT_ENFORCE_INVALID(page.fields.size() == 4, "PagedAttention: unexpected page fields.");
          const int64_t start = Scalar(Field(page, "start"));
          const int64_t length = Scalar(Field(page, "length"));
          const PageView key(Field(page, "key")), value(Field(page, "value"));
          EXT_ENFORCE_INVALID(start >= 0 && length > 0 && length <= INT64_MAX - start &&
                                  length <= key.capacity && key.capacity == value.capacity,
                              "PagedAttention: invalid page shape, start, length or capacity.");
          return Summary{start, start + length, key.capacity, key.width, value.width};
        },
        [](const Summary &left, const Summary &right) {
          EXT_ENFORCE_INVALID(left.end == right.first && left.key_width == right.key_width &&
                                  left.value_width == right.value_width,
                              "PagedAttention: noncontiguous pages or inconsistent widths.");
          return Summary{left.first, right.end, std::max(left.capacity, right.capacity),
                         left.key_width, left.value_width};
        });
  }
};

PagedAttention::PagedAttention(const KernelContext &context)
    : KernelBase(context), cache_analysis_(std::make_shared<CacheAnalysis>()) {}

PagedAttention::PagedAttention(const KernelContext &context, FormatSelector format_selector)
    : KernelBase(context), cache_analysis_(std::make_shared<CacheAnalysis>()),
      format_selector_(std::move(format_selector)) {
  EXT_ENFORCE_INVALID(format_selector_ != nullptr,
                      "PagedAttention: format selector must not be empty.");
}

RuntimeValue PagedAttention::EmptyCache() {
  RuntimeValue result;
  result.fields.emplace("blocks", RuntimeValue(std::vector<RuntimeValue>{}));
  return result;
}

PagedAttention::Result PagedAttention::operator()(const Tensor &q, const Tensor &k, const Tensor &v,
                                                  const RuntimeValue &past, const Options &options,
                                                  RuntimeContext *rt) const {
  EXT_ENFORCE_INVALID(options.block_size > 0 && options.max_tokens > 0 &&
                          options.left_window_size >= -1,
                      "PagedAttention: invalid block size, capacity or window.");
  CheckFormat(options.key_format);
  CheckFormat(options.value_format);
  for (const Tensor *input : {&q, &k, &v})
    CheckTensor(*input);
  EXT_ENFORCE_INVALID(q.shape[2] == k.shape[2] && k.shape[2] == v.shape[2] &&
                          q.shape[3] == k.shape[3],
                      "PagedAttention: mismatched query/key/value dimensions.");
  const auto &pages = Field(past, "blocks");
  EXT_ENFORCE_INVALID(past.fields.size() == 1 && pages.kind == RuntimeValue::Kind::kSequence,
                      "PagedAttention: cache must contain only a blocks sequence.");
  const int64_t length = k.shape[2];
  EXT_ENFORCE_INVALID(length <= options.max_tokens, "PagedAttention: max_tokens exceeded.");
  Result result;
  const auto summary = cache_analysis_->Validate(pages.elements, result.statistics);
  const int64_t past_length = summary.end;
  EXT_ENFORCE_INVALID(pages.elements.empty() ||
                          (summary.first == 0 && summary.capacity <= options.block_size &&
                           summary.key_width == k.shape[3] && summary.value_width == v.shape[3] &&
                           past_length <= options.max_tokens),
                      "PagedAttention: invalid cache shape or capacity.");
  EXT_ENFORCE_INVALID(length <= options.max_tokens - past_length,
                      "PagedAttention: max_tokens exceeded.");
  for (const Tensor *input : {&q, &k, &v})
    for (size_t i = 0; i < input->size_bytes() / sizeof(float); ++i)
      EXT_ENFORCE_INVALID(std::isfinite(input->AsFloat()[i]),
                          "PagedAttention: inputs must be finite.");
  const int64_t total = past_length + length;
  const size_t output_bytes = CheckedBytes(length, v.shape[3]);
  const size_t workspace_bytes = length == 0 ? 0 : CheckedBytes(1, v.shape[3], sizeof(double));
  Formats formats{options.key_format, options.value_format};
  if (length > 0 && format_selector_)
    formats = format_selector_(k, v, past_length, formats);
  CheckFormat(formats.key);
  CheckFormat(formats.value);
  const StructTypeCatalogue empty_catalogue;
  const auto &catalogue = rt ? rt->struct_type_catalogue() : empty_catalogue;
  result.present = past.BorrowView().Retain(catalogue);
  auto &present_pages = result.present.fields.at("blocks").elements;
  for (int64_t begin = 0; begin < length;) {
    const int64_t chunk = std::min(options.block_size, length - begin);
    RuntimeValue page;
    page.fields.emplace("start",
                        RuntimeValue(Tensor::FromInt64("", {}, {past_length + begin})).Retain());
    page.fields.emplace("length", RuntimeValue(Tensor::FromInt64("", {}, {chunk})).Retain());
    page.fields.emplace("key", NewPage(k, begin, chunk, formats.key, rt, result.statistics));
    page.fields.emplace("value", NewPage(v, begin, chunk, formats.value, rt, result.statistics));
    present_pages.push_back(std::move(page));
    begin += chunk;
  }
  result.present = std::move(result.present).Retain(catalogue);
  const auto &retained_pages = result.present.fields.at("blocks").elements;
  cache_analysis_->Validate(retained_pages, result.statistics);
  result.Y = Allocate(rt, 0, DataType::FLOAT, {1, 1, length, v.shape[3]}, output_bytes);
  if (length == 0)
    return result;
  Tensor workspace =
      rt ? rt->MakeTemporaryTensor(DataType::DOUBLE, {v.shape[3]}, workspace_bytes)
         : MakeOutputTensor(DataType::DOUBLE, {v.shape[3]}, workspace_bytes, nullptr);
  result.statistics.peak_workspace_bytes = workspace_bytes;
  double *accumulator = workspace.AsDouble();
  const double scale = 1.0 / std::sqrt(static_cast<double>(q.shape[3]));
  struct ActivePage {
    int64_t start, length;
    PageView key, value;
  };
  const int64_t first_attended =
      options.left_window_size < 0 ? 0
                                   : past_length - std::min(past_length, options.left_window_size);
  size_t low = 0, high = retained_pages.size();
  while (low < high) {
    const size_t middle = low + (high - low) / 2;
    const auto &page = retained_pages[middle];
    if (Scalar(Field(page, "start")) + Scalar(Field(page, "length")) <= first_attended)
      low = middle + 1;
    else
      high = middle;
  }
  std::vector<ActivePage> active;
  active.reserve(retained_pages.size() - low);
  for (size_t i = low; i < retained_pages.size(); ++i) {
    const auto &page = retained_pages[i];
    active.push_back({Scalar(Field(page, "start")), Scalar(Field(page, "length")),
                      PageView(Field(page, "key"), true), PageView(Field(page, "value"), true)});
  }
  for (int64_t row = 0; row < length; ++row) {
    std::fill(accumulator, accumulator + v.shape[3], 0);
    const int64_t position = past_length + row;
    const int64_t first =
        options.left_window_size < 0 ? 0 : position - std::min(position, options.left_window_size);
    const int64_t end = options.is_causal ? position + 1 : total;
    double maximum = -std::numeric_limits<double>::infinity(), denominator = 0;
    for (const auto &page : active) {
      const int64_t start = page.start, count = page.length;
      if (start >= end || start + count <= first)
        continue;
      const auto &key = page.key;
      const auto &value = page.value;
      for (int64_t token = std::max(first, start); token < std::min(end, start + count); ++token) {
        double score = 0;
        for (int64_t d = 0; d < q.shape[3]; ++d)
          score += static_cast<double>(q.AsFloat()[static_cast<size_t>(row) * q.shape[3] + d]) *
                   key.Read(token - start, d, result.statistics);
        score *= scale;
        const double next_maximum = std::max(maximum, score);
        const double previous_weight = std::exp(maximum - next_maximum);
        const double weight = std::exp(score - next_maximum);
        denominator = denominator * previous_weight + weight;
        for (int64_t d = 0; d < v.shape[3]; ++d)
          accumulator[d] = accumulator[d] * previous_weight +
                           weight * value.Read(token - start, d, result.statistics);
        maximum = next_maximum;
      }
    }
    for (int64_t d = 0; d < v.shape[3]; ++d)
      result.Y.AsFloat()[static_cast<size_t>(row) * v.shape[3] + d] =
          static_cast<float>(accumulator[d] / denominator);
  }
  return result;
}

void PagedAttention::Run(RuntimeContext &rt) {
  EXT_ENFORCE_INVALID(node_ != nullptr, "PagedAttention: missing node.");
  const auto &node = *node_;
  EXT_ENFORCE_INVALID(node.input_size() == 4 && node.output_size() == 2 &&
                          !node.output(0).empty() && !node.output(1).empty() &&
                          node.output(0) != node.output(1),
                      "PagedAttention: expects Q,K,V,past and Y,present.");
  Options options;
  std::unordered_set<std::string> seen;
  for (const auto &attribute : node.attribute()) {
    const std::string &name = attribute.name();
    EXT_ENFORCE_INVALID(seen.insert(name).second, "PagedAttention: duplicate attribute ", name);
    if (name == "key_scale" || name == "value_scale") {
      EXT_ENFORCE_INVALID(attribute.type() == AttributeProto::FLOAT && attribute.has_f(),
                          "PagedAttention: scale attribute must be FLOAT.");
      (name == "key_scale" ? options.key_format : options.value_format).scale = attribute.f();
      continue;
    }
    EXT_ENFORCE_INVALID(attribute.type() == AttributeProto::INT && attribute.has_i(),
                        "PagedAttention: unsupported attribute or attribute type: ", name);
    const int64_t value = attribute.i();
    if (name == "block_size")
      options.block_size = value;
    else if (name == "max_tokens")
      options.max_tokens = value;
    else if (name == "left_window_size")
      options.left_window_size = value;
    else if (name == "is_causal") {
      EXT_ENFORCE_INVALID(value == 0 || value == 1, "PagedAttention: is_causal must be 0 or 1.");
      options.is_causal = value != 0;
    } else if (name == "key_storage_type" || name == "value_storage_type" ||
               name == "key_zero_point" || name == "value_zero_point") {
      EXT_ENFORCE_INVALID(value >= std::numeric_limits<int32_t>::min() &&
                              value <= std::numeric_limits<int32_t>::max(),
                          "PagedAttention: integer attribute out of range.");
      auto &format = name.compare(0, 4, "key_") == 0 ? options.key_format : options.value_format;
      if (name.find("storage_type") != std::string::npos)
        format.storage_type = static_cast<int32_t>(value);
      else
        format.zero_point = static_cast<int32_t>(value);
    } else
      EXT_THROW_INVALID("PagedAttention: unsupported attribute: ", name);
  }
  const auto past = rt.values().find(node.input(3));
  EXT_ENFORCE_INVALID(past != rt.values().end(), "PagedAttention: missing past cache.");
  Result result = (*this)(GetInput(node, 0, rt.tensors()), GetInput(node, 1, rt.tensors()),
                          GetInput(node, 2, rt.tensors()), past->second, options, &rt);
  SetOutput(node, 0, std::move(result.Y), rt);
  rt.PutValue(node.output(1), std::move(result.present));
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_kernels::kernel

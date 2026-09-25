// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_value.h"
#include "onnx_core/runtime/quantization.h"

#include <cstring>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

RuntimeValue::RuntimeValue(EncodedValueProto value)
    : kind(Kind::kEncoded), encoded(std::make_shared<EncodedValueProto>(std::move(value))) {}

RuntimeValue RuntimeValue::FromEncodedView(const EncodedValueProto &value,
                                           std::shared_ptr<void> owner) {
  EXT_ENFORCE_INVALID(owner.use_count() != 0,
                      "RuntimeValue::FromEncodedView: an owner token is required.");
  RuntimeValue result;
  result.kind = Kind::kEncoded;
  result.encoded = std::shared_ptr<const EncodedValueProto>(std::move(owner), &value);
  return result;
}

const EncodedValueProto &RuntimeValue::Encoded() const {
  EXT_ENFORCE_INVALID(encoded != nullptr, "RuntimeValue: missing encoded message.");
  return *encoded;
}

RuntimeValue
RuntimeValue::FromPagedCache(PagedCacheProto value, const StructTypeCatalogue &catalogue,
                             std::shared_ptr<const QuantizationParameterCatalogue> parameters) {
  catalogue.ValidatePagedCache(value);
  auto owner = std::make_shared<PagedCacheProto>(std::move(value));
  const auto payload = [&](const TensorProto *dense, const EncodedValueProto *encoded) {
    if (encoded) {
      RuntimeValue result = FromEncodedView(*encoded, owner);
      result.quantization_parameters = parameters;
      return result;
    }
    Shape shape;
    for (const auto dim : dense->dims())
      shape.push_back(static_cast<int64_t>(dim));
    const auto &raw = dense->raw_data();
    const bool has_raw = dense->has_raw_data();
    return RuntimeValue(Tensor::Borrow(
        dense->name(), DataType::FLOAT, shape,
        has_raw ? raw.data() : reinterpret_cast<const uint8_t *>(dense->float_data().data()),
        has_raw ? raw.size() : dense->float_data().size() * sizeof(float),
        has_raw && raw.is_borrowed() ? raw.owner() : owner));
  };
  std::vector<RuntimeValue> blocks;
  blocks.reserve(owner->blocks().size());
  for (const auto &block : owner->blocks()) {
    RuntimeValue page;
    page.fields.emplace("start", RuntimeValue(Tensor::FromInt64("", {}, {block.start()})));
    page.fields.emplace("length", RuntimeValue(Tensor::FromInt64("", {}, {block.length()})));
    page.fields.emplace("key", payload(block.has_key() ? &block.key() : nullptr,
                                       block.has_encoded_key() ? &block.encoded_key() : nullptr));
    page.fields.emplace("value",
                        payload(block.has_value() ? &block.value() : nullptr,
                                block.has_encoded_value() ? &block.encoded_value() : nullptr));
    blocks.push_back(std::move(page));
  }
  RuntimeValue result;
  result.fields.emplace("blocks", RuntimeValue(std::move(blocks)));
  return std::move(result).Retain(catalogue);
}

PagedCacheProto RuntimeValue::ToPagedCache(const std::string &name,
                                           const StructTypeCatalogue &catalogue) const {
  const auto field = [](const RuntimeValue &value, const char *key) -> const RuntimeValue & {
    const auto found = value.fields.find(key);
    EXT_ENFORCE_INVALID(value.kind == Kind::kStruct && found != value.fields.end(),
                        "ToPagedCache: missing field ", key, ".");
    return found->second;
  };
  const auto scalar = [&](const RuntimeValue &value, const char *key) {
    const auto &item = field(value, key);
    EXT_ENFORCE_INVALID(item.kind == Kind::kTensor && item.tensor.data_type == DataType::INT64 &&
                            item.tensor.shape.empty() &&
                            item.tensor.size_bytes() == sizeof(int64_t) &&
                            item.tensor.bytes() != nullptr,
                        "ToPagedCache: start/length must be INT64 scalars.");
    int64_t result;
    std::memcpy(&result, item.tensor.bytes(), sizeof(result));
    return result;
  };
  const auto &blocks = field(*this, "blocks");
  EXT_ENFORCE_INVALID(fields.size() == 1 && blocks.kind == Kind::kSequence,
                      "ToPagedCache: requires a blocks sequence.");
  PagedCacheProto result;
  if (!name.empty())
    result.set_name(name);
  for (const auto &page : blocks.elements) {
    EXT_ENFORCE_INVALID(page.fields.size() == 4, "ToPagedCache: unexpected page fields.");
    auto *block = result.add_blocks();
    block->set_start(scalar(page, "start"));
    block->set_length(scalar(page, "length"));
    for (const char *key : {"key", "value"}) {
      const auto &source = field(page, key);
      const bool is_key = std::string(key) == "key";
      if (source.kind == Kind::kEncoded) {
        if (source.Encoded().has_parameter_ref()) {
          EXT_ENFORCE_INVALID(source.quantization_parameters != nullptr,
                              "ToPagedCache: missing shared parameter catalogue.");
          source.quantization_parameters->Validate(source.Encoded(), catalogue);
        }
        *(is_key ? block->mutable_encoded_key() : block->mutable_encoded_value()) =
            source.Encoded();
      } else {
        EXT_ENFORCE_INVALID(source.kind == Kind::kTensor &&
                                source.tensor.data_type == DataType::FLOAT,
                            "ToPagedCache: requires FLOAT tensor or encoded pages.");
        const auto &tensor = source.tensor;
        const auto count = tensor.shape.product(0, tensor.shape.size(), "ToPagedCache");
        EXT_ENFORCE_INVALID(tensor.size_bytes() == PackedByteSize(DataType::FLOAT, count) &&
                                (tensor.size_bytes() == 0 || tensor.bytes() != nullptr),
                            "ToPagedCache: invalid tensor payload.");
        auto *dense = is_key ? block->mutable_key() : block->mutable_value();
        dense->set_data_type(DataType::FLOAT);
        for (const auto dim : tensor.shape)
          dense->add_dims(dim);
        Tensor storage =
            tensor.borrowed_owner().use_count() != 0 ? tensor.BorrowView() : tensor.ToOwned();
        storage = std::move(storage).RetainStorage();
        dense->ref_raw_data().assign_borrowed(storage.bytes(), storage.size_bytes(),
                                              storage.borrowed_owner());
      }
    }
  }
  catalogue.ValidatePagedCache(result);
  return result;
}

RuntimeValue RuntimeValue::BorrowView() const { return CopyAtDepth(0, false); }

RuntimeValue RuntimeValue::DeepCopy() const { return CopyAtDepth(0, true); }

RuntimeValue RuntimeValue::Retain(const StructTypeCatalogue &catalogue) && {
  RetainAtDepth(0, catalogue);
  return std::move(*this);
}

void RuntimeValue::RetainAtDepth(size_t depth, const StructTypeCatalogue &catalogue) {
  EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
  if (kind == Kind::kTensor)
    tensor = std::move(tensor).RetainStorage();
  else if (kind == Kind::kEncoded) {
    if (Encoded().has_parameter_ref()) {
      EXT_ENFORCE_INVALID(quantization_parameters != nullptr,
                          "RuntimeValue::Retain: missing shared quantization parameter catalogue.");
      quantization_parameters->Validate(Encoded(), catalogue);
    }
    if (Encoded().has_struct_type())
      ValidatePersistentStructType(catalogue, Encoded().struct_type());
    if (Encoded().has_logical_type())
      ValidatePersistentType(catalogue, Encoded().logical_type());
    const auto retain_payload = [](const auto &raw) {
      EXT_ENFORCE_INVALID(!raw.is_borrowed() || raw.owner().use_count() != 0 || raw.empty(),
                          "RuntimeValue::Retain: cannot retain an ownerless encoded payload.");
    };
    retain_payload(Encoded().raw_data());
    if (Encoded().has_affine()) {
      retain_payload(Encoded().affine().scale().raw_data());
      retain_payload(Encoded().affine().zero_point().raw_data());
    }
  } else if (kind == Kind::kSequence) {
    for (auto &value : elements)
      value.RetainAtDepth(depth + 1, catalogue);
  } else
    for (auto &[name, value] : fields)
      value.RetainAtDepth(depth + 1, catalogue);
}

RuntimeValue RuntimeValue::CopyAtDepth(size_t depth, bool owned) const {
  EXT_ENFORCE_INVALID(depth <= kMaxDepth, "RuntimeValue: maximum nesting depth exceeded.");
  if (kind == Kind::kTensor)
    return RuntimeValue(owned || tensor.data_type == DataType::STRING ? tensor.ToOwned()
                                                                      : tensor.BorrowView());
  if (kind == Kind::kEncoded) {
    if (!owned) {
      RuntimeValue result;
      result.kind = kind;
      result.encoded = encoded;
      result.quantization_parameters = quantization_parameters;
      return result;
    }
    EncodedValueProto copy;
    copy.ParseFromString(Encoded().SerializeAsString());
    RuntimeValue result(std::move(copy));
    result.quantization_parameters = quantization_parameters;
    return result;
  }
  RuntimeValue result;
  result.kind = kind;
  if (kind == Kind::kSequence) {
    result.elements.reserve(elements.size());
    for (const auto &value : elements)
      result.elements.push_back(value.CopyAtDepth(depth + 1, owned));
  } else {
    for (const auto &[name, value] : fields)
      result.fields.emplace(name, value.CopyAtDepth(depth + 1, owned));
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

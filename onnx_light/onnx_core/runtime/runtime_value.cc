// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/runtime_value.h"
#include "onnx_core/runtime/quantization.h"

#include <cstring>
#include <string_view>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

RuntimeSequence::Node::Node(RuntimeValue element, bool is_retained)
    : value(std::make_shared<RuntimeValue>(std::move(element))), depth(value->Depth()),
      retained(is_retained), catalogue_independent(value->CatalogueIndependent()) {}

RuntimeSequence::Node::Node(Root first, Root second)
    : left(std::move(first)), right(std::move(second)), size(left->size + right->size),
      height(1 + std::max(left->height, right->height)), depth(std::max(left->depth, right->depth)),
      retained(left->retained && right->retained),
      catalogue_independent(left->catalogue_independent && right->catalogue_independent) {}

RuntimeSequence::Root RuntimeSequence::Append(Root root, Root leaf) {
  if (!root)
    return leaf;
  if (root->value)
    return std::make_shared<Node>(std::move(root), std::move(leaf));
  auto right = Append(root->right, std::move(leaf));
  if (right->height > root->left->height + 1) {
    if (right->left->height > right->right->height)
      return std::make_shared<Node>(std::make_shared<Node>(root->left, right->left->left),
                                    std::make_shared<Node>(right->left->right, right->right));
    return std::make_shared<Node>(std::make_shared<Node>(root->left, right->left), right->right);
  }
  return std::make_shared<Node>(root->left, std::move(right));
}

RuntimeSequence::Root RuntimeSequence::Replace(const Root &root, size_t index, Root leaf) {
  if (root->value)
    return leaf;
  if (index < root->left->size)
    return std::make_shared<Node>(Replace(root->left, index, std::move(leaf)), root->right);
  return std::make_shared<Node>(root->left,
                                Replace(root->right, index - root->left->size, std::move(leaf)));
}

const RuntimeValue &RuntimeSequence::At(const Root &root, size_t index) {
  const Node *node = root.get();
  while (!node->value) {
    if (index < node->left->size)
      node = node->left.get();
    else {
      index -= node->left->size;
      node = node->right.get();
    }
  }
  return *node->value;
}

RuntimeSequence::RuntimeSequence(std::vector<RuntimeValue> values) {
  for (auto &value : values)
    push_back(std::move(value));
}

const RuntimeValue &RuntimeSequence::at(size_t index) const {
  EXT_ENFORCE_INVALID(index < size(), "RuntimeSequence: index out of range.");
  return At(root_, index);
}

void RuntimeSequence::push_back(RuntimeValue value) {
  root_ = Append(root_, std::make_shared<Node>(std::move(value), false));
}

void RuntimeSequence::Set(size_t index, RuntimeValue value) {
  EXT_ENFORCE_INVALID(index < size(), "RuntimeSequence: index out of range.");
  root_ = Replace(root_, index, std::make_shared<Node>(std::move(value), false));
}

RuntimeSequence::Root
RuntimeSequence::RetainNode(const Root &root, const StructTypeCatalogue &catalogue, size_t depth) {
  if (!root || (root->retained && root->catalogue_independent))
    return root;
  if (root->value) {
    RuntimeValue copy = *root->value;
    copy.RetainAtDepth(depth, catalogue);
    return root->retained ? root : std::make_shared<Node>(std::move(copy), true);
  }
  auto left = RetainNode(root->left, catalogue, depth);
  auto right = RetainNode(root->right, catalogue, depth);
  if (left == root->left && right == root->right)
    return root;
  return std::make_shared<Node>(std::move(left), std::move(right));
}

void RuntimeSequence::Retain(const StructTypeCatalogue &catalogue, size_t depth) {
  EXT_ENFORCE_INVALID(empty() || depth + this->depth() <= RuntimeValue::kMaxDepth,
                      "RuntimeValue: maximum nesting depth exceeded.");
  root_ = RetainNode(root_, catalogue, depth);
}

RuntimeSequence::Root
RuntimeSequence::TransformNode(const Root &root,
                               const std::function<RuntimeValue(const RuntimeValue &)> &transform) {
  if (!root)
    return {};
  if (root->value)
    return std::make_shared<Node>(transform(*root->value), false);
  return std::make_shared<Node>(TransformNode(root->left, transform),
                                TransformNode(root->right, transform));
}

RuntimeSequence RuntimeSequence::Transform(
    const std::function<RuntimeValue(const RuntimeValue &)> &transform) const {
  RuntimeSequence result;
  result.root_ = TransformNode(root_, transform);
  return result;
}

RuntimeSequence::Root
RuntimeSequence::TransformOwnedNode(const Root &root, bool shared,
                                    const std::function<void(RuntimeValue &)> &transform) {
  if (!root)
    return {};
  shared = shared || root.use_count() != 1;
  if (root->value) {
    // Node constructs a mutable value, exposed only through const views. Moving
    // unique unpublished storage preserves allocator handles without a second allocation.
    RuntimeValue value =
        shared ? *root->value : std::move(*std::const_pointer_cast<RuntimeValue>(root->value));
    transform(value);
    return std::make_shared<Node>(std::move(value), false);
  }
  return std::make_shared<Node>(TransformOwnedNode(root->left, shared, transform),
                                TransformOwnedNode(root->right, shared, transform));
}

void RuntimeSequence::TransformInPlace(const std::function<void(RuntimeValue &)> &transform) {
  root_ = TransformOwnedNode(root_, false, transform);
}

size_t RuntimeValue::Depth(size_t depth) const {
  if (depth > kMaxDepth)
    return kMaxDepth + 1;
  if (kind == Kind::kSequence)
    return elements.empty() ? 0 : std::min(kMaxDepth + 1, elements.depth() + 1);
  size_t result = 0;
  if (kind == Kind::kStruct)
    for (const auto &[name, value] : fields)
      result = std::max(result, std::min(kMaxDepth + 1, value.Depth(depth + 1) + 1));
  return result;
}

bool RuntimeValue::CatalogueIndependent(size_t depth) const {
  if (depth > kMaxDepth)
    return false;
  if (kind == Kind::kEncoded)
    return encoded && encoded->has_affine() && !encoded->has_parameter_ref();
  if (kind == Kind::kSequence)
    return !elements.root_ || elements.root_->catalogue_independent;
  if (kind == Kind::kStruct)
    for (const auto &[name, value] : fields)
      if (!value.CatalogueIndependent(depth + 1))
        return false;
  return true;
}

namespace {

TensorProto BorrowTensorProto(const TensorProto &source, const std::shared_ptr<void> &owner) {
  TensorProto result;
  result.ref_dims() = source.ref_dims();
  if (source.has_data_type())
    result.set_data_type(source.data_type());
  if (source.has_segment())
    result.mutable_segment()->CopyFrom(source.segment());
  result.ref_float_data() = source.ref_float_data();
  result.ref_int32_data() = source.ref_int32_data();
  result.ref_string_data() = source.ref_string_data();
  result.ref_int64_data() = source.ref_int64_data();
  if (source.has_name())
    result.set_name(source.name().value());
  if (source.has_raw_data())
    result.ref_raw_data().assign_borrowed(source.raw_data().data(), source.raw_data().size(),
                                          owner);
  result.ref_double_data() = source.ref_double_data();
  result.ref_uint64_data() = source.ref_uint64_data();
  if (source.has_doc_string())
    result.set_doc_string(source.doc_string().value());
  result.ref_external_data() = source.ref_external_data();
  if (source.has_data_location())
    result.set_data_location(source.data_location());
  result.ref_metadata_props() = source.ref_metadata_props();
  return result;
}

EncodedValueProto BorrowEncodedValue(const RuntimeValue &source) {
  const EncodedValueProto &value = source.Encoded();
  const std::shared_ptr<void> owner(source.encoded,
                                    const_cast<EncodedValueProto *>(source.encoded.get()));
  EncodedValueProto result;
  if (value.has_affine()) {
    auto *affine = result.mutable_affine();
    if (value.affine().has_storage_type())
      affine->set_storage_type(value.affine().storage_type());
    if (value.affine().has_scale())
      *affine->mutable_scale() = BorrowTensorProto(value.affine().scale(), owner);
    if (value.affine().has_zero_point())
      *affine->mutable_zero_point() = BorrowTensorProto(value.affine().zero_point(), owner);
    if (value.affine().has_axis())
      affine->set_axis(value.affine().axis());
    if (value.affine().has_block_size())
      affine->set_block_size(value.affine().block_size());
    if (value.affine().has_signed_storage())
      affine->set_signed_storage(value.affine().signed_storage());
  } else if (value.has_struct_type()) {
    result.mutable_struct_type()->CopyFrom(value.struct_type());
  }
  if (value.has_logical_type())
    result.mutable_logical_type()->CopyFrom(value.logical_type());
  if (value.has_raw_data())
    result.ref_raw_data().assign_borrowed(value.raw_data().data(), value.raw_data().size(), owner);
  result.ref_external_data() = value.ref_external_data();
  if (value.has_data_location())
    result.set_data_location(value.data_location());
  if (value.has_name())
    result.set_name(value.name().value());
  if (value.has_doc_string())
    result.set_doc_string(value.doc_string().value());
  if (value.has_parameter_ref())
    result.set_parameter_ref(value.parameter_ref().value());
  return result;
}

} // namespace

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
    EXT_ENFORCE_INVALID(has_raw || dense->data_type() == DataType::FLOAT,
                        "FromPagedCache: non-FLOAT dense pages require raw_data.");
    return RuntimeValue(Tensor::Borrow(
        dense->name(), dense->data_type(), shape,
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

bool RuntimeValue::HasPagedCacheStructure() const {
  if (kind != Kind::kStruct || fields.size() != 1)
    return false;
  const auto blocks = fields.find("blocks");
  if (blocks == fields.end() || blocks->second.kind != Kind::kSequence)
    return false;
  for (const auto &page : blocks->second.elements) {
    if (page.kind != Kind::kStruct || page.fields.size() != 4)
      return false;
    for (const char *name : {"start", "length", "key", "value"}) {
      const auto field = page.fields.find(name);
      if (field == page.fields.end())
        return false;
      const auto &value = field->second;
      if (std::string_view(name) == "start" || std::string_view(name) == "length") {
        if (value.kind != Kind::kTensor || value.tensor.data_type != DataType::INT64 ||
            !value.tensor.shape.empty())
          return false;
      } else if (value.kind != Kind::kTensor && value.kind != Kind::kEncoded) {
        return false;
      }
    }
  }
  return true;
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
            BorrowEncodedValue(source);
      } else {
        EXT_ENFORCE_INVALID(
            source.kind == Kind::kTensor && (source.tensor.data_type == DataType::FLOAT ||
                                             source.tensor.data_type == DataType::FLOAT16 ||
                                             source.tensor.data_type == DataType::BFLOAT16),
            "ToPagedCache: requires a FLOAT, FLOAT16 or BFLOAT16 tensor or encoded pages.");
        const auto &tensor = source.tensor;
        const auto count = tensor.shape.product(0, tensor.shape.size(), "ToPagedCache");
        EXT_ENFORCE_INVALID(tensor.size_bytes() == PackedByteSize(tensor.data_type, count) &&
                                (tensor.size_bytes() == 0 || tensor.bytes() != nullptr),
                            "ToPagedCache: invalid tensor payload.");
        auto *dense = is_key ? block->mutable_key() : block->mutable_value();
        dense->set_data_type(source.tensor.data_type);
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
    if (Encoded().has_logical_type()) {
      ValidatePersistentType(catalogue, Encoded().logical_type());
      if (Encoded().logical_type().has_tensor_type()) {
        const auto &tensor = Encoded().logical_type().tensor_type();
        EXT_ENFORCE_INVALID(tensor.has_shape(),
                            "RuntimeValue::Retain: encoded values require a concrete shape.");
        for (const auto &dimension : tensor.shape().dim())
          EXT_ENFORCE_INVALID(dimension.has_dim_value() && !dimension.has_dim_param() &&
                                  dimension.dim_value() >= 0,
                              "RuntimeValue::Retain: encoded values require concrete dimensions.");
      }
    }
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
    elements.Retain(catalogue, depth + 1);
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
    EXT_ENFORCE_INVALID(elements.empty() || depth + 1 + elements.depth() <= kMaxDepth,
                        "RuntimeValue: maximum nesting depth exceeded.");
    result.elements = owned ? elements.Transform([depth](const RuntimeValue &value) {
      return value.CopyAtDepth(depth + 1, true);
    })
                            : elements;
  } else {
    for (const auto &[name, value] : fields)
      result.fields.emplace(name, value.CopyAtDepth(depth + 1, owned));
  }
  return result;
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

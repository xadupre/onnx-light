// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_verify.h"

#include <cstring>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {
namespace {

// Checks presence before accessing optional submessages and adds field names only on failure.
#define COMPARE_FIELD(name)                                                                        \
  if (left.has_##name() != right.has_##name()) {                                                   \
    return Fail(#name ": presence differs");                                                       \
  }                                                                                                \
  if (left.has_##name() && !Compare(left.ref_##name(), right.ref_##name())) {                      \
    Prefix(#name);                                                                                 \
    return false;                                                                                  \
  }

class ProtoComparison {
public:
  explicit ProtoComparison(std::string *difference) : difference_(difference) {
    if (difference_)
      difference_->clear();
  }

  template <typename T>
    requires(!std::is_base_of_v<Message, T>)
  bool Compare(const T &left, const T &right) {
    if constexpr (std::is_floating_point_v<T>) {
      if (std::memcmp(&left, &right, sizeof(T)) == 0)
        return true;
    } else if (left == right) {
      return true;
    }
    return Fail(": values differ");
  }

  template <typename T> bool CompareRepeated(const T &left, const T &right) {
    if (left.size() != right.size())
      return Fail(": size differs (" + std::to_string(left.size()) +
                  " != " + std::to_string(right.size()) + ")");
    for (size_t i = 0; i < left.size(); ++i) {
      if (!Compare(left[i], right[i])) {
        Prefix("[" + std::to_string(i) + "]");
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool Compare(const utils::RepeatedField<T> &left, const utils::RepeatedField<T> &right) {
    return CompareRepeated(left, right);
  }

  template <typename T>
  bool Compare(const utils::RepeatedProtoField<T> &left,
               const utils::RepeatedProtoField<T> &right) {
    return CompareRepeated(left, right);
  }

  bool Compare(const utils::RepeatedStringField &left, const utils::RepeatedStringField &right) {
    return CompareRepeated(left, right);
  }

  bool Compare(const TypeProto &left, const TypeProto &right) {
    COMPARE_FIELD(tensor_type)
    COMPARE_FIELD(sequence_type)
    COMPARE_FIELD(map_type)
    COMPARE_FIELD(opaque_type)
    COMPARE_FIELD(denotation)
    COMPARE_FIELD(sparse_tensor_type)
    COMPARE_FIELD(optional_type)
    COMPARE_FIELD(struct_type)
    return true;
  }

  bool Compare(const TypeProto::Tensor &left, const TypeProto::Tensor &right) {
    COMPARE_FIELD(elem_type)
    COMPARE_FIELD(shape)
    return true;
  }

  bool Compare(const TypeProto::SparseTensor &left, const TypeProto::SparseTensor &right) {
    COMPARE_FIELD(elem_type)
    COMPARE_FIELD(shape)
    return true;
  }

  bool Compare(const TypeProto::Sequence &left, const TypeProto::Sequence &right) {
    COMPARE_FIELD(elem_type)
    return true;
  }

  bool Compare(const TypeProto::Optional &left, const TypeProto::Optional &right) {
    COMPARE_FIELD(elem_type)
    return true;
  }

  bool Compare(const TypeProto::Map &left, const TypeProto::Map &right) {
    COMPARE_FIELD(key_type)
    COMPARE_FIELD(value_type)
    return true;
  }

  bool Compare(const TypeProto::Opaque &left, const TypeProto::Opaque &right) {
    COMPARE_FIELD(domain)
    COMPARE_FIELD(name)
    return true;
  }

  bool Compare(const TensorShapeProto &left, const TensorShapeProto &right) {
    COMPARE_FIELD(dim)
    return true;
  }

  bool Compare(const TensorShapeProto::Dimension &left, const TensorShapeProto::Dimension &right) {
    COMPARE_FIELD(dim_value)
    COMPARE_FIELD(dim_param)
    COMPARE_FIELD(denotation)
    return true;
  }

  bool Compare(const StructTypeProto &left, const StructTypeProto &right) {
    COMPARE_FIELD(array)
    COMPARE_FIELD(structure)
    COMPARE_FIELD(bit_packing)
    COMPARE_FIELD(type_ref)
    COMPARE_FIELD(decoder)
    COMPARE_FIELD(encoder)
    COMPARE_FIELD(name)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(metadata_props)
    COMPARE_FIELD(type_id)
    return true;
  }

  bool Compare(const StructTypeProto::Array &left, const StructTypeProto::Array &right) {
    COMPARE_FIELD(element_type)
    COMPARE_FIELD(dimension)
    return true;
  }

  bool Compare(const StructTypeProto::Structure &left, const StructTypeProto::Structure &right) {
    COMPARE_FIELD(field)
    return true;
  }

  bool Compare(const StructTypeProto::Structure::Field &left,
               const StructTypeProto::Structure::Field &right) {
    COMPARE_FIELD(name)
    COMPARE_FIELD(type)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(constant)
    return true;
  }

  bool Compare(const StructTypeProto::BitPacking &left, const StructTypeProto::BitPacking &right) {
    COMPARE_FIELD(component)
    COMPARE_FIELD(dimension)
    return true;
  }

  bool Compare(const StructTypeProto::BitPacking::Component &left,
               const StructTypeProto::BitPacking::Component &right) {
    COMPARE_FIELD(name)
    COMPARE_FIELD(bit_width)
    return true;
  }

  bool Compare(const StringStringEntryProto &left, const StringStringEntryProto &right) {
    COMPARE_FIELD(key)
    COMPARE_FIELD(value)
    return true;
  }

  bool Compare(const TensorProto &left, const TensorProto &right) {
    COMPARE_FIELD(dims)
    COMPARE_FIELD(data_type)
    COMPARE_FIELD(segment)
    COMPARE_FIELD(data_location)
    COMPARE_FIELD(name)
    COMPARE_FIELD(raw_data)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(external_data)
    COMPARE_FIELD(metadata_props)
    COMPARE_FIELD(double_data)
    COMPARE_FIELD(float_data)
    COMPARE_FIELD(int32_data)
    COMPARE_FIELD(int64_data)
    COMPARE_FIELD(uint64_data)
    COMPARE_FIELD(string_data)
    return true;
  }

  bool Compare(const TensorProto::Segment &left, const TensorProto::Segment &right) {
    COMPARE_FIELD(begin)
    COMPARE_FIELD(end)
    return true;
  }

  bool Compare(const SparseTensorProto &left, const SparseTensorProto &right) {
    COMPARE_FIELD(values)
    COMPARE_FIELD(indices)
    COMPARE_FIELD(dims)
    return true;
  }

  bool Compare(const FunctionProto &left, const FunctionProto &right) {
    COMPARE_FIELD(name)
    COMPARE_FIELD(input)
    COMPARE_FIELD(output)
    COMPARE_FIELD(attribute)
    COMPARE_FIELD(attribute_proto)
    COMPARE_FIELD(node)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(opset_import)
    COMPARE_FIELD(domain)
    COMPARE_FIELD(overload)
    COMPARE_FIELD(value_info)
    COMPARE_FIELD(metadata_props)
    return true;
  }

  bool Compare(const OperatorSetIdProto &left, const OperatorSetIdProto &right) {
    COMPARE_FIELD(domain)
    COMPARE_FIELD(version)
    return true;
  }

  bool Compare(const ValueInfoProto &left, const ValueInfoProto &right) {
    COMPARE_FIELD(name)
    COMPARE_FIELD(type)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(metadata_props)
    return true;
  }

  bool Compare(const NodeProto &left, const NodeProto &right) {
    COMPARE_FIELD(input)
    COMPARE_FIELD(output)
    COMPARE_FIELD(name)
    COMPARE_FIELD(op_type)
    COMPARE_FIELD(attribute)
    COMPARE_FIELD(domain)
    COMPARE_FIELD(overload)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(metadata_props)
    COMPARE_FIELD(device_configurations)
    return true;
  }

  bool Compare(const AttributeProto &left, const AttributeProto &right) {
    COMPARE_FIELD(name)
    COMPARE_FIELD(ref_attr_name)
    COMPARE_FIELD(type)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(f)
    COMPARE_FIELD(i)
    COMPARE_FIELD(s)
    COMPARE_FIELD(t)
    COMPARE_FIELD(sparse_tensor)
    COMPARE_FIELD(g)
    COMPARE_FIELD(tp)
    COMPARE_FIELD(floats)
    COMPARE_FIELD(ints)
    COMPARE_FIELD(strings)
    COMPARE_FIELD(tensors)
    COMPARE_FIELD(sparse_tensors)
    COMPARE_FIELD(graphs)
    COMPARE_FIELD(type_protos)
    return true;
  }

  bool Compare(const GraphProto &left, const GraphProto &right) {
    COMPARE_FIELD(node)
    COMPARE_FIELD(name)
    COMPARE_FIELD(initializer)
    COMPARE_FIELD(sparse_initializer)
    COMPARE_FIELD(doc_string)
    COMPARE_FIELD(input)
    COMPARE_FIELD(output)
    COMPARE_FIELD(value_info)
    COMPARE_FIELD(quantization_annotation)
    COMPARE_FIELD(metadata_props)
    COMPARE_FIELD(encoded_initializer)
    COMPARE_FIELD(persistent_bindings)
    return true;
  }

  bool Compare(const TensorAnnotation &left, const TensorAnnotation &right) {
    COMPARE_FIELD(tensor_name)
    COMPARE_FIELD(quant_parameter_tensor_names)
    return true;
  }

  bool Compare(const PersistentBindingProto &left, const PersistentBindingProto &right) {
    COMPARE_FIELD(input_name)
    COMPARE_FIELD(output_name)
    return true;
  }

  bool Compare(const EncodedValueProto &left, const EncodedValueProto &right) {
    COMPARE_FIELD(affine)
    COMPARE_FIELD(struct_type)
    COMPARE_FIELD(logical_type)
    COMPARE_FIELD(raw_data)
    COMPARE_FIELD(external_data)
    COMPARE_FIELD(data_location)
    COMPARE_FIELD(name)
    COMPARE_FIELD(doc_string)
    return true;
  }

  bool Compare(const AffineLayoutProto &left, const AffineLayoutProto &right) {
    COMPARE_FIELD(storage_type)
    COMPARE_FIELD(scale)
    COMPARE_FIELD(zero_point)
    COMPARE_FIELD(axis)
    COMPARE_FIELD(block_size)
    return true;
  }

  bool Compare(const NodeDeviceConfigurationProto &left,
               const NodeDeviceConfigurationProto &right) {
    COMPARE_FIELD(configuration_id)
    COMPARE_FIELD(sharding_spec)
    COMPARE_FIELD(pipeline_stage)
    return true;
  }

  bool Compare(const ShardingSpecProto &left, const ShardingSpecProto &right) {
    COMPARE_FIELD(tensor_name)
    COMPARE_FIELD(device)
    COMPARE_FIELD(index_to_device_group_map)
    COMPARE_FIELD(sharded_dim)
    return true;
  }

  bool Compare(const IntIntListEntryProto &left, const IntIntListEntryProto &right) {
    COMPARE_FIELD(key)
    COMPARE_FIELD(value)
    return true;
  }

  bool Compare(const ShardedDimProto &left, const ShardedDimProto &right) {
    COMPARE_FIELD(axis)
    COMPARE_FIELD(simple_sharding)
    return true;
  }

  bool Compare(const SimpleShardedDimProto &left, const SimpleShardedDimProto &right) {
    COMPARE_FIELD(dim_value)
    COMPARE_FIELD(dim_param)
    COMPARE_FIELD(num_shards)
    return true;
  }

private:
  bool Fail(const std::string &reason) {
    if (difference_)
      *difference_ = reason;
    return false;
  }

  void Prefix(const std::string &field) {
    if (difference_)
      *difference_ = field +
                     (difference_->starts_with("[") || difference_->starts_with(":") ? "" : ".") +
                     *difference_;
  }

  std::string *difference_;
};

#undef COMPARE_FIELD

} // namespace

bool EqualProto(const TypeProto &left, const TypeProto &right, std::string *difference) {
  return ProtoComparison(difference).Compare(left, right);
}

bool EqualProto(const StructTypeProto &left, const StructTypeProto &right,
                std::string *difference) {
  return ProtoComparison(difference).Compare(left, right);
}

bool EqualProto(const EncodedValueProto &left, const EncodedValueProto &right,
                std::string *difference) {
  return ProtoComparison(difference).Compare(left, right);
}

} // namespace ONNX_LIGHT_NAMESPACE

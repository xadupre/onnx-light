// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx.h"

#include <cstring>
#include <type_traits>

namespace ONNX_LIGHT_NAMESPACE {
namespace {

// Checks presence before accessing optional submessages and adds field names only on failure.
#define COMPARE_FIELD(name)                                                                        \
  if (left.has_##name() != right.has_##name()) {                                                   \
    return comparison.Fail(#name ": presence differs");                                            \
  }                                                                                                \
  if (left.has_##name() && !comparison.Compare(left.ref_##name(), right.ref_##name())) {           \
    comparison.Prefix(#name);                                                                      \
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

  template <typename T>
    requires(std::is_base_of_v<Message, T>)
  bool Compare(const T &left, const T &right) {
    return left.Equals(right, difference_);
  }

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

} // namespace

bool TypeProto::Equals(const TypeProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool TypeProto::Tensor::Equals(const TypeProto::Tensor &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(elem_type)
  COMPARE_FIELD(shape)
  return true;
}

bool TypeProto::SparseTensor::Equals(const TypeProto::SparseTensor &right,
                                     std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(elem_type)
  COMPARE_FIELD(shape)
  return true;
}

bool TypeProto::Sequence::Equals(const TypeProto::Sequence &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(elem_type)
  return true;
}

bool TypeProto::Optional::Equals(const TypeProto::Optional &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(elem_type)
  return true;
}

bool TypeProto::Map::Equals(const TypeProto::Map &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(key_type)
  COMPARE_FIELD(value_type)
  return true;
}

bool TypeProto::Opaque::Equals(const TypeProto::Opaque &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(domain)
  COMPARE_FIELD(name)
  return true;
}

bool TensorShapeProto::Equals(const TensorShapeProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(dim)
  return true;
}

bool TensorShapeProto::Dimension::Equals(const TensorShapeProto::Dimension &right,
                                         std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(dim_value)
  COMPARE_FIELD(dim_param)
  COMPARE_FIELD(denotation)
  return true;
}

bool StructTypeProto::Equals(const StructTypeProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool StructTypeProto::Array::Equals(const StructTypeProto::Array &right,
                                    std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(element_type)
  COMPARE_FIELD(dimension)
  return true;
}

bool StructTypeProto::Structure::Equals(const StructTypeProto::Structure &right,
                                        std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(field)
  return true;
}

bool StructTypeProto::Structure::Field::Equals(const StructTypeProto::Structure::Field &right,
                                               std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(type)
  COMPARE_FIELD(doc_string)
  COMPARE_FIELD(constant)
  return true;
}

bool StructTypeProto::BitPacking::Equals(const StructTypeProto::BitPacking &right,
                                         std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(component)
  COMPARE_FIELD(dimension)
  return true;
}

bool StructTypeProto::BitPacking::Component::Equals(
    const StructTypeProto::BitPacking::Component &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(bit_width)
  return true;
}

bool StringStringEntryProto::Equals(const StringStringEntryProto &right,
                                    std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(key)
  COMPARE_FIELD(value)
  return true;
}

bool TensorProto::Equals(const TensorProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool TensorProto::Segment::Equals(const TensorProto::Segment &right,
                                  std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(begin)
  COMPARE_FIELD(end)
  return true;
}

bool SparseTensorProto::Equals(const SparseTensorProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(values)
  COMPARE_FIELD(indices)
  COMPARE_FIELD(dims)
  return true;
}

bool FunctionProto::Equals(const FunctionProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool OperatorSetIdProto::Equals(const OperatorSetIdProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(domain)
  COMPARE_FIELD(version)
  return true;
}

bool ValueInfoProto::Equals(const ValueInfoProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(type)
  COMPARE_FIELD(doc_string)
  COMPARE_FIELD(metadata_props)
  return true;
}

bool NodeProto::Equals(const NodeProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool AttributeProto::Equals(const AttributeProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool GraphProto::Equals(const GraphProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool TensorAnnotation::Equals(const TensorAnnotation &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(tensor_name)
  COMPARE_FIELD(quant_parameter_tensor_names)
  return true;
}

bool PersistentBindingProto::Equals(const PersistentBindingProto &right,
                                    std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(input_name)
  COMPARE_FIELD(output_name)
  return true;
}

bool EncodedValueProto::Equals(const EncodedValueProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
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

bool AffineLayoutProto::Equals(const AffineLayoutProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(storage_type)
  COMPARE_FIELD(scale)
  COMPARE_FIELD(zero_point)
  COMPARE_FIELD(axis)
  COMPARE_FIELD(block_size)
  return true;
}

bool NodeDeviceConfigurationProto::Equals(const NodeDeviceConfigurationProto &right,
                                          std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(configuration_id)
  COMPARE_FIELD(sharding_spec)
  COMPARE_FIELD(pipeline_stage)
  return true;
}

bool ShardingSpecProto::Equals(const ShardingSpecProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(tensor_name)
  COMPARE_FIELD(device)
  COMPARE_FIELD(index_to_device_group_map)
  COMPARE_FIELD(sharded_dim)
  return true;
}

bool IntIntListEntryProto::Equals(const IntIntListEntryProto &right,
                                  std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(key)
  COMPARE_FIELD(value)
  return true;
}

bool ShardedDimProto::Equals(const ShardedDimProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(axis)
  COMPARE_FIELD(simple_sharding)
  return true;
}

bool SimpleShardedDimProto::Equals(const SimpleShardedDimProto &right,
                                   std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(dim_value)
  COMPARE_FIELD(dim_param)
  COMPARE_FIELD(num_shards)
  return true;
}

bool DeviceConfigurationProto::Equals(const DeviceConfigurationProto &right,
                                      std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(num_devices)
  COMPARE_FIELD(device)
  return true;
}

bool ModelProto::Equals(const ModelProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(ir_version)
  COMPARE_FIELD(opset_import)
  COMPARE_FIELD(producer_name)
  COMPARE_FIELD(producer_version)
  COMPARE_FIELD(domain)
  COMPARE_FIELD(model_version)
  COMPARE_FIELD(doc_string)
  COMPARE_FIELD(graph)
  COMPARE_FIELD(metadata_props)
  COMPARE_FIELD(functions)
  COMPARE_FIELD(configuration)
  COMPARE_FIELD(struct_types)
  return true;
}

bool SequenceProto::Equals(const SequenceProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(elem_type)
  COMPARE_FIELD(tensor_values)
  COMPARE_FIELD(sparse_tensor_values)
  COMPARE_FIELD(sequence_values)
  COMPARE_FIELD(map_values)
  COMPARE_FIELD(optional_values)
  return true;
}

bool MapProto::Equals(const MapProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(key_type)
  COMPARE_FIELD(keys)
  COMPARE_FIELD(string_keys)
  COMPARE_FIELD(values)
  return true;
}

bool OptionalProto::Equals(const OptionalProto &right, std::string *difference) const {
  const auto &left = *this;
  ProtoComparison comparison(difference);
  COMPARE_FIELD(name)
  COMPARE_FIELD(elem_type)
  COMPARE_FIELD(tensor_value)
  COMPARE_FIELD(sparse_tensor_value)
  COMPARE_FIELD(sequence_value)
  COMPARE_FIELD(map_value)
  COMPARE_FIELD(optional_value)
  return true;
}

#undef COMPARE_FIELD

} // namespace ONNX_LIGHT_NAMESPACE

#include "onnx.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

template <class Type> void CreateDims(Type &proto, int num_dims) {
  auto &shape = proto.ref_shape();
  shape.ref_dim().clear();
  for (int i = 0; i < num_dims; ++i) {
    shape.add_dim();
  }
}

template <class Type> void SetDimValues(Type &proto, const std::vector<int64_t> &values) {
  auto &dims = proto.ref_shape().ref_dim();
  ASSERT_EQ(dims.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] != -1) {
      dims[i].set_dim_value(values[i]);
    }
  }
}

template <class Type>
void SetDimParams(Type &proto, const std::vector<const std::string *> &values) {
  auto &dims = proto.ref_shape().ref_dim();
  ASSERT_EQ(dims.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    if (values[i] != nullptr) {
      dims[i].set_dim_param(*values[i]);
    }
  }
}

void MergeDimensionInfo(const TensorShapeProto::Dimension &source,
                        TensorShapeProto::Dimension &target, size_t index) {
  if (source.has_dim_value()) {
    if (target.has_dim_value() && target.ref_dim_value() != source.ref_dim_value()) {
      throw std::runtime_error("Mismatch between dimension values at index " +
                               std::to_string(index));
    }
    target.set_dim_value(source.ref_dim_value());
    return;
  }
  if (!source.has_dim_param()) {
    return;
  }
  if (!target.has_dim_value() && !target.has_dim_param()) {
    target.set_dim_param(source.ref_dim_param());
  }
}

template <class Type> void MergeInShapeInfo(const Type &source, Type &target) {
  if (!source.has_shape()) {
    return;
  }
  if (!target.has_shape()) {
    target.ref_shape() = source.ref_shape();
    return;
  }
  const auto &source_dims = source.ref_shape().ref_dim();
  auto &target_dims = target.ref_shape().ref_dim();
  if (source_dims.size() != target_dims.size()) {
    throw std::runtime_error("Mismatch between number of inferred and declared dimensions");
  }
  for (size_t i = 0; i < source_dims.size(); ++i) {
    MergeDimensionInfo(source_dims[i], target_dims[i], i);
  }
}

} // namespace

TEST(onnx_shape_inference, mergeShapeInfo_HasShape) {
  TypeProto::Tensor source_tensor;
  TypeProto::Tensor target_tensor;
  CreateDims(source_tensor, 1);
  SetDimValues(source_tensor, {1});
  MergeInShapeInfo(source_tensor, target_tensor);
  ASSERT_EQ(target_tensor.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target_tensor.ref_shape().ref_dim()[0].ref_dim_value(), 1);

  TypeProto::SparseTensor source_sparse;
  TypeProto::SparseTensor target_sparse;
  CreateDims(source_sparse, 1);
  SetDimValues(source_sparse, {1});
  MergeInShapeInfo(source_sparse, target_sparse);
  ASSERT_EQ(target_sparse.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target_sparse.ref_shape().ref_dim()[0].ref_dim_value(), 1);
}

TEST(onnx_shape_inference, mergeShapeInfo_PreferValueOverParam) {
  const std::string param = "A";
  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 1);
  SetDimValues(source, {1});
  CreateDims(target, 1);
  SetDimParams(target, {&param});
  MergeInShapeInfo(source, target);
  ASSERT_EQ(target.ref_shape().ref_dim().size(), 1u);
  EXPECT_TRUE(target.ref_shape().ref_dim()[0].has_dim_value());
  EXPECT_EQ(target.ref_shape().ref_dim()[0].ref_dim_value(), 1);
}

TEST(onnx_shape_inference, mergeShapeInfo_SourceParamCopiedWhenTargetUnknown) {
  const std::string param = "A";
  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 1);
  SetDimParams(source, {&param});
  CreateDims(target, 1);
  MergeInShapeInfo(source, target);
  ASSERT_EQ(target.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target.ref_shape().ref_dim()[0].ref_dim_param(), "A");
}

TEST(onnx_shape_inference, mergeShapeInfo_CombineShapes) {
  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 2);
  SetDimValues(source, {-1, 2});
  CreateDims(target, 2);
  SetDimValues(target, {1, -1});
  MergeInShapeInfo(source, target);
  ASSERT_EQ(target.ref_shape().ref_dim().size(), 2u);
  EXPECT_EQ(target.ref_shape().ref_dim()[0].ref_dim_value(), 1);
  EXPECT_EQ(target.ref_shape().ref_dim()[1].ref_dim_value(), 2);
}

TEST(onnx_shape_inference, mergeShapeInfo_Mismatches) {
  TypeProto::Tensor rank_source;
  TypeProto::Tensor rank_target;
  CreateDims(rank_source, 2);
  SetDimValues(rank_source, {-1, 2});
  CreateDims(rank_target, 3);
  SetDimValues(rank_target, {1, -1, 1});
  EXPECT_THROW(MergeInShapeInfo(rank_source, rank_target), std::runtime_error);

  TypeProto::Tensor source;
  TypeProto::Tensor target;
  CreateDims(source, 2);
  SetDimValues(source, {2, 2});
  CreateDims(target, 2);
  SetDimValues(target, {2, 1});
  EXPECT_THROW(MergeInShapeInfo(source, target), std::runtime_error);

  const std::string param_a = "A";
  const std::string param_b = "B";
  TypeProto::SparseTensor source_sparse;
  TypeProto::SparseTensor target_sparse;
  CreateDims(source_sparse, 1);
  SetDimParams(source_sparse, {&param_a});
  CreateDims(target_sparse, 1);
  SetDimParams(target_sparse, {&param_b});
  MergeInShapeInfo(source_sparse, target_sparse);
  ASSERT_EQ(target_sparse.ref_shape().ref_dim().size(), 1u);
  EXPECT_EQ(target_sparse.ref_shape().ref_dim()[0].ref_dim_param(), "B");
}

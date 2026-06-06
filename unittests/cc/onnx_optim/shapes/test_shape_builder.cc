// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0
//
// C++ translation of
// https://github.com/xadupre/yet-another-onnx-builder/blob/main/unittests/xshape/test_shape_builder.py
//
// Exercises ``ComputeShapeModel`` / ``ApplyInferredShapesToGraph`` on small
// graphs that mirror the Python ``BasicShapeBuilder.run_model`` tests, and
// verifies that the shape-inference helper functions (``GetAttributeOr``,
// ``GetAttributeInts``, ``FindAttribute``) behave consistently with the
// Python ``get_attribute_with_default`` / ``get_attributes_with_default``
// wrappers.

#include "onnx_optim/shapes/shape_inference.h"

#include "onnx_optim/optim_tensor.h"
#include "onnx_optim/shapes/shapes_context.h"
#include "onnx_proto/onnx.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

namespace {

// ── proto-building helpers ──────────────────────────────────────────────────

NodeProto MakeNode(const std::string &op_type, const std::vector<std::string> &inputs,
                   const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type(op_type);
  for (const auto &in : inputs) {
    node.add_input(in);
  }
  for (const auto &out : outputs) {
    node.add_output(out);
  }
  return node;
}

// Adds an int64 1-D initializer to ``graph``.
void AddInt64Initializer(GraphProto &graph, const std::string &name,
                         const std::vector<int64_t> &values) {
  TensorProto *init = graph.add_initializer();
  init->set_name(name);
  init->set_data_type(static_cast<int>(TensorProto::DataType::INT64));
  init->add_dims(std::vector<uint64_t>{values.size()});
  init->add_int64_data(values);
}

// Adds a float input to ``graph`` whose shape mixes concrete and symbolic
// dimensions. ``shape[i] >= 0`` is treated as a concrete ``dim_value``;
// ``shape[i] < 0`` picks ``symbolic_names[i]`` as the ``dim_param`` (or
// ``"?"`` when out of range).
void AddFloatInput(GraphProto &graph, const std::string &name, const std::vector<int64_t> &shape,
                   const std::vector<std::string> &symbolic_names = {}) {
  ValueInfoProto *vi = graph.add_input();
  vi->set_name(name);
  TypeProto *tp = vi->add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    TensorShapeProto::Dimension *d = sp->add_dim();
    if (shape[i] < 0) {
      d->set_dim_param(i < symbolic_names.size() ? symbolic_names[i] : std::string("?"));
    } else {
      d->set_dim_value(shape[i]);
    }
  }
}

// Adds a float output to ``graph`` with the same shape-encoding convention as
// ``AddFloatInput``.
void AddFloatOutput(GraphProto &graph, const std::string &name, const std::vector<int64_t> &shape,
                    const std::vector<std::string> &symbolic_names = {}) {
  ValueInfoProto *vi = graph.add_output();
  vi->set_name(name);
  TypeProto *tp = vi->add_type();
  TypeProto::Tensor *tt = tp->add_tensor_type();
  tt->set_elem_type(static_cast<int>(TensorProto::DataType::FLOAT));
  TensorShapeProto *sp = tt->add_shape();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    TensorShapeProto::Dimension *d = sp->add_dim();
    if (shape[i] < 0) {
      d->set_dim_param(i < symbolic_names.size() ? symbolic_names[i] : std::string("?"));
    } else {
      d->set_dim_value(shape[i]);
    }
  }
}

// Checks that ``ctx`` contains ``name``, that its dtype matches ``expected``,
// and that every dimension in ``shape`` equals the corresponding ``OptimDim``.
void CheckShape(const onnx_optim::shapes::ShapesContext &ctx, const std::string &name,
                const onnx_optim::OptimShape &expected_shape,
                onnx_optim::TensorType expected_dtype) {
  ASSERT_TRUE(ctx.Has(name)) << "missing tensor: " << name;
  EXPECT_EQ(ctx.Get(name).Dtype(), expected_dtype) << "dtype mismatch for " << name;
  ASSERT_EQ(ctx.Get(name).Shape().Rank(), expected_shape.Rank()) << "rank mismatch for " << name;
  for (std::size_t i = 0; i < expected_shape.Rank(); ++i) {
    EXPECT_EQ(ctx.Get(name).Shape()[i], expected_shape[i])
        << "dim[" << i << "] mismatch for " << name;
  }
}

// Checks that dim ``idx`` of tensor ``name`` in ``ctx`` is a concrete integer
// equal to ``value``.
void CheckConcreteDim(const onnx_optim::shapes::ShapesContext &ctx, const std::string &name,
                      std::size_t idx, int64_t value) {
  ASSERT_TRUE(ctx.Has(name)) << "missing tensor: " << name;
  const onnx_optim::OptimShape &s = ctx.Get(name).Shape();
  ASSERT_LT(idx, s.Rank()) << "dim index out of range for " << name;
  ASSERT_TRUE(s[idx].IsInt()) << "dim[" << idx << "] of " << name << " is not concrete";
  EXPECT_EQ(s[idx].AsInt(), value) << "dim[" << idx << "] of " << name;
}

// Checks that dim ``idx`` of tensor ``name`` in ``ctx`` is a symbolic
// expression equal to ``expr``.
void CheckSymbolicDim(const onnx_optim::shapes::ShapesContext &ctx, const std::string &name,
                      std::size_t idx, const std::string &expr) {
  ASSERT_TRUE(ctx.Has(name)) << "missing tensor: " << name;
  const onnx_optim::OptimShape &s = ctx.Get(name).Shape();
  ASSERT_LT(idx, s.Rank()) << "dim index out of range for " << name;
  ASSERT_TRUE(s[idx].IsExpr()) << "dim[" << idx << "] of " << name << " is not symbolic";
  EXPECT_EQ(s[idx].AsExpr(), expr) << "dim[" << idx << "] of " << name;
}

// Checks that dim ``idx`` of tensor ``name`` in ``ctx`` is symbolic (any
// expression).
void CheckIsSymbolic(const onnx_optim::shapes::ShapesContext &ctx, const std::string &name,
                     std::size_t idx) {
  ASSERT_TRUE(ctx.Has(name)) << "missing tensor: " << name;
  const onnx_optim::OptimShape &s = ctx.Get(name).Shape();
  ASSERT_LT(idx, s.Rank()) << "dim index out of range for " << name;
  EXPECT_TRUE(s[idx].IsExpr()) << "dim[" << idx << "] of " << name << " should be symbolic";
}

} // namespace

// ── test_check_shape ────────────────────────────────────────────────────────
//
// Translation of ``TestShapeBuilder.test_check_shape``.
//
// Graph:
//   xu1    = Unsqueeze(X,     zero)    # zero=int64[1]=[0]
//   xu2    = Unsqueeze(xu1,   un)      # un=int64[1]=[1]
//   xm1    = Reshape(xu2,    shape1)   # shape1=[1,32,128]
//   xm2c   = Reshape(Y,      shape2)   # shape2=[15,128,64]
//   xm2    = Cast(xm2c, to=1)
//   xm     = MatMul(xm1,     xm2)
//   Z      = Reshape(xm,     shape3)   # shape3=[3,5,32,64]
//
// Inputs:
//   X: float[D32, D128]
//   Y: float[batch, channel, D128, D64]
// Output:
//   Z: float[batch, channel, D32, 64]   (declared; inferred = [3,5,32,64])
TEST(OnnxOptimShapeBuilder, CheckShapeComputesExpectedRankTypesAndConcreteDims) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(21);

  GraphProto *graph = model.add_graph();
  graph->set_name("check_shape");

  // inputs
  AddFloatInput(*graph, "X", {-1, -1}, {"D32", "D128"});
  AddFloatInput(*graph, "Y", {-1, -1, -1, -1}, {"batch", "channel", "D128", "D64"});

  // output (declared shape is advisory; the test uses inferred shapes)
  AddFloatOutput(*graph, "Z", {-1, -1, -1, 64}, {"batch", "channel", "D32"});

  // initializers
  AddInt64Initializer(*graph, "zero", {0});
  AddInt64Initializer(*graph, "un", {1});
  AddInt64Initializer(*graph, "shape1", {1, 32, 128});
  AddInt64Initializer(*graph, "shape2", {15, 128, 64});
  AddInt64Initializer(*graph, "shape3", {3, 5, 32, 64});

  // nodes
  *graph->add_node() = MakeNode("Unsqueeze", {"X", "zero"}, {"xu1"});
  *graph->add_node() = MakeNode("Unsqueeze", {"xu1", "un"}, {"xu2"});
  *graph->add_node() = MakeNode("Reshape", {"xu2", "shape1"}, {"xm1"});
  *graph->add_node() = MakeNode("Reshape", {"Y", "shape2"}, {"xm2c"});
  {
    NodeProto cast_node = MakeNode("Cast", {"xm2c"}, {"xm2"});
    AddAttribute<int64_t>(cast_node, "to", static_cast<int64_t>(TensorProto::DataType::FLOAT));
    *graph->add_node() = std::move(cast_node);
  }
  *graph->add_node() = MakeNode("MatMul", {"xm1", "xm2"}, {"xm"});
  *graph->add_node() = MakeNode("Reshape", {"xm", "shape3"}, {"Z"});

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  // ── initializer ranks / types ────────────────────────────────────────────
  CheckShape(ctx, "zero", {onnx_optim::OptimDim(1)}, onnx_optim::TensorType::kInt64);
  CheckShape(ctx, "un", {onnx_optim::OptimDim(1)}, onnx_optim::TensorType::kInt64);
  CheckShape(ctx, "shape1", {onnx_optim::OptimDim(3)}, onnx_optim::TensorType::kInt64);
  CheckShape(ctx, "shape2", {onnx_optim::OptimDim(3)}, onnx_optim::TensorType::kInt64);
  CheckShape(ctx, "shape3", {onnx_optim::OptimDim(4)}, onnx_optim::TensorType::kInt64);

  // ── initializer ValueAsShape ─────────────────────────────────────────────
  // int64 1-D initializers with < 8 elements get a ValueAsShape annotation.
  ASSERT_TRUE(ctx.Get("zero").HasValueAsShape());
  ASSERT_EQ(ctx.Get("zero").ValueAsShape().Rank(), 1u);
  EXPECT_EQ(ctx.Get("zero").ValueAsShape()[0], onnx_optim::OptimDim(int64_t{0}));

  ASSERT_TRUE(ctx.Get("un").HasValueAsShape());
  ASSERT_EQ(ctx.Get("un").ValueAsShape().Rank(), 1u);
  EXPECT_EQ(ctx.Get("un").ValueAsShape()[0], onnx_optim::OptimDim(1));

  ASSERT_TRUE(ctx.Get("shape1").HasValueAsShape());
  {
    const onnx_optim::OptimShape &vas = ctx.Get("shape1").ValueAsShape();
    ASSERT_EQ(vas.Rank(), 3u);
    EXPECT_EQ(vas[0], onnx_optim::OptimDim(1));
    EXPECT_EQ(vas[1], onnx_optim::OptimDim(32));
    EXPECT_EQ(vas[2], onnx_optim::OptimDim(128));
  }

  ASSERT_TRUE(ctx.Get("shape2").HasValueAsShape());
  {
    const onnx_optim::OptimShape &vas = ctx.Get("shape2").ValueAsShape();
    ASSERT_EQ(vas.Rank(), 3u);
    EXPECT_EQ(vas[0], onnx_optim::OptimDim(15));
    EXPECT_EQ(vas[1], onnx_optim::OptimDim(128));
    EXPECT_EQ(vas[2], onnx_optim::OptimDim(64));
  }

  ASSERT_TRUE(ctx.Get("shape3").HasValueAsShape());
  {
    const onnx_optim::OptimShape &vas = ctx.Get("shape3").ValueAsShape();
    ASSERT_EQ(vas.Rank(), 4u);
    EXPECT_EQ(vas[0], onnx_optim::OptimDim(3));
    EXPECT_EQ(vas[1], onnx_optim::OptimDim(5));
    EXPECT_EQ(vas[2], onnx_optim::OptimDim(32));
    EXPECT_EQ(vas[3], onnx_optim::OptimDim(64));
  }

  // ── input shapes ─────────────────────────────────────────────────────────
  // X: float[D32, D128]
  ASSERT_TRUE(ctx.Has("X"));
  ASSERT_EQ(ctx.Get("X").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("X").Dtype(), onnx_optim::TensorType::kFloat);
  CheckSymbolicDim(ctx, "X", 0, "D32");
  CheckSymbolicDim(ctx, "X", 1, "D128");

  // Y: float[batch, channel, D128, D64]
  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 4u);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);
  CheckSymbolicDim(ctx, "Y", 0, "batch");
  CheckSymbolicDim(ctx, "Y", 1, "channel");
  CheckSymbolicDim(ctx, "Y", 2, "D128");
  CheckSymbolicDim(ctx, "Y", 3, "D64");

  // ── intermediate shapes ──────────────────────────────────────────────────
  // xu1 = Unsqueeze(X, zero=0) → (1, D32, D128)
  ASSERT_TRUE(ctx.Has("xu1"));
  ASSERT_EQ(ctx.Get("xu1").Shape().Rank(), 3u);
  EXPECT_EQ(ctx.Get("xu1").Dtype(), onnx_optim::TensorType::kFloat);
  CheckConcreteDim(ctx, "xu1", 0, 1);
  CheckSymbolicDim(ctx, "xu1", 1, "D32");
  CheckSymbolicDim(ctx, "xu1", 2, "D128");

  // xu2 = Unsqueeze(xu1, un=1) → (1, 1, D32, D128)
  ASSERT_TRUE(ctx.Has("xu2"));
  ASSERT_EQ(ctx.Get("xu2").Shape().Rank(), 4u);
  EXPECT_EQ(ctx.Get("xu2").Dtype(), onnx_optim::TensorType::kFloat);
  CheckConcreteDim(ctx, "xu2", 0, 1);
  CheckConcreteDim(ctx, "xu2", 1, 1);
  CheckSymbolicDim(ctx, "xu2", 2, "D32");
  CheckSymbolicDim(ctx, "xu2", 3, "D128");

  // xm1 = Reshape(xu2, shape1=[1,32,128]) → (1, 32, 128)
  CheckShape(ctx, "xm1",
             {onnx_optim::OptimDim(1), onnx_optim::OptimDim(32), onnx_optim::OptimDim(128)},
             onnx_optim::TensorType::kFloat);

  // xm2c = Reshape(Y, shape2=[15,128,64]) → (15, 128, 64)
  CheckShape(ctx, "xm2c",
             {onnx_optim::OptimDim(15), onnx_optim::OptimDim(128), onnx_optim::OptimDim(64)},
             onnx_optim::TensorType::kFloat);

  // xm2 = Cast(xm2c, to=float) → (15, 128, 64), float
  CheckShape(ctx, "xm2",
             {onnx_optim::OptimDim(15), onnx_optim::OptimDim(128), onnx_optim::OptimDim(64)},
             onnx_optim::TensorType::kFloat);

  // xm = MatMul(xm1=(1,32,128), xm2=(15,128,64)) → (15, 32, 64)
  CheckShape(ctx, "xm",
             {onnx_optim::OptimDim(15), onnx_optim::OptimDim(32), onnx_optim::OptimDim(64)},
             onnx_optim::TensorType::kFloat);

  // Z = Reshape(xm, shape3=[3,5,32,64]) → (3, 5, 32, 64)
  CheckShape(ctx, "Z",
             {onnx_optim::OptimDim(3), onnx_optim::OptimDim(5), onnx_optim::OptimDim(32),
              onnx_optim::OptimDim(64)},
             onnx_optim::TensorType::kFloat);
}

// ── test_reshape_reshape ────────────────────────────────────────────────────
//
// Translation of ``TestShapeBuilder.test_reshape_reshape``.
//
// Graph:
//   xr  = Reshape(X, shape1)  # shape1=[0,0,2,-1]
//   xrr = Reshape(xr, shape2) # shape2=[0,0,-1]
//   Y   = Add(xrr, one)       # one=float[1]
//
// Input: X: float[a, b, c]
// Output: Y: float[a, b, c]
//
// In C++ the ``-1`` dimension is resolved to a symbolic placeholder
// (``Reshape_neg1_<index>``) rather than a symbolic arithmetic expression
// (``c//2``) because C++ shape inference does not perform symbolic
// arithmetic on unknown dimension products.  The test therefore checks
// ranks, concrete intermediate dims and that ``-1`` positions remain
// symbolic.
TEST(OnnxOptimShapeBuilder, ReshapeReshapePreservesRankAndPartialDims) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(21);

  GraphProto *graph = model.add_graph();
  graph->set_name("reshape_reshape");

  // inputs / outputs
  AddFloatInput(*graph, "X", {-1, -1, -1}, {"a", "b", "c"});
  AddFloatOutput(*graph, "Y", {-1, -1, -1}, {"a", "b", "c"});

  // initializers
  AddInt64Initializer(*graph, "shape1", {0, 0, 2, -1});
  AddInt64Initializer(*graph, "shape2", {0, 0, -1});
  // one: float[1] scalar  — a float initializer; add it as a raw float input
  // for simplicity (the actual data value doesn't affect shape inference).
  {
    TensorProto *one_init = graph->add_initializer();
    one_init->set_name("one");
    one_init->set_data_type(static_cast<int>(TensorProto::DataType::FLOAT));
    one_init->add_dims(std::vector<uint64_t>{1});
    one_init->add_float_data(std::vector<float>{1.0f});
  }

  // nodes
  *graph->add_node() = MakeNode("Reshape", {"X", "shape1"}, {"xr"});
  *graph->add_node() = MakeNode("Reshape", {"xr", "shape2"}, {"xrr"});
  *graph->add_node() = MakeNode("Add", {"xrr", "one"}, {"Y"});

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  // X: float[a, b, c]
  ASSERT_EQ(ctx.Get("X").Shape().Rank(), 3u);
  EXPECT_EQ(ctx.Get("X").Dtype(), onnx_optim::TensorType::kFloat);

  // xr = Reshape(X, [0,0,2,-1]) → rank 4; dim[2]=2, dim[3] is symbolic
  ASSERT_TRUE(ctx.Has("xr"));
  ASSERT_EQ(ctx.Get("xr").Shape().Rank(), 4u);
  EXPECT_EQ(ctx.Get("xr").Dtype(), onnx_optim::TensorType::kFloat);
  // dim 0 and 1 copy from input: symbolic "a" and "b"
  CheckSymbolicDim(ctx, "xr", 0, "a");
  CheckSymbolicDim(ctx, "xr", 1, "b");
  // dim 2 is the concrete 2 from the target shape
  CheckConcreteDim(ctx, "xr", 2, 2);
  // dim 3 is the -1 placeholder — symbolic (exact name may vary)
  CheckIsSymbolic(ctx, "xr", 3);

  // xrr = Reshape(xr, [0,0,-1]) → rank 3; first two dims copy from xr
  ASSERT_TRUE(ctx.Has("xrr"));
  ASSERT_EQ(ctx.Get("xrr").Shape().Rank(), 3u);
  EXPECT_EQ(ctx.Get("xrr").Dtype(), onnx_optim::TensorType::kFloat);
  CheckSymbolicDim(ctx, "xrr", 0, "a");
  CheckSymbolicDim(ctx, "xrr", 1, "b");
  // dim 2 is the -1 placeholder — symbolic
  CheckIsSymbolic(ctx, "xrr", 2);

  // Y = Add(xrr, one) — broadcast; shape matches xrr's shape
  ASSERT_TRUE(ctx.Has("Y"));
  ASSERT_EQ(ctx.Get("Y").Shape().Rank(), 3u);
  EXPECT_EQ(ctx.Get("Y").Dtype(), onnx_optim::TensorType::kFloat);

  // initializer shapes and types
  CheckShape(ctx, "shape1", {onnx_optim::OptimDim(4)}, onnx_optim::TensorType::kInt64);
  CheckShape(ctx, "shape2", {onnx_optim::OptimDim(3)}, onnx_optim::TensorType::kInt64);
  CheckShape(ctx, "one", {onnx_optim::OptimDim(1)}, onnx_optim::TensorType::kFloat);
}

// ── test_value_as_shape (ids_weight model) ──────────────────────────────────
//
// Translation of ``TestShapeBuilder.test_value_as_shape``.
//
// Graph:
//   shape    = Shape(ids_weight, start=0, end=2)   # int64[2]
//   new_shape = Concat([shape, init328], axis=0)    # int64[4]
//   A1        = MatMul(ids_weight, A)               # float[batch,seq,256]
//   B1        = MatMul(ids_weight, B)
//   C1        = MatMul(ids_weight, C)
//   Areshaped = Reshape(A1, new_shape)              # float[batch,seq,32,8]
//   Breshaped = Reshape(B1, new_shape)
//   Creshaped = Reshape(C1, new_shape)
//   At        = Transpose(Areshaped, perm=[0,2,1,3]) # float[batch,32,seq,8]
//   Bt        = Transpose(Breshaped, perm=[0,2,1,3])
//   Ct        = Transpose(Creshaped, perm=[0,2,1,3])
//
// Verifies that ``ValueAsShape`` is correctly threaded from the concrete
// initializer ``init328`` and from the symbolic ``Shape`` output through
// ``Concat`` and finally used by ``Reshape``.
TEST(OnnxOptimShapeBuilder, ValueAsShapeFromShapeConcatMatMulReshapeTranspose) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);

  GraphProto *graph = model.add_graph();
  graph->set_name("ids_weight");

  // inputs
  AddFloatInput(*graph, "ids_weight", {-1, -1, 256}, {"batch", "seq"});
  AddFloatInput(*graph, "A", {256, 256});
  AddFloatInput(*graph, "B", {256, 256});
  AddFloatInput(*graph, "C", {256, 256});

  // outputs
  AddFloatOutput(*graph, "At", {-1, 32, -1, 8}, {"batch", "seq"});
  AddFloatOutput(*graph, "Bt", {-1, 32, -1, 8}, {"batch", "seq"});
  AddFloatOutput(*graph, "Ct", {-1, 32, -1, 8}, {"batch", "seq"});

  // initializer: init328 = int64[2] = [32, 8]
  AddInt64Initializer(*graph, "init328", {32, 8});

  // nodes
  {
    NodeProto shape_node = MakeNode("Shape", {"ids_weight"}, {"shape"});
    AddAttribute<int64_t>(shape_node, "start", 0);
    AddAttribute<int64_t>(shape_node, "end", 2);
    *graph->add_node() = std::move(shape_node);
  }
  {
    NodeProto concat_node = MakeNode("Concat", {"shape", "init328"}, {"new_shape"});
    AddAttribute<int64_t>(concat_node, "axis", 0);
    *graph->add_node() = std::move(concat_node);
  }
  *graph->add_node() = MakeNode("MatMul", {"ids_weight", "A"}, {"A1"});
  *graph->add_node() = MakeNode("MatMul", {"ids_weight", "B"}, {"B1"});
  *graph->add_node() = MakeNode("MatMul", {"ids_weight", "C"}, {"C1"});
  *graph->add_node() = MakeNode("Reshape", {"A1", "new_shape"}, {"Areshaped"});
  *graph->add_node() = MakeNode("Reshape", {"B1", "new_shape"}, {"Breshaped"});
  *graph->add_node() = MakeNode("Reshape", {"C1", "new_shape"}, {"Creshaped"});
  {
    NodeProto at_node = MakeNode("Transpose", {"Areshaped"}, {"At"});
    AddAttribute<std::vector<int64_t>>(at_node, "perm", {0, 2, 1, 3});
    *graph->add_node() = std::move(at_node);
  }
  {
    NodeProto bt_node = MakeNode("Transpose", {"Breshaped"}, {"Bt"});
    AddAttribute<std::vector<int64_t>>(bt_node, "perm", {0, 2, 1, 3});
    *graph->add_node() = std::move(bt_node);
  }
  {
    NodeProto ct_node = MakeNode("Transpose", {"Creshaped"}, {"Ct"});
    AddAttribute<std::vector<int64_t>>(ct_node, "perm", {0, 2, 1, 3});
    *graph->add_node() = std::move(ct_node);
  }

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  // ── init328 ValueAsShape ─────────────────────────────────────────────────
  ASSERT_TRUE(ctx.Has("init328"));
  ASSERT_TRUE(ctx.Get("init328").HasValueAsShape());
  {
    const onnx_optim::OptimShape &vas = ctx.Get("init328").ValueAsShape();
    ASSERT_EQ(vas.Rank(), 2u);
    EXPECT_EQ(vas[0], onnx_optim::OptimDim(32));
    EXPECT_EQ(vas[1], onnx_optim::OptimDim(8));
  }

  // ── shape = Shape(ids_weight, start=0, end=2) ────────────────────────────
  // Output: int64[2], ValueAsShape = (batch, seq)
  ASSERT_TRUE(ctx.Has("shape"));
  EXPECT_EQ(ctx.Get("shape").Dtype(), onnx_optim::TensorType::kInt64);
  ASSERT_EQ(ctx.Get("shape").Shape().Rank(), 1u);
  CheckConcreteDim(ctx, "shape", 0, 2);
  ASSERT_TRUE(ctx.Get("shape").HasValueAsShape());
  {
    const onnx_optim::OptimShape &vas = ctx.Get("shape").ValueAsShape();
    ASSERT_EQ(vas.Rank(), 2u);
    EXPECT_TRUE(vas[0].IsExpr());
    EXPECT_EQ(vas[0].AsExpr(), "batch");
    EXPECT_TRUE(vas[1].IsExpr());
    EXPECT_EQ(vas[1].AsExpr(), "seq");
  }

  // ── new_shape = Concat([shape, init328], axis=0) ─────────────────────────
  // Output: int64[4], ValueAsShape = (batch, seq, 32, 8)
  ASSERT_TRUE(ctx.Has("new_shape"));
  EXPECT_EQ(ctx.Get("new_shape").Dtype(), onnx_optim::TensorType::kInt64);
  ASSERT_EQ(ctx.Get("new_shape").Shape().Rank(), 1u);
  CheckConcreteDim(ctx, "new_shape", 0, 4);
  ASSERT_TRUE(ctx.Get("new_shape").HasValueAsShape());
  {
    const onnx_optim::OptimShape &vas = ctx.Get("new_shape").ValueAsShape();
    ASSERT_EQ(vas.Rank(), 4u);
    EXPECT_TRUE(vas[0].IsExpr());
    EXPECT_EQ(vas[0].AsExpr(), "batch");
    EXPECT_TRUE(vas[1].IsExpr());
    EXPECT_EQ(vas[1].AsExpr(), "seq");
    EXPECT_EQ(vas[2], onnx_optim::OptimDim(32));
    EXPECT_EQ(vas[3], onnx_optim::OptimDim(8));
  }

  // ── MatMul outputs ───────────────────────────────────────────────────────
  // ids_weight(batch,seq,256) @ A(256,256) → A1(batch,seq,256)
  for (const std::string &name : {"A1", "B1", "C1"}) {
    ASSERT_TRUE(ctx.Has(name)) << "missing: " << name;
    ASSERT_EQ(ctx.Get(name).Shape().Rank(), 3u) << name;
    EXPECT_EQ(ctx.Get(name).Dtype(), onnx_optim::TensorType::kFloat) << name;
    CheckSymbolicDim(ctx, name, 0, "batch");
    CheckSymbolicDim(ctx, name, 1, "seq");
    CheckConcreteDim(ctx, name, 2, 256);
  }

  // ── Reshape outputs ──────────────────────────────────────────────────────
  // Reshape(A1=(batch,seq,256), new_shape.ValueAsShape=(batch,seq,32,8))
  //   → (batch, seq, 32, 8)
  for (const std::string &name : {"Areshaped", "Breshaped", "Creshaped"}) {
    ASSERT_TRUE(ctx.Has(name)) << "missing: " << name;
    ASSERT_EQ(ctx.Get(name).Shape().Rank(), 4u) << name;
    EXPECT_EQ(ctx.Get(name).Dtype(), onnx_optim::TensorType::kFloat) << name;
    CheckSymbolicDim(ctx, name, 0, "batch");
    CheckSymbolicDim(ctx, name, 1, "seq");
    CheckConcreteDim(ctx, name, 2, 32);
    CheckConcreteDim(ctx, name, 3, 8);
  }

  // ── Transpose outputs ────────────────────────────────────────────────────
  // Transpose(Areshaped=(batch,seq,32,8), perm=[0,2,1,3]) → (batch,32,seq,8)
  for (const std::string &name : {"At", "Bt", "Ct"}) {
    ASSERT_TRUE(ctx.Has(name)) << "missing: " << name;
    ASSERT_EQ(ctx.Get(name).Shape().Rank(), 4u) << name;
    EXPECT_EQ(ctx.Get(name).Dtype(), onnx_optim::TensorType::kFloat) << name;
    CheckSymbolicDim(ctx, name, 0, "batch");
    CheckConcreteDim(ctx, name, 1, 32);
    CheckSymbolicDim(ctx, name, 2, "seq");
    CheckConcreteDim(ctx, name, 3, 8);
  }
}

// ── test_evaluate_shape ─────────────────────────────────────────────────────
//
// Translation of ``TestShapeBuilder.test_evaluate_shape``.
//
// Graph: Z = Concat(X, Y, axis=1)
//   X: float[batch, seq2], Y: float[batch, seq1]
//
// The Python builder records Z's axis-1 dim as the symbolic expression
// ``"seq1+seq2"``; in C++ the axis dimension is always a new symbolic
// placeholder (``"Concat_axis1"``).  The test verifies that the axis dim
// is symbolic and that ``ApplyInferredShapesToGraph`` correctly writes
// ``dim_param`` into the graph's value_info for Z.
TEST(OnnxOptimShapeBuilder, ConcatProducesSymbolicAxisDimAndAppliedToGraph) {
  ModelProto model;
  model.set_ir_version(10);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);

  GraphProto *graph = model.add_graph();
  graph->set_name("concat_symbolic");

  AddFloatInput(*graph, "Y", {-1, -1}, {"batch", "seq1"});
  AddFloatInput(*graph, "X", {-1, -1}, {"batch", "seq2"});
  AddFloatOutput(*graph, "Z", {-1, -1}, {});

  NodeProto concat_node = MakeNode("Concat", {"X", "Y"}, {"Z"});
  AddAttribute<int64_t>(concat_node, "axis", 1);
  *graph->add_node() = std::move(concat_node);

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  // Z has rank 2
  ASSERT_TRUE(ctx.Has("Z"));
  ASSERT_EQ(ctx.Get("Z").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("Z").Dtype(), onnx_optim::TensorType::kFloat);

  // dim 0: non-concat axis; X has "batch" and Y has "batch" → "batch"
  // (MergeDim: first symbolic wins when both are symbolic and equal)
  // In this case both inputs have "batch" for dim[0].
  CheckSymbolicDim(ctx, "Z", 0, "batch");

  // dim 1: concat axis → symbolic placeholder
  CheckIsSymbolic(ctx, "Z", 1);

  // ── ApplyInferredShapesToGraph writes shapes back to the proto ────────────
  onnx_optim::shapes::ApplyInferredShapesToGraph(ctx, *model.mutable_graph());

  // The output Z now has a shape in the proto.
  ASSERT_EQ(model.graph().output_size(), 1);
  const ValueInfoProto &out = model.graph().output(0);
  ASSERT_TRUE(out.type().tensor_type().has_shape());
  const TensorShapeProto &shape = out.type().tensor_type().shape();
  ASSERT_EQ(shape.dim_size(), 2);
  // dim 0: "batch"
  EXPECT_EQ(shape.dim(0).dim_param().as_string(), "batch");
  // dim 1: symbolic (some non-empty string)
  EXPECT_FALSE(shape.dim(1).dim_param().empty());
}

// ── test_concat_split ───────────────────────────────────────────────────────
//
// Translation of ``TestShapeBuilder.test_concat_split``.
//
// Graph:
//   xy = Concat(X, Y, axis=1)
//   S1, S2 = Split(xy, axis=1, num_outputs=2)
//   zs = Concat(S2, S1, axis=1)
//   Z  = Tanh(zs)
//
//   X: float[a, b]   Y: float[a, c]   Z: float[a, e]
//
// In C++ the concat-axis dim of ``xy`` is a symbolic placeholder; Split
// then assigns fresh symbolic names to S1/S2 (since the axis dim is
// symbolic).  The test checks ranks, non-axis dims, and that
// ``ApplyInferredShapesToGraph`` writes ``value_info`` for every
// intermediate tensor.
TEST(OnnxOptimShapeBuilder, ConcatSplitApplyInferredShapesToGraph) {
  ModelProto model;
  model.set_ir_version(9);
  OperatorSetIdProto *osi = model.add_opset_import();
  osi->set_domain("");
  osi->set_version(18);

  GraphProto *graph = model.add_graph();
  graph->set_name("concat_split");

  AddFloatInput(*graph, "X", {-1, -1}, {"a", "b"});
  AddFloatInput(*graph, "Y", {-1, -1}, {"a", "c"});
  AddFloatOutput(*graph, "Z", {-1, -1}, {"a", "e"});

  {
    NodeProto xy_node = MakeNode("Concat", {"X", "Y"}, {"xy"});
    AddAttribute<int64_t>(xy_node, "axis", 1);
    *graph->add_node() = std::move(xy_node);
  }
  {
    NodeProto split_node = MakeNode("Split", {"xy"}, {"S1", "S2"});
    AddAttribute<int64_t>(split_node, "axis", 1);
    AddAttribute<int64_t>(split_node, "num_outputs", 2);
    *graph->add_node() = std::move(split_node);
  }
  {
    NodeProto zs_node = MakeNode("Concat", {"S2", "S1"}, {"zs"});
    AddAttribute<int64_t>(zs_node, "axis", 1);
    *graph->add_node() = std::move(zs_node);
  }
  *graph->add_node() = MakeNode("Tanh", {"zs"}, {"Z"});

  onnx_optim::shapes::ShapesContext ctx;
  onnx_optim::shapes::ComputeShapeModel(ctx, model);

  // xy = Concat(X, Y, axis=1): rank 2
  ASSERT_TRUE(ctx.Has("xy"));
  ASSERT_EQ(ctx.Get("xy").Shape().Rank(), 2u);
  EXPECT_EQ(ctx.Get("xy").Dtype(), onnx_optim::TensorType::kFloat);
  // non-concat dim (axis=0) comes from both X and Y which both have "a"
  CheckSymbolicDim(ctx, "xy", 0, "a");
  // concat axis: symbolic (b + c cannot be summed symbolically in C++)
  CheckIsSymbolic(ctx, "xy", 1);

  // S1, S2: each has rank 2; the non-concat dim is "a"; axis dim is symbolic
  for (const std::string &name : {"S1", "S2"}) {
    ASSERT_TRUE(ctx.Has(name)) << "missing: " << name;
    ASSERT_EQ(ctx.Get(name).Shape().Rank(), 2u) << name;
    EXPECT_EQ(ctx.Get(name).Dtype(), onnx_optim::TensorType::kFloat) << name;
    CheckSymbolicDim(ctx, name, 0, "a");
    CheckIsSymbolic(ctx, name, 1);
  }

  // zs = Concat(S2, S1, axis=1): rank 2, dim[0]="a", dim[1] is symbolic
  ASSERT_TRUE(ctx.Has("zs"));
  ASSERT_EQ(ctx.Get("zs").Shape().Rank(), 2u);
  CheckSymbolicDim(ctx, "zs", 0, "a");
  CheckIsSymbolic(ctx, "zs", 1);

  // Z = Tanh(zs): same shape as zs
  ASSERT_TRUE(ctx.Has("Z"));
  ASSERT_EQ(ctx.Get("Z").Shape().Rank(), 2u);
  CheckSymbolicDim(ctx, "Z", 0, "a");
  CheckIsSymbolic(ctx, "Z", 1);

  // ── ApplyInferredShapesToGraph ────────────────────────────────────────────
  onnx_optim::shapes::ApplyInferredShapesToGraph(ctx, *model.mutable_graph());

  // value_info should contain entries for the intermediate tensors:
  // xy, S1, S2, zs (not X, Y, init/output)
  std::unordered_map<std::string, const ValueInfoProto *> vi_map;
  for (int i = 0; i < model.graph().value_info_size(); ++i) {
    const ValueInfoProto &vi = model.graph().value_info(i);
    vi_map[vi.name().as_string()] = &vi;
  }
  for (const std::string &name : {"xy", "S1", "S2", "zs"}) {
    EXPECT_TRUE(vi_map.count(name) > 0) << "value_info missing for " << name;
  }

  // Each intermediate value_info has rank 2, float, with the first dim_param = "a"
  for (const std::string &name : {"xy", "S1", "S2", "zs"}) {
    auto it = vi_map.find(name);
    if (it == vi_map.end()) {
      continue;
    }
    const ValueInfoProto &vi = *it->second;
    EXPECT_EQ(vi.type().tensor_type().elem_type(), static_cast<int>(TensorProto::DataType::FLOAT))
        << name;
    ASSERT_EQ(vi.type().tensor_type().shape().dim_size(), 2) << name;
    EXPECT_EQ(vi.type().tensor_type().shape().dim(0).dim_param().as_string(), "a") << name;
    EXPECT_FALSE(vi.type().tensor_type().shape().dim(1).dim_param().empty()) << name;
  }
}

// ── attribute-helper tests ──────────────────────────────────────────────────
//
// Translations of the ``test_get_attribute_with_default_*`` family in
// ``TestShapeBuilder``.  These exercise the C++ helpers ``FindAttribute``,
// ``GetAttributeOr<T>`` and ``GetAttributeInts`` from
// ``onnx_proto/onnx_helper.h``.

TEST(OnnxOptimShapeBuilder, GetAttributeOrReturnsIntValue) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "axis", 2);
  EXPECT_EQ(GetAttributeOr<int64_t>(node, "axis", 0), 2);
}

TEST(OnnxOptimShapeBuilder, GetAttributeOrReturnsFloatValue) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  AddAttribute<float>(node, "alpha", 0.5f);
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(node, "alpha", 1.0f), 0.5f);
}

TEST(OnnxOptimShapeBuilder, GetAttributeOrReturnsStringValue) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  AddAttribute<std::string>(node, "mode", std::string("constant"));
  EXPECT_EQ(GetAttributeOr<std::string>(node, "mode", std::string("")), "constant");
}

TEST(OnnxOptimShapeBuilder, GetAttributeOrReturnsDefaultWhenMissing) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "axis", 1);
  // "missing" is not present → default 42
  EXPECT_EQ(GetAttributeOr<int64_t>(node, "missing", static_cast<int64_t>(42)), 42);
  // float default
  EXPECT_FLOAT_EQ(GetAttributeOr<float>(node, "missing_f", 3.14f), 3.14f);
  // string default
  EXPECT_EQ(GetAttributeOr<std::string>(node, "missing_s", std::string("default")), "default");
}

TEST(OnnxOptimShapeBuilder, GetAttributeIntsAppendsValues) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  AddAttribute<std::vector<int64_t>>(node, "perm", {0, 2, 1});
  std::vector<int64_t> out;
  EXPECT_TRUE(GetAttributeInts(node, "perm", out));
  ASSERT_EQ(out.size(), 3u);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 2);
  EXPECT_EQ(out[2], 1);
}

TEST(OnnxOptimShapeBuilder, GetAttributeIntsReturnsFalseWhenMissing) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  std::vector<int64_t> out;
  EXPECT_FALSE(GetAttributeInts(node, "missing", out));
  EXPECT_TRUE(out.empty());
}

TEST(OnnxOptimShapeBuilder, FindAttributeReturnsPointerWhenPresent) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  AddAttribute<int64_t>(node, "axis", 5);
  const AttributeProto *attr = FindAttribute(node, "axis");
  ASSERT_NE(attr, nullptr);
  EXPECT_EQ(attr->ref_i(), 5);
}

TEST(OnnxOptimShapeBuilder, FindAttributeReturnsNullWhenMissing) {
  NodeProto node = MakeNode("SomeOp", {"X"}, {"Y"});
  EXPECT_EQ(FindAttribute(node, "missing"), nullptr);
}

TEST(OnnxOptimShapeBuilder, GetAttributeOrIntsReturnsListValue) {
  // Verifies that AddAttribute with a vector<int64_t> stores INTS and that
  // GetAttributeInts retrieves the values correctly.
  NodeProto node = MakeNode("Transpose", {"X"}, {"Y"});
  AddAttribute<std::vector<int64_t>>(node, "perm", {0, 2, 1, 3});
  std::vector<int64_t> out;
  ASSERT_TRUE(GetAttributeInts(node, "perm", out));
  ASSERT_EQ(out.size(), 4u);
  EXPECT_EQ(out[0], 0);
  EXPECT_EQ(out[1], 2);
  EXPECT_EQ(out[2], 1);
  EXPECT_EQ(out[3], 3);
}

} // namespace Test

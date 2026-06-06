// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_optim/shapes/shape_check.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>

using namespace ONNX_LIGHT_NAMESPACE;

namespace Test {

TEST(OnnxOptimShapesCheck, AcceptsMatchingOpAndOutput) {
  NodeProto node;
  node.set_op_type("Abs");
  node.add_output("Y");
  EXPECT_NO_THROW(onnx_optim::shapes::CheckNodeOpAndOutput(node, "Abs", "ComputeShapeAbs"));
}

TEST(OnnxOptimShapesCheck, RejectsWrongOpType) {
  NodeProto node;
  node.set_op_type("Neg");
  node.add_output("Y");
  try {
    onnx_optim::shapes::CheckNodeOpAndOutput(node, "Abs", "ComputeShapeAbs");
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("ComputeShapeAbs"), std::string::npos);
    EXPECT_NE(what.find("Abs"), std::string::npos);
    EXPECT_NE(what.find("Neg"), std::string::npos);
  }
}

TEST(OnnxOptimShapesCheck, RejectsNodeWithoutOutput) {
  NodeProto node;
  node.set_op_type("Abs");
  try {
    onnx_optim::shapes::CheckNodeOpAndOutput(node, "Abs", "ComputeShapeAbs");
    FAIL() << "Expected std::invalid_argument";
  } catch (const std::invalid_argument &e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("ComputeShapeAbs"), std::string::npos);
    EXPECT_NE(what.find("no output"), std::string::npos);
  }
}

} // namespace Test

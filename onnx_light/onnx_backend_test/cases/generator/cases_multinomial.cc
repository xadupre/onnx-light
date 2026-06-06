// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/generator/include_generator_cases.h"
#include "onnx_kernels/kernels/generator/include_generator_kernels.h"
#include "onnx_kernels/test_case.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {

namespace {

// Adds a FLOAT-typed attribute carrying ``value`` to ``node``.
void AddFloatAttr(NodeProto &node, const char *name, float value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::FLOAT);
  attr->set_f(value);
}

// Adds an INT-typed attribute carrying ``value`` to ``node``.
void AddIntAttr(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

} // namespace

// ---------------------------------------------------------------------------
// Multinomial — draws ``sample_size`` samples per batch row from a
// multinomial distribution whose per-row unnormalized log-probabilities
// are given by a 2-D input tensor of shape ``[batch_size, class_size]``.
// The schema marks the operator non-deterministic; the reference kernel
// uses a deterministic ``std::mt19937`` engine so these cases produce
// stable expected outputs.
// ---------------------------------------------------------------------------
void RegisterMultinomialCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};

  // Default attributes (sample_size=1, no seed, dtype=INT32). Two
  // batch rows of three classes; the second row strongly favors class 2.
  {
    const Tensor x =
        Tensor::FromFloat("x", {2, 3}, std::vector<float>{0.0f, 0.0f, 0.0f, -5.0f, -5.0f, 5.0f});

    NodeProto node;
    node.set_op_type("Multinomial");
    node.add_input("x");
    node.add_output("y");

    Tensor y = kernel::Multinomial(ctx)(x);
    Expect(node, {x}, {y}, "test_cc_multinomial", {opset}, "backend-test", registry);
  }

  // sample_size=5, explicit seed=42.
  {
    const Tensor x = Tensor::FromFloat(
        "x", {2, 4}, std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 0.5f, 0.25f, 0.125f, 0.0625f});

    NodeProto node;
    node.set_op_type("Multinomial");
    node.add_input("x");
    node.add_output("y");
    AddIntAttr(node, "sample_size", 5);
    AddFloatAttr(node, "seed", 42.0f);

    Tensor y = kernel::Multinomial(ctx)(x, /*sample_size=*/5, /*seed=*/42, /*dtype=*/0);
    Expect(node, {x}, {y}, "test_cc_multinomial_seeded", {opset}, "backend-test", registry);
  }

  // dtype=INT64.
  {
    const Tensor x = Tensor::FromFloat("x", {1, 3}, std::vector<float>{1.0f, 1.0f, 1.0f});

    NodeProto node;
    node.set_op_type("Multinomial");
    node.add_input("x");
    node.add_output("y");
    AddIntAttr(node, "sample_size", 4);
    AddIntAttr(node, "dtype", static_cast<int64_t>(DataType::INT64));

    Tensor y = kernel::Multinomial(ctx)(x, /*sample_size=*/4, kernel::Multinomial::kNoSeed,
                                        /*dtype=*/static_cast<int32_t>(DataType::INT64));
    Expect(node, {x}, {y}, "test_cc_multinomial_int64", {opset}, "backend-test", registry);
  }
}

} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

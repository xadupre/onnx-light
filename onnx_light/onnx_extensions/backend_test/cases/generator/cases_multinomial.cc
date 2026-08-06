// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

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
void RegisterMultinomialCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Multinomial");
    node.add_input("x");
    node.add_output("y");

    const onnx_kernels::kernel::Multinomial multinomial_kernel{ctx};
    Expect(registry, std::move(node), "test_cc_multinomial_benchmark", {opset}, {1024 * 4096},
           {1024}, [multinomial_kernel]() -> IoData {
             Tensor x =
                 Tensor::FromFloat("x", {1024, 4096}, Randn<float>({1024, 4096}, 987654321ULL));
             Tensor y = multinomial_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Default attributes (sample_size=1, no seed, dtype=INT32). Two
  // batch rows of three classes; the second row strongly favors class 2.
  {
    const Tensor x =
        Tensor::FromFloat("x", {2, 3}, std::vector<float>{0.0f, 0.0f, 0.0f, -5.0f, -5.0f, 5.0f});

    NodeProto node;
    node.set_op_type("Multinomial");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_multinomial", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::Multinomial(ctx)(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_multinomial_seeded", {opset}, [=]() -> IoData {
      Tensor y =
          onnx_kernels::kernel::Multinomial(ctx)(x, /*sample_size=*/5, /*seed=*/42, /*dtype=*/0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
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
    Expect(registry, std::move(node), "test_cc_multinomial_int64", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::Multinomial(ctx)(
          x, /*sample_size=*/4, onnx_kernels::kernel::Multinomial::kNoSeed,
          /*dtype=*/static_cast<int32_t>(DataType::INT64));
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

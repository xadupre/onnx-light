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

// Adds an INTS-typed attribute carrying ``values`` to ``node``.
void AddIntsAttr(NodeProto &node, const char *name, const std::vector<int64_t> &values) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INTS);
  for (int64_t v : values) {
    attr->ints().push_back(v);
  }
}

} // namespace

// ---------------------------------------------------------------------------
// RandomNormal — produces an output tensor of the given ``shape`` filled
// with samples drawn from a normal distribution with mean ``mean`` and
// standard deviation ``scale``. The operator is officially
// non-deterministic, but the reference kernel uses a deterministic
// SplitMix64 + Irwin-Hall RNG so these cases produce stable expected
// outputs.
// ---------------------------------------------------------------------------
void RegisterRandomNormalCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const std::vector<int64_t> shape = {2, 3};

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> large_shape = {kBenchmarkElementwiseSize};
    NodeProto node;
    node.set_op_type("RandomNormal");
    node.add_output("y");
    AddIntsAttr(node, "shape", large_shape);

    const onnx_kernels::kernel::RandomNormal random_normal_kernel{ctx};
    Expect(registry, std::move(node), "test_cc_randomnormal_benchmark", {opset},
           /*in_counts=*/{}, {kBenchmarkElementwiseSize},
           [random_normal_kernel, large_shape]() -> IoData {
             Tensor y = random_normal_kernel(large_shape);
             return IoData{/*inputs=*/{}, {std::move(y)}};
           });
    return;
  }

  // Default attributes (mean=0, scale=1, no seed, dtype=FLOAT).
  {
    NodeProto node;
    node.set_op_type("RandomNormal");
    node.add_output("y");
    AddIntsAttr(node, "shape", shape);
    Expect(registry, std::move(node), "test_cc_randomnormal", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomNormal(ctx)(shape);
      return IoData{{}, {std::move(y)}};
    });
  }

  // Explicit seed=42, mean=1, scale=2.
  {
    NodeProto node;
    node.set_op_type("RandomNormal");
    node.add_output("y");
    AddFloatAttr(node, "mean", 1.0f);
    AddFloatAttr(node, "scale", 2.0f);
    AddFloatAttr(node, "seed", 42.0f);
    AddIntsAttr(node, "shape", shape);
    Expect(registry, std::move(node), "test_cc_randomnormal_seeded", {opset}, [=]() -> IoData {
      Tensor y =
          onnx_kernels::kernel::RandomNormal(ctx)(shape, /*mean=*/1.0, /*scale=*/2.0, /*seed=*/42,
                                                  /*dtype=*/0);
      return IoData{{}, {std::move(y)}};
    });
  }

  // dtype=DOUBLE.
  {
    NodeProto node;
    node.set_op_type("RandomNormal");
    node.add_output("y");
    AddIntAttr(node, "dtype", static_cast<int64_t>(DataType::DOUBLE));
    AddIntsAttr(node, "shape", shape);
    Expect(registry, std::move(node), "test_cc_randomnormal_double", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomNormal(ctx)(
          shape, /*mean=*/0.0, /*scale=*/1.0, onnx_kernels::kernel::RandomNormal::kNoSeed,
          /*dtype=*/static_cast<int32_t>(DataType::DOUBLE));
      return IoData{{}, {std::move(y)}};
    });
  }
}

// ---------------------------------------------------------------------------
// RandomUniform — produces an output tensor of the given ``shape`` filled
// with samples drawn uniformly from ``[low, high)``.
// ---------------------------------------------------------------------------
void RegisterRandomUniformCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};
  const std::vector<int64_t> shape = {2, 3};

  if (mode == TestMode::BENCHMARK) {
    const std::vector<int64_t> large_shape = {kBenchmarkElementwiseSize};
    NodeProto node;
    node.set_op_type("RandomUniform");
    node.add_output("y");
    AddIntsAttr(node, "shape", large_shape);

    const onnx_kernels::kernel::RandomUniform random_uniform_kernel{ctx};
    Expect(registry, std::move(node), "test_cc_randomuniform_benchmark", {opset},
           /*in_counts=*/{}, {kBenchmarkElementwiseSize},
           [random_uniform_kernel, large_shape]() -> IoData {
             Tensor y = random_uniform_kernel(large_shape);
             return IoData{/*inputs=*/{}, {std::move(y)}};
           });
    return;
  }

  // Default attributes (low=0, high=1, no seed, dtype=FLOAT).
  {
    NodeProto node;
    node.set_op_type("RandomUniform");
    node.add_output("y");
    AddIntsAttr(node, "shape", shape);
    Expect(registry, std::move(node), "test_cc_randomuniform", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomUniform(ctx)(shape);
      return IoData{{}, {std::move(y)}};
    });
  }

  // Explicit seed=42, low=-1, high=3.
  {
    NodeProto node;
    node.set_op_type("RandomUniform");
    node.add_output("y");
    AddFloatAttr(node, "low", -1.0f);
    AddFloatAttr(node, "high", 3.0f);
    AddFloatAttr(node, "seed", 42.0f);
    AddIntsAttr(node, "shape", shape);
    Expect(registry, std::move(node), "test_cc_randomuniform_seeded", {opset}, [=]() -> IoData {
      Tensor y =
          onnx_kernels::kernel::RandomUniform(ctx)(shape, /*low=*/-1.0, /*high=*/3.0, /*seed=*/42,
                                                   /*dtype=*/0);
      return IoData{{}, {std::move(y)}};
    });
  }

  // dtype=DOUBLE.
  {
    NodeProto node;
    node.set_op_type("RandomUniform");
    node.add_output("y");
    AddIntAttr(node, "dtype", static_cast<int64_t>(DataType::DOUBLE));
    AddIntsAttr(node, "shape", shape);
    Expect(registry, std::move(node), "test_cc_randomuniform_double", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomUniform(ctx)(
          shape, /*low=*/0.0, /*high=*/1.0, onnx_kernels::kernel::RandomUniform::kNoSeed,
          /*dtype=*/static_cast<int32_t>(DataType::DOUBLE));
      return IoData{{}, {std::move(y)}};
    });
  }
}

// ---------------------------------------------------------------------------
// RandomNormalLike — copies the shape from a template input tensor.
// ---------------------------------------------------------------------------
void RegisterRandomNormalLikeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("RandomNormalLike");
    node.add_input("x");
    node.add_output("y");

    const onnx_kernels::kernel::RandomNormalLike random_normal_like_kernel{ctx};
    Expect(registry, std::move(node), "test_cc_randomnormallike_benchmark", {opset},
           {kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize},
           [random_normal_like_kernel]() -> IoData {
             Tensor x = Tensor::FromFloat("x", {kBenchmarkElementwiseSize},
                                          Randn<float>({kBenchmarkElementwiseSize}, 987654321ULL));
             Tensor y = random_normal_like_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Default attributes; input is FLOAT so output dtype matches.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, std::vector<float>(6, 0.0f));

    NodeProto node;
    node.set_op_type("RandomNormalLike");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_randomnormallike", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomNormalLike(ctx)(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Override dtype=DOUBLE from a FLOAT input.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, std::vector<float>(6, 0.0f));

    NodeProto node;
    node.set_op_type("RandomNormalLike");
    node.add_input("x");
    node.add_output("y");
    AddIntAttr(node, "dtype", static_cast<int64_t>(DataType::DOUBLE));
    Expect(registry, std::move(node), "test_cc_randomnormallike_double", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomNormalLike(ctx)(
          x, /*mean=*/0.0, /*scale=*/1.0, onnx_kernels::kernel::RandomNormalLike::kNoSeed,
          /*dtype=*/static_cast<int32_t>(DataType::DOUBLE));
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Explicit seed.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, std::vector<float>(6, 0.0f));

    NodeProto node;
    node.set_op_type("RandomNormalLike");
    node.add_input("x");
    node.add_output("y");
    AddFloatAttr(node, "mean", 1.0f);
    AddFloatAttr(node, "scale", 0.5f);
    AddFloatAttr(node, "seed", 7.0f);
    Expect(registry, std::move(node), "test_cc_randomnormallike_seeded", {opset}, [=]() -> IoData {
      Tensor y =
          onnx_kernels::kernel::RandomNormalLike(ctx)(x, /*mean=*/1.0, /*scale=*/0.5, /*seed=*/7,
                                                      /*dtype=*/0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

// ---------------------------------------------------------------------------
// RandomUniformLike — copies the shape from a template input tensor.
// ---------------------------------------------------------------------------
void RegisterRandomUniformLikeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("RandomUniformLike");
    node.add_input("x");
    node.add_output("y");

    const onnx_kernels::kernel::RandomUniformLike random_uniform_like_kernel{ctx};
    Expect(registry, std::move(node), "test_cc_randomuniformlike_benchmark", {opset},
           {kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize},
           [random_uniform_like_kernel]() -> IoData {
             Tensor x = Tensor::FromFloat("x", {kBenchmarkElementwiseSize},
                                          Randn<float>({kBenchmarkElementwiseSize}, 987654322ULL));
             Tensor y = random_uniform_like_kernel(x);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Default attributes.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, std::vector<float>(6, 0.0f));

    NodeProto node;
    node.set_op_type("RandomUniformLike");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_cc_randomuniformlike", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomUniformLike(ctx)(x);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Override dtype=DOUBLE from a FLOAT input.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, std::vector<float>(6, 0.0f));

    NodeProto node;
    node.set_op_type("RandomUniformLike");
    node.add_input("x");
    node.add_output("y");
    AddIntAttr(node, "dtype", static_cast<int64_t>(DataType::DOUBLE));
    Expect(registry, std::move(node), "test_cc_randomuniformlike_double", {opset}, [=]() -> IoData {
      Tensor y = onnx_kernels::kernel::RandomUniformLike(ctx)(
          x, /*low=*/0.0, /*high=*/1.0, onnx_kernels::kernel::RandomUniformLike::kNoSeed,
          /*dtype=*/static_cast<int32_t>(DataType::DOUBLE));
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Explicit seed and range.
  {
    const Tensor x = Tensor::FromFloat("x", {2, 3}, std::vector<float>(6, 0.0f));

    NodeProto node;
    node.set_op_type("RandomUniformLike");
    node.add_input("x");
    node.add_output("y");
    AddFloatAttr(node, "low", -2.0f);
    AddFloatAttr(node, "high", 5.0f);
    AddFloatAttr(node, "seed", 7.0f);
    Expect(registry, std::move(node), "test_cc_randomuniformlike_seeded", {opset}, [=]() -> IoData {
      Tensor y =
          onnx_kernels::kernel::RandomUniformLike(ctx)(x, /*low=*/-2.0, /*high=*/5.0, /*seed=*/7,
                                                       /*dtype=*/0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

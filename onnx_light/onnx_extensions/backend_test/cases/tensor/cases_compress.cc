// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <optional>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

NodeProto MakeCompressNode(std::optional<int64_t> axis) {
  NodeProto node;
  node.set_op_type("Compress");
  node.add_input("input");
  node.add_input("condition");
  node.add_output("output");
  if (axis.has_value()) {
    AddAttribute<int64_t>(node, "axis", *axis);
  }
  return node;
}

} // namespace

void RegisterCompressCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(11);
  const auto compress_kernel = MakeReferenceKernel<onnx_kernels::kernel::Compress>(opset);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeCompressNode(std::nullopt);
    Expect(registry, std::move(node), "test_cc_compress_no_axis_benchmark", {opset},
           {kBenchmarkElementwiseSize, kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize / 2},
           [compress_kernel]() -> IoData {
             Tensor input = RandnTensor(DataType::FLOAT, {4096, 1024}, 2001);
             std::vector<uint8_t> condition_values(kBenchmarkElementwiseSize);
             for (int64_t i = 0; i < kBenchmarkElementwiseSize; ++i) {
               condition_values[static_cast<std::size_t>(i)] = static_cast<uint8_t>((i % 2) == 0);
             }
             Tensor condition =
                 Tensor::FromBool("condition", {kBenchmarkElementwiseSize}, condition_values);
             Tensor output = compress_kernel.Invoke(
                 [&](const auto &kernel) { return kernel(input, condition, std::nullopt); });
             return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
           });
    return;
  }

  // test_cc_compress_no_axis — flatten then select elements.
  {
    Expect(registry, MakeCompressNode(std::nullopt), "test_cc_compress_no_axis", {opset},
           [=]() -> IoData {
             Tensor input =
                 Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
             Tensor condition = Tensor::FromBool("condition", {6}, {1, 0, 1, 1, 0, 0});
             Tensor output = compress_kernel.Invoke(
                 [&](const auto &kernel) { return kernel(input, condition, std::nullopt); });
             return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
           });
  }

  // test_cc_compress_axis0 — select rows (axis=0).
  {
    Expect(registry, MakeCompressNode(0), "test_cc_compress_axis0", {opset}, [=]() -> IoData {
      Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor condition = Tensor::FromBool("condition", {3}, {1, 0, 1});
      Tensor output =
          compress_kernel.Invoke([&](const auto &kernel) { return kernel(input, condition, 0); });
      return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
    });
  }

  // test_cc_compress_axis1 — select columns (axis=1).
  {
    Expect(registry, MakeCompressNode(1), "test_cc_compress_axis1", {opset}, [=]() -> IoData {
      Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor condition = Tensor::FromBool("condition", {2}, {0, 1});
      Tensor output =
          compress_kernel.Invoke([&](const auto &kernel) { return kernel(input, condition, 1); });
      return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
    });
  }

  // test_cc_compress_negative_axis — negative axis (-1 == last axis).
  {
    Expect(
        registry, MakeCompressNode(-1), "test_cc_compress_negative_axis", {opset}, [=]() -> IoData {
          Tensor input = Tensor::FromFloat("input", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
          Tensor condition = Tensor::FromBool("condition", {3}, {1, 0, 1});
          Tensor output = compress_kernel.Invoke(
              [&](const auto &kernel) { return kernel(input, condition, -1); });
          return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
        });
  }

  // test_cc_compress_short_condition — condition shorter than axis dim.
  {
    Expect(registry, MakeCompressNode(0), "test_cc_compress_short_condition", {opset},
           [=]() -> IoData {
             Tensor input = Tensor::FromFloat("input", {4, 2},
                                              {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
             Tensor condition = Tensor::FromBool("condition", {2}, {1, 0});
             Tensor output = compress_kernel.Invoke(
                 [&](const auto &kernel) { return kernel(input, condition, 0); });
             return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
           });
  }

  // test_cc_compress_int64 — non-float input dtype.
  {
    Expect(registry, MakeCompressNode(0), "test_cc_compress_int64", {opset}, [=]() -> IoData {
      Tensor input = Tensor::FromInt64("input", {3}, {10, 20, 30});
      Tensor condition = Tensor::FromBool("condition", {3}, {0, 1, 1});
      Tensor output =
          compress_kernel.Invoke([&](const auto &kernel) { return kernel(input, condition, 0); });
      return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
    });
  }

  // test_cc_compress_0 — mirrors ONNX ``test_compress_0`` (axis=0, condition
  // selects the last two rows of a 3x2 matrix).
  {
    Expect(registry, MakeCompressNode(0), "test_cc_compress_0", {opset}, [=]() -> IoData {
      Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor condition = Tensor::FromBool("condition", {3}, {0, 1, 1});
      Tensor output =
          compress_kernel.Invoke([&](const auto &kernel) { return kernel(input, condition, 0); });
      return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
    });
  }

  // test_cc_compress_1 — mirrors ONNX ``test_compress_1`` (axis=1, condition
  // selects the last column of a 3x2 matrix).
  {
    Expect(registry, MakeCompressNode(1), "test_cc_compress_1", {opset}, [=]() -> IoData {
      Tensor input = Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor condition = Tensor::FromBool("condition", {2}, {0, 1});
      Tensor output =
          compress_kernel.Invoke([&](const auto &kernel) { return kernel(input, condition, 1); });
      return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
    });
  }

  // test_cc_compress_default_axis — mirrors ONNX ``test_compress_default_axis``
  // (no axis attribute; the input is flattened first and ``condition`` selects
  // individual elements).
  {
    Expect(registry, MakeCompressNode(std::nullopt), "test_cc_compress_default_axis", {opset},
           [=]() -> IoData {
             Tensor input =
                 Tensor::FromFloat("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
             Tensor condition = Tensor::FromBool("condition", {5}, {0, 1, 0, 0, 1});
             Tensor output = compress_kernel.Invoke(
                 [&](const auto &kernel) { return kernel(input, condition, std::nullopt); });
             return IoData{{std::move(input), std::move(condition)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

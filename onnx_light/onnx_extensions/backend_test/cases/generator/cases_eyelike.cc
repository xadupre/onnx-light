// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterEyeLikeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);
  const KernelContext ctx{opset};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    const onnx_kernels::kernel::EyeLike eye_like_kernel{ctx};
    Expect(registry, std::move(node), "test_eyelike_without_dtype_benchmark", {opset},
           {2048 * 2048}, {2048 * 2048}, [eye_like_kernel]() -> IoData {
             Tensor x =
                 Tensor::FromFloat("x", {2048, 2048}, Randn<float>({2048, 2048}, 987654321ULL));
             Tensor y = eye_like_kernel(x, /*k=*/0, /*dtype=*/0);
             return IoData{{std::move(x)}, {std::move(y)}};
           });
    return;
  }

  // Upstream-style case without dtype attribute: output dtype follows input.
  {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_eyelike_without_dtype", {opset}, [=]() -> IoData {
      const Tensor x = Tensor::FromFloat(
          "x", {3, 4}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
      const Tensor y = onnx_kernels::kernel::EyeLike(ctx)(x, /*k=*/0, /*dtype=*/0);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream-style case with dtype override and non-zero diagonal offset.
  {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    auto *k_attr = node.add_attribute();
    k_attr->set_name("k");
    k_attr->set_type(AttributeProto::AttributeType::INT);
    k_attr->set_i(1);
    auto *dtype_attr = node.add_attribute();
    dtype_attr->set_name("dtype");
    dtype_attr->set_type(AttributeProto::AttributeType::INT);
    Expect(registry, std::move(node), "test_eyelike_with_dtype", {opset}, [=]() -> IoData {
      dtype_attr->set_i(static_cast<int64_t>(DataType::INT64));

      const Tensor x =
          Tensor::FromFloat("x", {2, 4}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
      const Tensor y =
          onnx_kernels::kernel::EyeLike(ctx)(x, /*k=*/1, static_cast<int32_t>(DataType::INT64));
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Upstream-style case populating an off-main diagonal with explicit FLOAT dtype.
  {
    NodeProto node;
    node.set_op_type("EyeLike");
    node.add_input("x");
    node.add_output("y");
    auto *k_attr = node.add_attribute();
    k_attr->set_name("k");
    k_attr->set_type(AttributeProto::AttributeType::INT);
    k_attr->set_i(1);
    auto *dtype_attr = node.add_attribute();
    dtype_attr->set_name("dtype");
    dtype_attr->set_type(AttributeProto::AttributeType::INT);
    Expect(registry, std::move(node), "test_eyelike_populate_off_main_diagonal", {opset},
           [=]() -> IoData {
             dtype_attr->set_i(static_cast<int64_t>(DataType::FLOAT));

             const Tensor x = Tensor::FromFloat(
                 "x", {4, 5}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
             const Tensor y = onnx_kernels::kernel::EyeLike(ctx)(
                 x, /*k=*/1, static_cast<int32_t>(DataType::FLOAT));
             return IoData{{std::move(x)}, {std::move(y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

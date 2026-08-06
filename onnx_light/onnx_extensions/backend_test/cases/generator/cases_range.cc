// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/generator/include_generator_cases.h"
#include "onnx_extensions/kernels/kernels/generator/include_generator_kernels.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// IEEE-754 binary16 / bfloat16 encoders are provided by
// ``onnx_core/runtime/cast_helper.h``; the scalar tensor builders
// ``MakeFloat16Scalar`` / ``MakeBfloat16Scalar`` are used
// directly by the case registrations below.

} // namespace

void RegisterRangeCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset_v11 = DefaultOpset(11);
  const OpsetId opset_v27 = DefaultOpset(27);
  const KernelContext ctx_v11{opset_v11};
  const KernelContext ctx_v27{opset_v27};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");

    const onnx_kernels::kernel::Range range_kernel{ctx_v11};
    Expect(registry, std::move(node), "test_range_float_type_positive_delta_benchmark", {opset_v11},
           {1, 1, 1}, {kBenchmarkElementwiseSize}, [range_kernel]() -> IoData {
             Tensor start = Tensor::FromFloat("start", {}, {0.0f});
             Tensor limit =
                 Tensor::FromFloat("limit", {}, {static_cast<float>(kBenchmarkElementwiseSize)});
             Tensor delta = Tensor::FromFloat("delta", {}, {1.0f});
             Tensor output = range_kernel(start, limit, delta);
             return IoData{{std::move(start), std::move(limit), std::move(delta)},
                           {std::move(output)}};
           });
    return;
  }

  // Upstream test: range_float_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0]
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");
    Expect(registry, std::move(node), "test_range_float_type_positive_delta", {opset_v11},
           [=]() -> IoData {
             const Tensor start = Tensor::FromFloat("start", {}, {1.0f});
             const Tensor limit = Tensor::FromFloat("limit", {}, {5.0f});
             const Tensor delta = Tensor::FromFloat("delta", {}, {2.0f});
             const Tensor output = onnx_kernels::kernel::Range(ctx_v11)(start, limit, delta);
             return IoData{{std::move(start), std::move(limit), std::move(delta)},
                           {std::move(output)}};
           });
  }

  // Upstream test: range_int32_type_negative_delta
  // start=10, limit=6, delta=-3  ->  [10, 7]
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");
    Expect(registry, std::move(node), "test_range_int32_type_negative_delta", {opset_v11},
           [=]() -> IoData {
             const Tensor start = Tensor::FromInt32("start", {}, {10});
             const Tensor limit = Tensor::FromInt32("limit", {}, {6});
             const Tensor delta = Tensor::FromInt32("delta", {}, {-3});
             const Tensor output = onnx_kernels::kernel::Range(ctx_v11)(start, limit, delta);
             return IoData{{std::move(start), std::move(limit), std::move(delta)},
                           {std::move(output)}};
           });
  }

  // Upstream test (opset 27): range_float16_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0] as float16
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");
    Expect(registry, std::move(node), "test_range_float16_type_positive_delta", {opset_v27},
           [=]() -> IoData {
             const Tensor start = MakeFloat16Scalar("start", 1.0f);
             const Tensor limit = MakeFloat16Scalar("limit", 5.0f);
             const Tensor delta = MakeFloat16Scalar("delta", 2.0f);
             const Tensor output = onnx_kernels::kernel::Range(ctx_v27)(start, limit, delta);
             return IoData{{std::move(start), std::move(limit), std::move(delta)},
                           {std::move(output)}};
           });
  }

  // Upstream test (opset 27): range_bfloat16_type_positive_delta
  // start=1, limit=5, delta=2  ->  [1.0, 3.0] as bfloat16
  {
    NodeProto node;
    node.set_op_type("Range");
    node.add_input("start");
    node.add_input("limit");
    node.add_input("delta");
    node.add_output("output");
    Expect(registry, std::move(node), "test_range_bfloat16_type_positive_delta", {opset_v27},
           [=]() -> IoData {
             const Tensor start = MakeBfloat16Scalar("start", 1.0f);
             const Tensor limit = MakeBfloat16Scalar("limit", 5.0f);
             const Tensor delta = MakeBfloat16Scalar("delta", 2.0f);
             const Tensor output = onnx_kernels::kernel::Range(ctx_v27)(start, limit, delta);
             return IoData{{std::move(start), std::move(limit), std::move(delta)},
                           {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

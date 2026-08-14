// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

// ---------------------------------------------------------------------------
// Min -- element-wise variadic min with NumPy-style broadcasting (since
// opset 8; opset 12 widens to all numeric types; opset 13 widens further to
// include bfloat16). Mirrors ``onnx.backend.test.case.node.min.Min``.
// ---------------------------------------------------------------------------
void RegisterMinCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Min min_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Min");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    const std::vector<int64_t> shape = {kBenchmarkElementwiseSize};
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_min_benchmark", {opset}, {count, count}, {count},
           [min_kernel, shape]() -> IoData {
             Tensor x0 = RandnTensor(DataType::FLOAT, shape, 421);
             Tensor x1 = RandnTensor(DataType::FLOAT, shape, 422);
             Tensor z = min_kernel({x0, x1});
             return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
           });
    return;
  }

  // ``Min.export()`` -- three-input example mirroring the upstream Python
  // reference test ``test_min_example``.
  {
    NodeProto node;
    node.set_op_type("Min");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_input("data_2");
    node.add_output("result");
    Expect(registry, std::move(node), "test_min_example", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
      Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 4.0f});
      Tensor x2 = Tensor::FromFloat("", {3}, {2.0f, 5.0f, 0.0f});
      Tensor z = min_kernel({x0, x1, x2});

      return IoData{{std::move(x0), std::move(x1), std::move(x2)}, {std::move(z)}};
    });
  }

  // ``Min.export()`` -- single-input variant ``test_min_one_input``.
  {
    NodeProto node;
    node.set_op_type("Min");
    node.add_input("data_0");
    node.add_output("result");
    Expect(registry, std::move(node), "test_min_one_input", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
      Tensor z = min_kernel({x0});

      return IoData{{std::move(x0)}, {std::move(z)}};
    });
  }

  // ``Min.export()`` -- two-input variant ``test_min_two_inputs``.
  {
    NodeProto node;
    node.set_op_type("Min");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    Expect(registry, std::move(node), "test_min_two_inputs", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
      Tensor x1 = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 4.0f});
      Tensor z = min_kernel({x0, x1});

      return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
    });
  }

  // ``Min.export_min_all_numeric_types()`` -- one two-input case per numeric
  // dtype supported by ``kernel::Min``.
  const std::vector<std::pair<std::string, std::function<IoData()>>> dtype_cases = {
      {"test_min_float32",
       [=]() -> IoData {
         auto x0 = Tensor::FromFloat("", {3}, {3.0f, 2.0f, 1.0f});
         auto x1 = Tensor::FromFloat("", {3}, {1.0f, 4.0f, 4.0f});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_float64",
       [=]() -> IoData {
         auto x0 = Tensor::FromDouble("", {3}, {3.0, 2.0, 1.0});
         auto x1 = Tensor::FromDouble("", {3}, {1.0, 4.0, 4.0});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_int8",
       [=]() -> IoData {
         auto x0 = Tensor::FromInt8("", {3}, {int8_t{3}, int8_t{2}, int8_t{1}});
         auto x1 = Tensor::FromInt8("", {3}, {int8_t{1}, int8_t{4}, int8_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_int16",
       [=]() -> IoData {
         auto x0 = Tensor::FromInt16("", {3}, {int16_t{3}, int16_t{2}, int16_t{1}});
         auto x1 = Tensor::FromInt16("", {3}, {int16_t{1}, int16_t{4}, int16_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_int32",
       [=]() -> IoData {
         auto x0 = Tensor::FromInt32("", {3}, {3, 2, 1});
         auto x1 = Tensor::FromInt32("", {3}, {1, 4, 4});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_int64",
       [=]() -> IoData {
         auto x0 = Tensor::FromInt64("", {3}, {int64_t{3}, int64_t{2}, int64_t{1}});
         auto x1 = Tensor::FromInt64("", {3}, {int64_t{1}, int64_t{4}, int64_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_uint8",
       [=]() -> IoData {
         auto x0 = Tensor::FromUint8("", {3}, {uint8_t{3}, uint8_t{2}, uint8_t{1}});
         auto x1 = Tensor::FromUint8("", {3}, {uint8_t{1}, uint8_t{4}, uint8_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_uint16",
       [=]() -> IoData {
         auto x0 = Tensor::FromUint16("", {3}, {uint16_t{3}, uint16_t{2}, uint16_t{1}});
         auto x1 = Tensor::FromUint16("", {3}, {uint16_t{1}, uint16_t{4}, uint16_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_uint32",
       [=]() -> IoData {
         auto x0 = Tensor::FromUint32("", {3}, {uint32_t{3}, uint32_t{2}, uint32_t{1}});
         auto x1 = Tensor::FromUint32("", {3}, {uint32_t{1}, uint32_t{4}, uint32_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
      {"test_min_uint64",
       [=]() -> IoData {
         auto x0 = Tensor::FromUint64("", {3}, {uint64_t{3}, uint64_t{2}, uint64_t{1}});
         auto x1 = Tensor::FromUint64("", {3}, {uint64_t{1}, uint64_t{4}, uint64_t{4}});
         Tensor z = min_kernel({x0, x1});
         return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
       }},
  };

  NodeProto two_input_node;
  two_input_node.set_op_type("Min");
  two_input_node.add_input("data_0");
  two_input_node.add_input("data_1");
  two_input_node.add_output("result");

  for (const auto &[name, make_io] : dtype_cases) {
    Expect(registry, two_input_node, name, {opset}, make_io);
  }

  // Broadcasting variant: rank-2 vs scalar.
  {
    NodeProto node;
    node.set_op_type("Min");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    Expect(registry, std::move(node), "test_cc_min_bcast", {opset}, [=]() -> IoData {
      Tensor x0 = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor x1 = Tensor::FromFloat("", {}, {2.5f});
      Tensor z = min_kernel({x0, x1});

      return IoData{{std::move(x0), std::move(x1)}, {std::move(z)}};
    });
  }

  // ``test_min_float16`` — Min on FLOAT16 inputs with hardcoded expected.
  {
    NodeProto node;
    node.set_op_type("Min");
    node.add_input("data_0");
    node.add_input("data_1");
    node.add_output("result");
    Expect(registry, std::move(node), "test_min_float16", {opset}, [=]() -> IoData {
      Tensor x0 = MakeFloat16Tensor("", {3}, {1.0f, 4.0f, 3.0f});
      Tensor x1 = MakeFloat16Tensor("", {3}, {3.0f, 2.0f, 5.0f});
      Tensor expected = MakeFloat16Tensor("", {3}, {1.0f, 2.0f, 3.0f});
      return IoData{{std::move(x0), std::move(x1)}, {std::move(expected)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/logical/include_logical_cases.h"
#include "onnx_extensions/kernels/kernels/logical/include_logical_kernels.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterWhereCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(16);
  const KernelContext ctx{opset};
  const onnx_kernels::kernel::Where where_kernel{ctx};

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});

    const std::vector<int64_t> shape = {1024, 4096};
    const int64_t count = 1024 * 4096;
    Expect(registry, std::move(node), "test_where_example_benchmark", {opset},
           {count, count, count}, {count}, [where_kernel, shape]() -> IoData {
             Tensor condition =
                 Tensor::FromBool("condition", shape, RandUint<uint8_t>(2, shape, /*seed=*/9401));
             Tensor x = Tensor::FromFloat("x", shape, Randn<float>(shape, /*seed=*/9402));
             Tensor y = Tensor::FromFloat("y", shape, Randn<float>(shape, /*seed=*/9403));
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
    return;
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_where_example", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromFloat("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
             Tensor y = Tensor::FromFloat("y", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    // condition:{2,1}=2 elements, x:{2,3}=6, y:{1,3}=3; broadcast output:{2,3}=6.
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_where_bcast", {opset}, {2, 6, 3}, {6},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 1}, {1, 0});
             Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
             Tensor y = Tensor::FromInt32("y", {1, 3}, {10, 20, 30});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  // Mirrors upstream onnx ``test_where_long_example``: int64 inputs.
  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_where_long_example", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 1});
             Tensor x = Tensor::FromInt64("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromInt64("y", {2, 2}, {9, 8, 7, 6});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  // Additional cases covering the remaining x/y dtypes supported by the Where
  // kernel. ``condition`` is always BOOL; ``x`` and ``y`` share the dtype.
  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_bool", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromBool("x", {2, 2}, {1, 1, 0, 0});
             Tensor y = Tensor::FromBool("y", {2, 2}, {0, 0, 1, 1});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_double", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromDouble("x", {2, 2}, {1.0, 2.0, 3.0, 4.0});
             Tensor y = Tensor::FromDouble("y", {2, 2}, {5.0, 6.0, 7.0, 8.0});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_int8", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromInt8("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromInt8("y", {2, 2}, {-5, -6, -7, -8});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_int16", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromInt16("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromInt16("y", {2, 2}, {-5, -6, -7, -8});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_uint8", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromUint8("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromUint8("y", {2, 2}, {5, 6, 7, 8});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_uint16", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromUint16("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromUint16("y", {2, 2}, {5, 6, 7, 8});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_uint32", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromUint32("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromUint32("y", {2, 2}, {5, 6, 7, 8});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_uint64", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromUint64("x", {2, 2}, {1, 2, 3, 4});
             Tensor y = Tensor::FromUint64("y", {2, 2}, {5, 6, 7, 8});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }

  {
    NodeProto node = MakeNode("Where", {"condition", "x", "y"}, {"output"});
    Expect(registry, std::move(node), "test_cc_where_string", {opset}, {4, 4, 4}, {4},
           [where_kernel]() -> IoData {
             Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
             Tensor x = Tensor::FromStrings("x", {2, 2}, {"a", "b", "c", "d"});
             Tensor y = Tensor::FromStrings("y", {2, 2}, {"e", "f", "g", "h"});
             Tensor output = where_kernel(condition, x, y);
             return IoData{{std::move(condition), std::move(x), std::move(y)}, {std::move(output)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

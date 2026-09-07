// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds a ``Mod`` NodeProto. ``fmod`` is the ``int`` attribute documented by
// the ONNX schema: when omitted (``fmod == 0``) the operator behaves like
// Python/numpy ``mod`` (sign follows divisor); when set to
// ``1`` it behaves like C ``fmod`` (sign follows dividend, both integer and
// floating-point inputs accepted).
NodeProto MakeModNode(int64_t fmod, bool explicit_fmod = false) {
  NodeProto node;
  node.set_op_type("Mod");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");
  if (fmod != 0 || explicit_fmod) {
    AttributeProto *attr = node.add_attribute();
    attr->set_name("fmod");
    attr->set_type(AttributeProto::INT);
    attr->set_i(fmod);
  }
  return node;
}

std::vector<int32_t> Arange30() {
  std::vector<int32_t> values(30);
  std::iota(values.begin(), values.end(), 0);
  return values;
}

Tensor MakeModFloatTensor(int32_t dtype, const std::vector<double> &values) {
  const std::vector<int64_t> shape{static_cast<int64_t>(values.size())};
  if (dtype == DataType::DOUBLE) {
    return Tensor::FromDouble("", shape, values);
  }
  std::vector<float> floats;
  floats.reserve(values.size());
  for (double value : values) {
    floats.push_back(static_cast<float>(value));
  }
  return dtype == DataType::FLOAT16 ? MakeFloat16Tensor("", shape, floats)
                                    : Tensor::FromFloat("", shape, floats);
}

// IEEE-754 binary16 encoder (round-to-nearest-even) and the FLOAT16 tensor
// builder ``MakeFloat16Tensor`` are provided by
// ``onnx_core/runtime/kernels/cast_helper.h``.

} // namespace

// ---------------------------------------------------------------------------
// Mod — z = x mod y, element-wise with broadcasting (since opset 13).
// ---------------------------------------------------------------------------
void RegisterModCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node = MakeModNode(/*fmod=*/1);
    const std::vector<int64_t> shape = {kBenchmarkElementwiseSize};
    const int64_t count = kBenchmarkElementwiseSize;
    Expect(registry, std::move(node), "test_cc_mod_benchmark", {opset}, {count, count}, {count},
           [shape]() -> IoData {
             const OpsetId opset = DefaultOpset(13);

             const KernelContext mod_kernel_ctx{opset};
             const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, shape, 427);
             Tensor y = RandnTensor(DataType::FLOAT, shape, 428);
             Tensor z = mod_kernel(x, y, /*fmod=*/1);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    return;
  }

  // Upstream ONNX backend test cases for the ``Mod`` operator (mirror the
  // ``onnx.backend.test.case.node.mod.Mod`` Python class).

  for (const auto &[dtype, dtype_name] :
       std::vector<std::pair<int32_t, std::string>>{{DataType::FLOAT16, "float16"},
                                                    {DataType::FLOAT, "float32"},
                                                    {DataType::DOUBLE, "float64"}}) {
    const OpsetId weekly_opset = DefaultOpset(28);
    Expect(registry, MakeModNode(0, true), "test_mod_" + dtype_name + "_mixed_sign_fmod_0",
           {weekly_opset}, [dtype]() -> IoData {
             const KernelContext ctx{DefaultOpset(28)};
             const onnx_kernels::kernel::Mod kernel{ctx};
             Tensor x = MakeModFloatTensor(dtype, {-4.3, 7.2, 5.0, 4.3, -7.2, 8.0});
             Tensor y = MakeModFloatTensor(dtype, {2.1, -3.4, 8.0, -2.1, 3.4, 5.0});
             Tensor z = kernel(x, y, 0);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
    Expect(registry, MakeModNode(0, true), "test_mod_float_edge_cases_fmod_0_" + dtype_name,
           {weekly_opset}, [dtype]() -> IoData {
             const KernelContext ctx{DefaultOpset(28)};
             const onnx_kernels::kernel::Mod kernel{ctx};
             const double inf = std::numeric_limits<double>::infinity();
             const double nan = std::numeric_limits<double>::quiet_NaN();
             Tensor x = MakeModFloatTensor(dtype, {0.0, -0.0, 0.0, -0.0, -3.0, 3.0, -1.0, 1.0, inf,
                                                   -inf, 1.0, 1.0, nan, 1.0});
             Tensor y = MakeModFloatTensor(dtype, {-2.0, 2.0, 2.0, -2.0, inf, inf, -inf, -inf, 2.0,
                                                   2.0, 0.0, -0.0, 2.0, nan});
             Tensor z = kernel(x, y, 0);
             return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
           });
  }

  // From Mod.export_mod_mixed_sign_float32() / _float64().
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Expect(registry, std::move(node), "test_mod_mixed_sign_float16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = MakeFloat16Tensor("", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
      Tensor y = MakeFloat16Tensor("", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
      Tensor z = mod_kernel(x, y, /*fmod=*/1);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Expect(registry, std::move(node), "test_mod_mixed_sign_bfloat16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = MakeBfloat16Tensor("", {6}, {-4.0f, 7.0f, 5.0f, 4.0f, -7.0f, 8.0f});
      Tensor y = MakeBfloat16Tensor("", {6}, {2.0f, -3.0f, 8.0f, -2.0f, 3.0f, 5.0f});
      Tensor z = mod_kernel(x, y, /*fmod=*/1);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Expect(registry, std::move(node), "test_mod_mixed_sign_float32", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromFloat("", {6}, {-4.3f, 7.2f, 5.0f, 4.3f, -7.2f, 8.0f});
      Tensor y = Tensor::FromFloat("", {6}, {2.1f, -3.4f, 8.0f, -2.1f, 3.4f, 5.0f});
      Tensor z = mod_kernel(x, y, /*fmod=*/1);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Expect(registry, std::move(node), "test_mod_mixed_sign_float64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromDouble("", {6}, {-4.3, 7.2, 5.0, 4.3, -7.2, 8.0});
      Tensor y = Tensor::FromDouble("", {6}, {2.1, -3.4, 8.0, -2.1, 3.4, 5.0});
      Tensor z = mod_kernel(x, y, /*fmod=*/1);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // From Mod.export_mod_mixed_sign_int{8,16,32,64}().
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_mixed_sign_int8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromInt8("", {6}, {-4, 7, 5, 4, -7, 8});
      Tensor y = Tensor::FromInt8("", {6}, {2, -3, 8, -2, 3, 5});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_mixed_sign_int16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromInt16("", {6}, {-4, 7, 5, 4, -7, 8});
      Tensor y = Tensor::FromInt16("", {6}, {2, -3, 8, -2, 3, 5});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_mixed_sign_int32", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromInt32("", {6}, {-4, 7, 5, 4, -7, 8});
      Tensor y = Tensor::FromInt32("", {6}, {2, -3, 8, -2, 3, 5});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_mixed_sign_int64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromInt64("", {6}, {-4, 7, 5, 4, -7, 8});
      Tensor y = Tensor::FromInt64("", {6}, {2, -3, 8, -2, 3, 5});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // From Mod.export_mod_uint{8,16,32,64}().
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_uint8", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromUint8("", {3}, {4, 7, 5});
      Tensor y = Tensor::FromUint8("", {3}, {2, 3, 8});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_uint16", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromUint16("", {3}, {4, 7, 5});
      Tensor y = Tensor::FromUint16("", {3}, {2, 3, 8});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_uint32", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromUint32("", {3}, {4u, 7u, 5u});
      Tensor y = Tensor::FromUint32("", {3}, {2u, 3u, 8u});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_uint64", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromUint64("", {3}, {4ull, 7ull, 5ull});
      Tensor y = Tensor::FromUint64("", {3}, {2ull, 3ull, 8ull});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // From Mod.export_mod_int64_fmod() — sign follows dividend.
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Expect(registry, std::move(node), "test_mod_int64_fmod", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromInt64("", {6}, {-4, 7, 5, 4, -7, 8});
      Tensor y = Tensor::FromInt64("", {6}, {2, -3, 8, -2, 3, 5});
      Tensor z = mod_kernel(x, y, /*fmod=*/1);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // From Mod.export_mod_broadcast() — int32, scalar-ish divisor.
  {
    NodeProto node = MakeModNode(/*fmod=*/0);
    Expect(registry, std::move(node), "test_mod_broadcast", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = Tensor::FromInt32("", {3, 2, 5}, Arange30());
      Tensor y = Tensor::FromInt32("", {1}, {7});
      Tensor z = mod_kernel(x, y);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }

  // BFLOAT16 with fmod=1
  {
    NodeProto node = MakeModNode(/*fmod=*/1);
    Expect(registry, std::move(node), "test_cc_mod_bfloat16_fmod", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(13);

      const KernelContext mod_kernel_ctx{opset};
      const onnx_kernels::kernel::Mod mod_kernel{mod_kernel_ctx};

      Tensor x = MakeBfloat16Tensor("", {3}, {4.5f, -4.5f, 7.0f});
      Tensor y = MakeBfloat16Tensor("", {3}, {3.0f, 3.0f, 2.5f});
      Tensor z = mod_kernel(x, y, /*fmod=*/1);
      return IoData{{std::move(x), std::move(y)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

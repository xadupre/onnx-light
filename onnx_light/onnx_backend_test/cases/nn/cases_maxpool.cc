// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/nn/include_nn_cases.h"
#include "onnx_backend_test/kernels/nn/include_nn_kernels.h"
#include "onnx_backend_test/random.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

namespace {

// Builds a vector ``{1.0f, 2.0f, ..., n}`` used to generate the deterministic
// inputs for the precomputed MaxPool reference cases.
std::vector<float> Range1ToN(int64_t n) {
  std::vector<float> v;
  v.reserve(static_cast<size_t>(n));
  for (int64_t i = 1; i <= n; ++i) {
    v.push_back(static_cast<float>(i));
  }
  return v;
}

} // namespace

// ---------------------------------------------------------------------------
// MaxPool — y = max-pool(x, kernel_shape[, strides, pads, ceil_mode,
// dilations, auto_pad, storage_order]) and optional ``Indices`` second output
// (since opset 8 in the ai.onnx domain). The kernel supports any number of
// spatial dimensions; the cases below mirror the ``test_maxpool_*``
// reference cases in the ONNX test suite.
//
// Cases registered (each is the C++ analogue of the like-named ONNX
// reference case, with the ``test_cc_`` prefix):
//
//   * ``test_cc_maxpool_1d_default``
//   * ``test_cc_maxpool_2d_default``
//   * ``test_cc_maxpool_2d_strides``
//   * ``test_cc_maxpool_2d_ceil``
//   * ``test_cc_maxpool_2d_ceil_output_size_reduce_by_one``
//   * ``test_cc_maxpool_2d_dilations``
//   * ``test_cc_maxpool_2d_pads``
//   * ``test_cc_maxpool_2d_precomputed_pads``
//   * ``test_cc_maxpool_2d_precomputed_same_upper``
//   * ``test_cc_maxpool_2d_precomputed_strides``
//   * ``test_cc_maxpool_2d_same_lower``
//   * ``test_cc_maxpool_2d_same_upper``
//   * ``test_cc_maxpool_3d_default``
//   * ``test_cc_maxpool_3d_dilations``
//   * ``test_cc_maxpool_3d_dilations_use_ref_impl``
//   * ``test_cc_maxpool_3d_dilations_use_ref_impl_large``
//   * ``test_cc_maxpool_2d_uint8`` — UINT8 ramp; expected output is
//     hardcoded because :cpp:class:`kernel::MaxPool` only consumes FLOAT.
//   * ``test_cc_maxpool_with_argmax_2d_precomputed_pads`` — second output
//     ``Indices`` is verified.
//   * ``test_cc_maxpool_with_argmax_2d_precomputed_strides`` — second output
//     ``Indices`` is verified for ``storage_order = 1`` (column-major flat
//     indices). Expected values are hardcoded because
//     :cpp:class:`kernel::MaxPool` only supports row-major indices.
//
// Inputs that are deterministic in the reference suite (the precomputed
// cases and the small ceil/dilations cases) reuse the same ``1..N`` ramp;
// inputs that are random in the reference suite (default/strides/pads on
// 1x3x32x32 etc.) are drawn from :cpp:func:`Randn` with a fixed seed so the
// values are reproducible — those values do not match the upstream
// ``np.random.randn`` outputs but are still valid backend tests because the
// expected outputs are computed by this very kernel.
// ---------------------------------------------------------------------------
void RegisterMaxPoolCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(22);
  const kernel::KernelContext ctx{opset};
  const kernel::MaxPool maxpool_kernel{ctx};

  // 1-D MaxPool with a 2-wide kernel on a 1x3x32 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2});

    Tensor x = Tensor::FromFloat("", {1, 3, 32}, Randn<float>({1, 3, 32}, /*seed=*/1));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2});

    Expect(node, {x}, {y}, "test_cc_maxpool_1d_default", {opset}, "backend-test", registry);
  }

  // Default 2x2 kernel on a 1x3x32x32 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});

    Tensor x = Tensor::FromFloat("", {1, 3, 32, 32}, Randn<float>({1, 3, 32, 32}, /*seed=*/2));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_default", {opset}, "backend-test", registry);
  }

  // 5x5 kernel with strides (3, 3) on a 1x3x32x32 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3, 3});

    Tensor x = Tensor::FromFloat("", {1, 3, 32, 32}, Randn<float>({1, 3, 32, 32}, /*seed=*/3));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{3, 3});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_strides", {opset}, "backend-test", registry);
  }

  // 3x3 kernel with strides (2, 2) and ``ceil_mode = 1``.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<int64_t>(node, "ceil_mode", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4}, Range1ToN(16));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                              /*ceil_mode=*/true);

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_ceil", {opset}, "backend-test", registry);
  }

  // 1x1 kernel, strides (2, 2), ``ceil_mode = 1`` — exercises the case
  // where the naive ceil-mode formula would overshoot the input by one and
  // the kernel must reduce the output size accordingly.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {1, 1});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<int64_t>(node, "ceil_mode", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{1, 1}, /*strides=*/{2, 2}, /*pads=*/{},
                              /*ceil_mode=*/true);

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_ceil_output_size_reduce_by_one", {opset},
           "backend-test", registry);
  }

  // 2x2 kernel, dilations (2, 2), strides (1, 1).
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4}, Range1ToN(16));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{1, 1}, /*pads=*/{},
                              /*ceil_mode=*/false, /*dilations=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_dilations", {opset}, "backend-test", registry);
  }

  // 3x3 kernel with pads (2, 2, 2, 2) on a 1x3x28x28 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

    Tensor x = Tensor::FromFloat("", {1, 3, 28, 28}, Randn<float>({1, 3, 28, 28}, /*seed=*/4));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{1, 1},
                              /*pads=*/{2, 2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_pads", {opset}, "backend-test", registry);
  }

  // 5x5 kernel, pads (2, 2, 2, 2) on a 1x1x5x5 ramp ``1..25``.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, Range1ToN(25));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                              /*pads=*/{2, 2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_precomputed_pads", {opset}, "backend-test",
           registry);
  }

  // 3x3 kernel, strides (2, 2), ``auto_pad = SAME_UPPER`` on a 1x1x5x5 ramp.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {3, 3});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, Range1ToN(25));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{3, 3}, /*strides=*/{2, 2}, /*pads=*/{},
                              /*ceil_mode=*/false, /*dilations=*/{}, /*storage_order=*/0,
                              /*auto_pad=*/"SAME_UPPER");

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_precomputed_same_upper", {opset}, "backend-test",
           registry);
  }

  // 2x2 kernel, strides (2, 2) on a 1x1x5x5 ramp.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, Range1ToN(25));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_precomputed_strides", {opset}, "backend-test",
           registry);
  }

  // 2x2 kernel, ``auto_pad = SAME_LOWER`` on a 1x3x32x32 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", "SAME_LOWER");

    Tensor x = Tensor::FromFloat("", {1, 3, 32, 32}, Randn<float>({1, 3, 32, 32}, /*seed=*/5));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                              /*ceil_mode=*/false, /*dilations=*/{}, /*storage_order=*/0,
                              /*auto_pad=*/"SAME_LOWER");

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_same_lower", {opset}, "backend-test", registry);
  }

  // 2x2 kernel, ``auto_pad = SAME_UPPER`` on a 1x3x32x32 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::string>(node, "auto_pad", "SAME_UPPER");

    Tensor x = Tensor::FromFloat("", {1, 3, 32, 32}, Randn<float>({1, 3, 32, 32}, /*seed=*/6));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2}, /*strides=*/{}, /*pads=*/{},
                              /*ceil_mode=*/false, /*dilations=*/{}, /*storage_order=*/0,
                              /*auto_pad=*/"SAME_UPPER");

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_same_upper", {opset}, "backend-test", registry);
  }

  // 3-D MaxPool, 2x2x2 kernel on a 1x3x32x32x32 input.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});

    Tensor x =
        Tensor::FromFloat("", {1, 3, 32, 32, 32}, Randn<float>({1, 3, 32, 32, 32}, /*seed=*/7));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_3d_default", {opset}, "backend-test", registry);
  }

  // 5x5 kernel, pads (2, 2, 2, 2), ``Indices`` second output verified.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, Range1ToN(25));
    auto yz = maxpool_kernel.WithIndices(x, /*kernel_shape=*/{5, 5}, /*strides=*/{1, 1},
                                         /*pads=*/{2, 2, 2, 2});

    Expect(node, {x}, {yz.first, yz.second}, "test_cc_maxpool_with_argmax_2d_precomputed_pads",
           {opset}, "backend-test", registry);
  }

  // 3-D MaxPool, 2x2x2 kernel, strides (1, 1, 1), dilations (2, 2, 2) on a
  // 1x1x4x4x4 input where each 4x4 slice is the same ``1..16`` ramp. Mirrors
  // the ONNX ``test_maxpool_3d_dilations`` reference case.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2, 2});

    std::vector<float> data;
    data.reserve(64);
    for (int slice = 0; slice < 4; ++slice) {
      for (int v = 1; v <= 16; ++v) {
        data.push_back(static_cast<float>(v));
      }
    }
    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4, 4}, data);
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2, 2}, /*strides=*/{1, 1, 1},
                              /*pads=*/{}, /*ceil_mode=*/false, /*dilations=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_3d_dilations", {opset}, "backend-test", registry);
  }

  // Same configuration as ``test_cc_maxpool_3d_dilations`` but mirroring the
  // ONNX ``..._use_ref_impl`` reference case (which also exercises the
  // reference ``pool`` helper). Inputs and expected outputs are identical
  // since the reference helper agrees with our kernel when no padding is
  // required.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {1, 1, 1});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2, 2});

    std::vector<float> data;
    data.reserve(64);
    for (int slice = 0; slice < 4; ++slice) {
      for (int v = 1; v <= 16; ++v) {
        data.push_back(static_cast<float>(v));
      }
    }
    Tensor x = Tensor::FromFloat("", {1, 1, 4, 4, 4}, data);
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{2, 2, 2}, /*strides=*/{1, 1, 1},
                              /*pads=*/{}, /*ceil_mode=*/false, /*dilations=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_3d_dilations_use_ref_impl", {opset}, "backend-test",
           registry);
  }

  // Large 3-D MaxPool with dilations and ceil_mode, mirroring the ONNX
  // ``test_maxpool_3d_dilations_use_ref_impl_large`` reference case. Inputs
  // are drawn from :cpp:func:`Randn` with a fixed seed so the case stays
  // reproducible; the expected output is computed by this very kernel.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5, 5});
    AddAttribute<std::vector<int64_t>>(node, "strides", {3, 3, 3});
    AddAttribute<std::vector<int64_t>>(node, "dilations", {2, 2, 2});
    AddAttribute<int64_t>(node, "ceil_mode", 1);

    Tensor x =
        Tensor::FromFloat("", {1, 1, 32, 32, 32}, Randn<float>({1, 1, 32, 32, 32}, /*seed=*/8));
    Tensor y = maxpool_kernel(x, /*kernel_shape=*/{5, 5, 5}, /*strides=*/{3, 3, 3},
                              /*pads=*/{}, /*ceil_mode=*/true, /*dilations=*/{2, 2, 2});

    Expect(node, {x}, {y}, "test_cc_maxpool_3d_dilations_use_ref_impl_large", {opset},
           "backend-test", registry);
  }

  // 5x5 kernel, pads (2, 2, 2, 2) on a 1x1x5x5 ``1..25`` ramp stored as
  // UINT8. The expected output mirrors ``test_cc_maxpool_2d_precomputed_pads``
  // but in UINT8; it is hardcoded because :cpp:class:`kernel::MaxPool` only
  // accepts FLOAT inputs.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {5, 5});
    AddAttribute<std::vector<int64_t>>(node, "pads", {2, 2, 2, 2});

    std::vector<uint8_t> xv;
    xv.reserve(25);
    for (int v = 1; v <= 25; ++v) {
      xv.push_back(static_cast<uint8_t>(v));
    }
    Tensor x = Tensor::FromUint8("", {1, 1, 5, 5}, xv);
    Tensor y =
        Tensor::FromUint8("", {1, 1, 5, 5}, {13, 14, 15, 15, 15, 18, 19, 20, 20, 20, 23, 24, 25,
                                             25, 25, 23, 24, 25, 25, 25, 23, 24, 25, 25, 25});

    Expect(node, {x}, {y}, "test_cc_maxpool_2d_uint8", {opset}, "backend-test", registry);
  }

  // 2x2 kernel, strides (2, 2), ``storage_order = 1`` (column-major
  // indices), ``Indices`` second output verified. Expected values are
  // hardcoded since :cpp:class:`kernel::MaxPool` only supports row-major
  // indices.
  {
    NodeProto node;
    node.set_op_type("MaxPool");
    node.add_input("x");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<std::vector<int64_t>>(node, "kernel_shape", {2, 2});
    AddAttribute<std::vector<int64_t>>(node, "strides", {2, 2});
    AddAttribute<int64_t>(node, "storage_order", 1);

    Tensor x = Tensor::FromFloat("", {1, 1, 5, 5}, Range1ToN(25));
    Tensor y = Tensor::FromFloat("", {1, 1, 2, 2}, {7.0f, 9.0f, 17.0f, 19.0f});
    Tensor z = Tensor::FromInt64("", {1, 1, 2, 2}, {6, 16, 8, 18});

    Expect(node, {x}, {y, z}, "test_cc_maxpool_with_argmax_2d_precomputed_strides", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

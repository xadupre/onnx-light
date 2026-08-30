// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_extensions/backend_test/cases/math/include_math_cases.h"
#include "onnx_extensions/kernels/kernels/math/include_math_kernels.h"

#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

// Builds an Einsum NodeProto for ``n_inputs`` input names and the given
// ``equation`` attribute.
NodeProto MakeEinsumNode(int n_inputs, const std::string &equation) {
  NodeProto node;
  node.set_op_type("Einsum");
  for (int i = 0; i < n_inputs; ++i) {
    node.add_input("X" + std::to_string(i));
  }
  node.add_output("Y");

  AttributeProto *attr = node.add_attribute();
  attr->set_name("equation");
  attr->set_type(AttributeProto::STRING);
  attr->set_s(equation);
  return node;
}

} // namespace

// ---------------------------------------------------------------------------
// Einsum — Einstein summation (since opset 12).
// ---------------------------------------------------------------------------
void RegisterEinsumCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(13);
  const auto einsum_kernel = MakeReferenceKernel<onnx_kernels::kernel::Einsum>(opset);

  if (mode == TestMode::BENCHMARK) {
    const std::string eq = "ij,jk->ik";
    NodeProto node = MakeEinsumNode(2, eq);
    const std::vector<int64_t> shape = {512, 512};
    const int64_t count = 512 * 512;
    Expect(registry, std::move(node), "test_cc_einsum_benchmark", {opset}, {count, count}, {count},
           [einsum_kernel, eq, shape]() -> IoData {
             Tensor a = RandnTensor(DataType::FLOAT, shape, 440);
             Tensor b = RandnTensor(DataType::FLOAT, shape, 441);
             Tensor z =
                 einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
             return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
           });
    return;
  }

  // Transpose: "ij->ji" (explicit).
  {
    const std::string eq = "ij->ji";
    NodeProto node = MakeEinsumNode(1, eq);
    Expect(registry, std::move(node), "test_cc_einsum_transpose", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({x}, eq); });
      return IoData{{std::move(x)}, {std::move(z)}};
    });
  }

  // Trace: "ii->" (explicit scalar).
  {
    const std::string eq = "ii->";
    NodeProto node = MakeEinsumNode(1, eq);
    Expect(registry, std::move(node), "test_cc_einsum_trace", {opset}, [=]() -> IoData {
      Tensor x =
          Tensor::FromFloat("", {3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({x}, eq); });
      return IoData{{std::move(x)}, {std::move(z)}};
    });
  }

  // Sum over an axis: "ij->i" (explicit).
  {
    const std::string eq = "ij->i";
    NodeProto node = MakeEinsumNode(1, eq);
    Expect(registry, std::move(node), "test_cc_einsum_sum_axis", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({x}, eq); });
      return IoData{{std::move(x)}, {std::move(z)}};
    });
  }

  // Matrix multiplication: "ij,jk->ik".
  {
    const std::string eq = "ij,jk->ik";
    NodeProto node = MakeEinsumNode(2, eq);
    Expect(registry, std::move(node), "test_cc_einsum_matmul_2d", {opset}, [=]() -> IoData {
      Tensor a = Tensor::FromFloat("", {2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
      Tensor b = Tensor::FromFloat("", {3, 2}, {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
      return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
    });
  }

  // Batched matrix multiplication: "bij,bjk->bik".
  {
    const std::string eq = "bij,bjk->bik";
    NodeProto node = MakeEinsumNode(2, eq);
    Expect(registry, std::move(node), "test_cc_einsum_batch_matmul", {opset}, [=]() -> IoData {
      Tensor a = Tensor::FromFloat("", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
      Tensor b = Tensor::FromFloat(
          "", {2, 3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
      return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
    });
  }

  // Inner product: "i,i->" (explicit scalar).
  {
    const std::string eq = "i,i->";
    NodeProto node = MakeEinsumNode(2, eq);
    Expect(registry, std::move(node), "test_cc_einsum_inner", {opset}, [=]() -> IoData {
      Tensor a = Tensor::FromFloat("", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor b = Tensor::FromFloat("", {4}, {5.0f, 6.0f, 7.0f, 8.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
      return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
    });
  }

  // Outer product: "i,j->ij" (explicit).
  {
    const std::string eq = "i,j->ij";
    NodeProto node = MakeEinsumNode(2, eq);
    Expect(registry, std::move(node), "test_cc_einsum_outer", {opset}, [=]() -> IoData {
      Tensor a = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
      Tensor b = Tensor::FromFloat("", {2}, {4.0f, 5.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
      return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
    });
  }

  // Implicit mode: "ij" — output keeps both labels (alphabetical order).
  {
    const std::string eq = "ij";
    NodeProto node = MakeEinsumNode(1, eq);
    Expect(registry, std::move(node), "test_cc_einsum_implicit_identity", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({x}, eq); });
      return IoData{{std::move(x)}, {std::move(z)}};
    });
  }

  // Ellipsis batch matmul: "...ij,...jk->...ik".
  {
    const std::string eq = "...ij,...jk->...ik";
    NodeProto node = MakeEinsumNode(2, eq);
    Expect(registry, std::move(node), "test_cc_einsum_ellipsis_matmul", {opset}, [=]() -> IoData {
      Tensor a = Tensor::FromFloat("", {2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
      Tensor b = Tensor::FromFloat(
          "", {2, 3, 2}, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 2.0f, 1.0f, 1.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
      return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
    });
  }

  // Batch diagonal: "...ii ->...i" — mirrors ONNX ``test_einsum_batch_diagonal``.
  {
    const std::string eq = "...ii ->...i";
    NodeProto node = MakeEinsumNode(1, eq);
    Expect(registry, std::move(node), "test_cc_einsum_batch_diagonal", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {2, 3, 3},
                                   {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f,
                                    11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({x}, eq); });
      return IoData{{std::move(x)}, {std::move(z)}};
    });
  }

  // Implicit inner product: "i,i" — mirrors ONNX ``test_einsum_inner_prod``
  // (no ``->`` so the scalar result is implicit).
  {
    const std::string eq = "i,i";
    NodeProto node = MakeEinsumNode(2, eq);
    Expect(registry, std::move(node), "test_cc_einsum_inner_prod", {opset}, [=]() -> IoData {
      Tensor a = Tensor::FromFloat("", {5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
      Tensor b = Tensor::FromFloat("", {5}, {6.0f, 7.0f, 8.0f, 9.0f, 10.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({a, b}, eq); });
      return IoData{{std::move(a), std::move(b)}, {std::move(z)}};
    });
  }

  // Scalar identity: "->" — mirrors ONNX ``test_einsum_scalar`` (a 0-D input
  // is returned as-is).
  {
    const std::string eq = "->";
    NodeProto node = MakeEinsumNode(1, eq);
    Expect(registry, std::move(node), "test_cc_einsum_scalar", {opset}, [=]() -> IoData {
      Tensor x = Tensor::FromFloat("", {}, {5.0f});
      Tensor z = einsum_kernel.Invoke([&](const auto &kernel) { return kernel({x}, eq); });
      return IoData{{std::move(x)}, {std::move(z)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

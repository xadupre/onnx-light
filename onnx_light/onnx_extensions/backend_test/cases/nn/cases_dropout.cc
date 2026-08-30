// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_extensions/backend_test/cases/nn/include_nn_cases.h"
#include "onnx_extensions/kernels/kernels/nn/include_nn_kernels.h"

#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterDropoutCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(22);

  if (mode == TestMode::BENCHMARK) {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("data");
    node.add_output("output");

    Expect(registry, std::move(node), "test_cc_dropout_default_inference_benchmark", {opset},
           {kBenchmarkElementwiseSize}, {kBenchmarkElementwiseSize}, []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext dropout_kernel_ctx{opset};
             const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

             Tensor data = RandnTensor(DataType::FLOAT, {kBenchmarkElementwiseSize}, 1601);
             Tensor mask("", static_cast<int32_t>(DataType::BOOL), data.shape,
                         std::vector<uint8_t>(static_cast<size_t>(kBenchmarkElementwiseSize), 1));
             Tensor output = dropout_kernel(data, /*ratio=*/0.5f,
                                            /*training_mode=*/false, mask,
                                            onnx_kernels::kernel::Dropout::kNoSeed);
             return IoData{{std::move(data)}, {std::move(output)}};
           });
    return;
  }

  // Inference-mode Dropout (default training_mode=false): output copies input.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("data");
    node.add_output("output");
    Expect(registry, std::move(node), "test_cc_dropout_default_inference", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
      Tensor mask("", static_cast<int32_t>(DataType::BOOL), data.shape, std::vector<uint8_t>(6, 1));
      Tensor output = dropout_kernel(data, /*ratio=*/0.5f, /*training_mode=*/false, mask,
                                     onnx_kernels::kernel::Dropout::kNoSeed);
      return IoData{{std::move(data)}, {std::move(output)}};
    });
  }

  // Training-mode Dropout with ratio/training_mode inputs and mask output.
  // Use ratio=0 so expected outputs are deterministic across runtimes.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("data");
    node.add_input("ratio");
    node.add_input("training_mode");
    node.add_output("output");
    node.add_output("mask");
    AddAttribute<int64_t>(node, "seed", 123);
    Expect(registry, std::move(node), "test_cc_dropout_training_mask", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor data = Tensor::FromFloat("", {2, 3}, {1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f});
      Tensor ratio = Tensor::FromFloat("", {}, {0.0f});
      Tensor training_mode = Tensor::FromBool("", {}, {1});
      auto produced = dropout_kernel(data, /*ratio=*/0.0f, /*training_mode=*/true, /*seed=*/123);

      return IoData{{std::move(data), std::move(ratio), std::move(training_mode)},
                    {std::move(produced.first), std::move(produced.second)}};
    });
  }

  // ---------------------------------------------------------------------------
  // Upstream ONNX backend tests for the ``Dropout`` operator (mirror
  // ``onnx.backend.test.case.node.dropout.Dropout``). Inputs are generated
  // deterministically via :cpp:func:`Randn` (mirroring the upstream
  // ``np.random.randn(...)`` pattern); expected outputs are computed with
  // :cpp:class:`kernel::Dropout`.
  // ---------------------------------------------------------------------------

  // Inference Dropout with an explicit ratio input (training_mode defaults to
  // false so the ratio has no effect on the output).
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_output("y");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_dropout_default_ratio", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor r = Tensor::FromFloat("", {}, {0.1f});
      Tensor y = dropout_kernel(x, /*ratio=*/0.1f, /*training_mode=*/false).first;
      return IoData{{std::move(x), std::move(r)}, {std::move(y)}};
    });
  }

  // Inference Dropout with mask output, no ratio input.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_dropout_default_mask", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      auto produced = dropout_kernel(x, /*ratio=*/0.5f, /*training_mode=*/false);
      return IoData{{std::move(x)}, {std::move(produced.first), std::move(produced.second)}};
    });
  }

  // Inference Dropout with both mask output and ratio input.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_dropout_default_mask_ratio", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor r = Tensor::FromFloat("", {}, {0.1f});
      auto produced = dropout_kernel(x, /*ratio=*/0.1f, /*training_mode=*/false);
      return IoData{{std::move(x), std::move(r)},
                    {std::move(produced.first), std::move(produced.second)}};
    });
  }

  // Training-mode Dropout, default ratio 0.5 (single output ``y``). The
  // expected output is whatever ``kernel::Dropout`` produces for the given
  // seed; the actual values depend on the kernel's RNG and are not expected
  // to match the upstream ONNX recorded outputs bit-for-bit.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_input("t");
    node.add_output("y");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_training_dropout_default", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor r = Tensor::FromFloat("", {}, {0.5f});
      Tensor t = Tensor::FromBool("", {}, {1});
      auto produced = dropout_kernel(x, /*ratio=*/0.5f, /*training_mode=*/true, /*seed=*/0);
      return IoData{{std::move(x), std::move(r), std::move(t)}, {std::move(produced.first)}};
    });
  }

  // Training-mode Dropout, default ratio 0.5, with mask output.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_input("t");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_training_dropout_default_mask", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext dropout_kernel_ctx{opset};
             const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
             Tensor r = Tensor::FromFloat("", {}, {0.5f});
             Tensor t = Tensor::FromBool("", {}, {1});
             auto produced = dropout_kernel(x, /*ratio=*/0.5f, /*training_mode=*/true, /*seed=*/0);
             return IoData{{std::move(x), std::move(r), std::move(t)},
                           {std::move(produced.first), std::move(produced.second)}};
           });
  }

  // Training-mode Dropout, ratio 0.75 (single output).
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_input("t");
    node.add_output("y");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_training_dropout", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor r = Tensor::FromFloat("", {}, {0.75f});
      Tensor t = Tensor::FromBool("", {}, {1});
      auto produced = dropout_kernel(x, /*ratio=*/0.75f, /*training_mode=*/true, /*seed=*/0);
      return IoData{{std::move(x), std::move(r), std::move(t)}, {std::move(produced.first)}};
    });
  }

  // Training-mode Dropout, ratio 0.75 with mask output.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_input("t");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_training_dropout_mask", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor r = Tensor::FromFloat("", {}, {0.75f});
      Tensor t = Tensor::FromBool("", {}, {1});
      auto produced = dropout_kernel(x, /*ratio=*/0.75f, /*training_mode=*/true, /*seed=*/0);
      return IoData{{std::move(x), std::move(r), std::move(t)},
                    {std::move(produced.first), std::move(produced.second)}};
    });
  }

  // Training-mode Dropout, ratio 0.0 (output equals input, mask is all ones).
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_input("t");
    node.add_output("y");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_training_dropout_zero_ratio", {opset}, []() -> IoData {
      const OpsetId opset = DefaultOpset(22);

      const KernelContext dropout_kernel_ctx{opset};
      const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor r = Tensor::FromFloat("", {}, {0.0f});
      Tensor t = Tensor::FromBool("", {}, {1});
      auto produced = dropout_kernel(x, /*ratio=*/0.0f, /*training_mode=*/true, /*seed=*/0);
      return IoData{{std::move(x), std::move(r), std::move(t)}, {std::move(produced.first)}};
    });
  }

  // Training-mode Dropout, ratio 0.0 with mask output.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_input("r");
    node.add_input("t");
    node.add_output("y");
    node.add_output("z");
    AddAttribute<int64_t>(node, "seed", 0);
    Expect(registry, std::move(node), "test_training_dropout_zero_ratio_mask", {opset},
           []() -> IoData {
             const OpsetId opset = DefaultOpset(22);

             const KernelContext dropout_kernel_ctx{opset};
             const onnx_kernels::kernel::Dropout dropout_kernel{dropout_kernel_ctx};

             Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
             Tensor r = Tensor::FromFloat("", {}, {0.0f});
             Tensor t = Tensor::FromBool("", {}, {1});
             auto produced = dropout_kernel(x, /*ratio=*/0.0f, /*training_mode=*/true, /*seed=*/0);
             return IoData{{std::move(x), std::move(r), std::move(t)},
                           {std::move(produced.first), std::move(produced.second)}};
           });
  }

  // ---------------------------------------------------------------------------
  // Legacy Dropout (opset 11): inference-only, no ``training_mode`` input and
  // output equals the input regardless of the optional ``ratio`` attribute.
  // ---------------------------------------------------------------------------
  const OpsetId opset_old = DefaultOpset(11);

  // Opset 11 Dropout, no ratio attribute.
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_output("y");
    Expect(registry, std::move(node), "test_dropout_default_old", {opset_old}, []() -> IoData {
      const OpsetId opset_old = DefaultOpset(11);

      const KernelContext dropout_kernel_old_ctx{opset_old};
      const onnx_kernels::kernel::Dropout dropout_kernel_old{dropout_kernel_old_ctx};

      Tensor x = Tensor::FromFloat("", {3}, {-1.0f, 0.0f, 1.0f});
      Tensor y = dropout_kernel_old(x, /*ratio=*/0.0f, /*training_mode=*/false).first;
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }

  // Opset 11 Dropout with ratio attribute (still inference-only, so the
  // expected output equals the input).
  {
    NodeProto node;
    node.set_op_type("Dropout");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<float>(node, "ratio", 0.2f);
    Expect(registry, std::move(node), "test_dropout_random_old", {opset_old}, []() -> IoData {
      const OpsetId opset_old = DefaultOpset(11);

      const KernelContext dropout_kernel_old_ctx{opset_old};
      const onnx_kernels::kernel::Dropout dropout_kernel_old{dropout_kernel_old_ctx};

      Tensor x = RandnTensor(DataType::FLOAT, {3, 4, 5}, /*seed=*/0);
      Tensor y = dropout_kernel_old(x, /*ratio=*/0.2f, /*training_mode=*/false).first;
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

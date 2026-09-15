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

// ---------------------------------------------------------------------------
// RotaryEmbedding — applies rotary positional embeddings (RoPE) to ``X``.
// Output shape and dtype match the input.
//
// The cases below mirror the variants exposed by the upstream ONNX backend
// test suite. Each case produces an output by invoking the local
// ``kernel::RotaryEmbedding`` reference implementation so the recorded
// expected output is self-consistent with this library's kernel.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeRotaryNode(const std::vector<std::string> &inputs,
                         const std::vector<std::string> &outputs) {
  NodeProto node;
  node.set_op_type("RotaryEmbedding");
  for (const auto &name : inputs) {
    node.add_input(name);
  }
  for (const auto &name : outputs) {
    node.add_output(name);
  }
  return node;
}

} // namespace

void RegisterRotaryEmbeddingCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset = DefaultOpset(23);

  if (mode == TestMode::BENCHMARK) {
    const int64_t batch = 8;
    const int64_t num_heads = 16;
    const int64_t seq = 256;
    const int64_t head_size = 64;
    const int64_t half = head_size / 2;
    const int64_t max_pos = 512;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache", "position_ids"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_rotary_embedding_benchmark", {opset},
           {batch * num_heads * seq * head_size, max_pos * half, max_pos * half, batch * seq},
           {batch * num_heads * seq * head_size}, []() -> IoData {
             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             const std::vector<int64_t> x_shape = {batch, num_heads, seq, head_size};
             const std::vector<int64_t> cache_shape = {max_pos, half};
             Tensor X = RandnTensor(DataType::FLOAT, x_shape, 2001);
             Tensor cos_cache = RandnTensor(DataType::FLOAT, cache_shape, 2002);
             Tensor sin_cache = RandnTensor(DataType::FLOAT, cache_shape, 2003);
             std::vector<int64_t> pos(static_cast<size_t>(batch * seq));
             for (int64_t b = 0; b < batch; ++b) {
               for (int64_t s = 0; s < seq; ++s) {
                 pos[static_cast<size_t>(b * seq + s)] = s;
               }
             }
             Tensor position_ids = Tensor::FromInt64("", {batch, seq}, pos);
             onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
             Tensor Y = kernel(X, cos_cache, sin_cache, position_ids, attrs);
             return IoData{{std::move(X), std::move(cos_cache), std::move(sin_cache),
                            std::move(position_ids)},
                           {std::move(Y)}};
           });
    return;
  }

  // 4D X = (batch=1, num_heads=2, seq=3, head_size=4); position_ids drive
  // selection from a (max_pos_plus_1=5, head_size/2=2) cos/sin cache.
  // Used by the "default" and "interleaved" variants.
  Tensor X4 = Tensor::FromFloat("", {1, 2, 3, 4}, {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                                   0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                                   1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
  Tensor cos2 = Tensor::FromFloat("", {5, 2},
                                  {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f, -0.5f, -0.4f, -1.0f, -0.8f});
  Tensor sin2 =
      Tensor::FromFloat("", {5, 2}, {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f, 0.9f, 0.8f, 0.5f, 0.4f});
  Tensor position_ids = Tensor::FromInt64("", {1, 3}, {0, 2, 4});

  // Case 1: default attributes (interleaved=0, full rotation).
  {
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache", "position_ids"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_rotary_embedding", {opset}, [attrs]() -> IoData {
      Tensor X4 =
          Tensor::FromFloat("", {1, 2, 3, 4}, {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                               0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                               1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
      Tensor cos2 = Tensor::FromFloat(
          "", {5, 2}, {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f, -0.5f, -0.4f, -1.0f, -0.8f});
      Tensor sin2 = Tensor::FromFloat("", {5, 2},
                                      {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f, 0.9f, 0.8f, 0.5f, 0.4f});
      Tensor position_ids = Tensor::FromInt64("", {1, 3}, {0, 2, 4});

      const OpsetId opset = DefaultOpset(23);

      const KernelContext kernel_ctx{opset};
      const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

      Tensor Y = kernel(X4, cos2, sin2, position_ids, attrs);
      return IoData{{std::move(X4), std::move(cos2), std::move(sin2), std::move(position_ids)},
                    {std::move(Y)}};
    });
  }

  // Case 2: interleaved=1.
  {
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    attrs.interleaved = true;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache", "position_ids"}, {"Y"});
    AddAttribute<int64_t>(node, "interleaved", 1);
    Expect(
        registry, std::move(node), "test_cc_rotary_embedding_interleaved", {opset},
        [attrs]() -> IoData {
          Tensor X4 =
              Tensor::FromFloat("", {1, 2, 3, 4}, {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                                   0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                                   1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
          Tensor cos2 = Tensor::FromFloat(
              "", {5, 2}, {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f, -0.5f, -0.4f, -1.0f, -0.8f});
          Tensor sin2 = Tensor::FromFloat(
              "", {5, 2}, {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f, 0.9f, 0.8f, 0.5f, 0.4f});
          Tensor position_ids = Tensor::FromInt64("", {1, 3}, {0, 2, 4});

          const OpsetId opset = DefaultOpset(23);

          const KernelContext kernel_ctx{opset};
          const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

          Tensor Y = kernel(X4, cos2, sin2, position_ids, attrs);
          return IoData{{std::move(X4), std::move(cos2), std::move(sin2), std::move(position_ids)},
                        {std::move(Y)}};
        });
  }

  // Case 3: rank-3 X = (batch, seq, hidden_size = num_heads * head_size)
  // with num_heads attribute set. hidden_size=8 = num_heads(2) * head_size(4).
  {
    Tensor X3 = Tensor::FromFloat("", {1, 3, 8}, {0.0f, 0.1f, 0.2f, 0.3f, 1.2f, 1.3f, 1.4f, 1.5f,
                                                  0.4f, 0.5f, 0.6f, 0.7f, 1.6f, 1.7f, 1.8f, 1.9f,
                                                  0.8f, 0.9f, 1.0f, 1.1f, 2.0f, 2.1f, 2.2f, 2.3f});
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    attrs.num_heads = 2;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache", "position_ids"}, {"Y"});
    AddAttribute<int64_t>(node, "num_heads", 2);
    Expect(registry, std::move(node), "test_cc_rotary_embedding_3d_input", {opset},
           [attrs]() -> IoData {
             Tensor cos2 = Tensor::FromFloat(
                 "", {5, 2}, {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f, -0.5f, -0.4f, -1.0f, -0.8f});
             Tensor sin2 = Tensor::FromFloat(
                 "", {5, 2}, {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f, 0.9f, 0.8f, 0.5f, 0.4f});
             Tensor position_ids = Tensor::FromInt64("", {1, 3}, {0, 2, 4});
             Tensor X3 =
                 Tensor::FromFloat("", {1, 3, 8}, {0.0f, 0.1f, 0.2f, 0.3f, 1.2f, 1.3f, 1.4f, 1.5f,
                                                   0.4f, 0.5f, 0.6f, 0.7f, 1.6f, 1.7f, 1.8f, 1.9f,
                                                   0.8f, 0.9f, 1.0f, 1.1f, 2.0f, 2.1f, 2.2f, 2.3f});

             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             Tensor Y = kernel(X3, cos2, sin2, position_ids, attrs);
             return IoData{
                 {std::move(X3), std::move(cos2), std::move(sin2), std::move(position_ids)},
                 {std::move(Y)}};
           });
  }

  // Case 4: no position_ids — cos/sin caches are rank-3
  // (batch, seq, head_size/2) and indexed positionally.
  Tensor cos3 = Tensor::FromFloat("", {1, 3, 2}, {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f});
  Tensor sin3 = Tensor::FromFloat("", {1, 3, 2}, {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f});
  {
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    Tensor empty;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache"}, {"Y"});
    Expect(registry, std::move(node), "test_cc_rotary_embedding_no_position_ids", {opset},
           [attrs]() -> IoData {
             Tensor X4 = Tensor::FromFloat("", {1, 2, 3, 4},
                                           {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                            0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                            1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
             Tensor cos3 = Tensor::FromFloat("", {1, 3, 2}, {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f});
             Tensor sin3 = Tensor::FromFloat("", {1, 3, 2}, {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f});
             Tensor empty;

             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             Tensor Y = kernel(X4, cos3, sin3, empty, attrs);
             return IoData{{std::move(X4), std::move(cos3), std::move(sin3)}, {std::move(Y)}};
           });
  }

  // Case 5: no position_ids + interleaved=1.
  {
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    attrs.interleaved = true;
    Tensor empty;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache"}, {"Y"});
    AddAttribute<int64_t>(node, "interleaved", 1);
    Expect(registry, std::move(node), "test_cc_rotary_embedding_no_position_ids_interleaved",
           {opset}, [attrs]() -> IoData {
             Tensor X4 = Tensor::FromFloat("", {1, 2, 3, 4},
                                           {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                            0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                            1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
             Tensor cos3 = Tensor::FromFloat("", {1, 3, 2}, {1.0f, 1.0f, 0.5f, 0.6f, 0.0f, 0.2f});
             Tensor sin3 = Tensor::FromFloat("", {1, 3, 2}, {0.0f, 0.1f, 0.3f, 0.4f, 0.7f, 0.6f});
             Tensor empty;

             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             Tensor Y = kernel(X4, cos3, sin3, empty, attrs);
             return IoData{{std::move(X4), std::move(cos3), std::move(sin3)}, {std::move(Y)}};
           });
  }

  // Case 6: partial rotation — rotary_embedding_dim < head_size. With
  // head_size=4 and rotary_embedding_dim=2 the cos/sin caches collapse to
  // (max_pos+1, 1) shapes, and the trailing 2 channels of each head are
  // passed through unchanged.
  {
    Tensor cos_partial = Tensor::FromFloat("", {5, 1}, {1.0f, 0.5f, 0.0f, -0.5f, -1.0f});
    Tensor sin_partial = Tensor::FromFloat("", {5, 1}, {0.0f, 0.3f, 0.7f, 0.9f, 0.5f});
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    attrs.rotary_embedding_dim = 2;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache", "position_ids"}, {"Y"});
    AddAttribute<int64_t>(node, "rotary_embedding_dim", 2);
    Expect(registry, std::move(node), "test_cc_rotary_embedding_with_rotary_dim", {opset},
           [attrs]() -> IoData {
             Tensor X4 = Tensor::FromFloat("", {1, 2, 3, 4},
                                           {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                            0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                            1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
             Tensor position_ids = Tensor::FromInt64("", {1, 3}, {0, 2, 4});
             Tensor cos_partial = Tensor::FromFloat("", {5, 1}, {1.0f, 0.5f, 0.0f, -0.5f, -1.0f});
             Tensor sin_partial = Tensor::FromFloat("", {5, 1}, {0.0f, 0.3f, 0.7f, 0.9f, 0.5f});

             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             Tensor Y = kernel(X4, cos_partial, sin_partial, position_ids, attrs);
             return IoData{{std::move(X4), std::move(cos_partial), std::move(sin_partial),
                            std::move(position_ids)},
                           {std::move(Y)}};
           });
  }

  // Case 7: partial rotation + interleaved=1.
  {
    Tensor cos_partial = Tensor::FromFloat("", {5, 1}, {1.0f, 0.5f, 0.0f, -0.5f, -1.0f});
    Tensor sin_partial = Tensor::FromFloat("", {5, 1}, {0.0f, 0.3f, 0.7f, 0.9f, 0.5f});
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    attrs.interleaved = true;
    attrs.rotary_embedding_dim = 2;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache", "position_ids"}, {"Y"});
    AddAttribute<int64_t>(node, "interleaved", 1);
    AddAttribute<int64_t>(node, "rotary_embedding_dim", 2);
    Expect(registry, std::move(node), "test_cc_rotary_embedding_with_interleaved_rotary_dim",
           {opset}, [attrs]() -> IoData {
             Tensor X4 = Tensor::FromFloat("", {1, 2, 3, 4},
                                           {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                            0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                            1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
             Tensor position_ids = Tensor::FromInt64("", {1, 3}, {0, 2, 4});
             Tensor cos_partial = Tensor::FromFloat("", {5, 1}, {1.0f, 0.5f, 0.0f, -0.5f, -1.0f});
             Tensor sin_partial = Tensor::FromFloat("", {5, 1}, {0.0f, 0.3f, 0.7f, 0.9f, 0.5f});

             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             Tensor Y = kernel(X4, cos_partial, sin_partial, position_ids, attrs);
             return IoData{{std::move(X4), std::move(cos_partial), std::move(sin_partial),
                            std::move(position_ids)},
                           {std::move(Y)}};
           });
  }

  // Case 8: no position_ids + partial rotation. cos/sin caches are rank-3
  // (batch, seq, rotary_embedding_dim/2).
  {
    Tensor cos_p3 = Tensor::FromFloat("", {1, 3, 1}, {1.0f, 0.5f, 0.0f});
    Tensor sin_p3 = Tensor::FromFloat("", {1, 3, 1}, {0.0f, 0.3f, 0.7f});
    onnx_kernels::kernel::RotaryEmbedding::Attributes attrs;
    attrs.rotary_embedding_dim = 2;
    Tensor empty;
    NodeProto node = MakeRotaryNode({"X", "cos_cache", "sin_cache"}, {"Y"});
    AddAttribute<int64_t>(node, "rotary_embedding_dim", 2);
    Expect(registry, std::move(node), "test_cc_rotary_embedding_no_position_ids_rotary_dim",
           {opset}, [attrs]() -> IoData {
             Tensor X4 = Tensor::FromFloat("", {1, 2, 3, 4},
                                           {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f,
                                            0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f,
                                            1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f});
             Tensor cos_p3 = Tensor::FromFloat("", {1, 3, 1}, {1.0f, 0.5f, 0.0f});
             Tensor sin_p3 = Tensor::FromFloat("", {1, 3, 1}, {0.0f, 0.3f, 0.7f});
             Tensor empty;

             const OpsetId opset = DefaultOpset(23);

             const KernelContext kernel_ctx{opset};
             const onnx_kernels::kernel::RotaryEmbedding kernel{kernel_ctx};

             Tensor Y = kernel(X4, cos_p3, sin_p3, empty, attrs);
             return IoData{{std::move(X4), std::move(cos_p3), std::move(sin_p3)}, {std::move(Y)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

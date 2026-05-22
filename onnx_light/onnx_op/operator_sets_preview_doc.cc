// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_preview_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace preview {

std::string MakeFlexAttentionDoc() {
  return R"DOC(
Computes scaled dot-product attention over rank-4 (batched, multi-head) inputs,
with optional user-provided customization subgraphs at two stages:

1. score_mod: Modify the attention score tensor after Q·K^T
2. prob_mod: Modify the probability tensor after Softmax

This operator mirrors the capabilities of PyTorch's flex_attention:
https://docs.pytorch.org/docs/stable/nn.attention.flex_attention.html

Input Shapes (MUST be rank-4 tensors):
- Q: `(batch_size, q_num_heads, q_sequence_length, head_size)`
- K: `(batch_size, kv_num_heads, kv_sequence_length, head_size)`
- V: `(batch_size, kv_num_heads, kv_sequence_length, v_head_size)`

Output Shape:
- Y: `(batch_size, q_num_heads, q_sequence_length, v_head_size)`

FlexAttention Computation:
```
Scores = (Q @ K^T) * scale
Scores = score_mod(Scores)             # if 'score_mod' is provided
Probs = Softmax(Scores, axis=-1)
Probs = prob_mod(Probs)                # if 'prob_mod' is provided
Y = Probs @ V
```

Grouped Query Attention (GQA):
When `q_num_heads != kv_num_heads`, each K/V head is shared by a contiguous
group of query heads in head-index order. Let
`group_size = q_num_heads / kv_num_heads`; then query head `h` uses K/V head
`floor(h / group_size)`. `q_num_heads` must be a multiple of
`kv_num_heads`.

Modifier Subgraphs (score_mod, prob_mod):
Each modifier subgraph takes exactly one rank-4 tensor input and must produce
exactly one rank-4 tensor output of the same shape and element type.
- score_mod input/output shape: `(batch_size, q_num_heads, q_sequence_length, kv_sequence_length)`
- prob_mod  input/output shape: `(batch_size, q_num_heads, q_sequence_length, kv_sequence_length)`
The element type is determined by softmax_precision (defaults to float32 for
non-double inputs, otherwise double).

Masking can be expressed in score_mod by writing masked positions as -inf (or a
large negative value appropriate for the target precision).
)DOC";
}

} // namespace preview
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

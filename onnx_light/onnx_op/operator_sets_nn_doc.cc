// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_nn_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace nn {

namespace {

constexpr const char *kAveragePoolDocOpset1 = R"DOC(
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements exclude pad.
 )DOC";

constexpr const char *kAveragePoolDocOpset7 = R"DOC(
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)

 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 )DOC";

constexpr const char *kAveragePoolDocOpset10 = R"DOC(
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape will be following:
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - kernel_spatial_shape[i]) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled

 ```
 * pad_shape[i] is sum of pads along axis i
 ```

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - kernel_spatial_shape[i] + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + kernel_spatial_shape[i] - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 )DOC";

constexpr const char *kAveragePoolDocOpset19 = R"DOC(
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 )DOC";

constexpr const char *kAveragePoolDocOpset22 = R"DOC(
 AveragePool consumes an input tensor X and applies average pooling across
 the tensor according to kernel sizes, stride sizes, and pad lengths.
 average pooling consisting of computing the average on all values of a
 subset of the input tensor according to the kernel size and downsampling the
 data into the output tensor Y for further processing. The output spatial shape is calculated differently
 depending on whether explicit padding is used, where pads is employed, or auto padding is used, where auto_pad is utilized.
 With explicit padding (https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html?highlight=maxpool#torch.nn.MaxPool2d):
 ```
 output_spatial_shape[i] = floor((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 or
 ```
 output_spatial_shape[i] = ceil((input_spatial_shape[i] + pad_shape[i] - dilation[i] * (kernel_shape[i] - 1) - 1) / strides_spatial_shape[i] + 1)
 ```
 if ceil_mode is enabled. `pad_shape[i]` is the sum of pads along axis `i`. Sliding windows that would start in the right padded region are ignored.

 `auto_pad` is a DEPRECATED attribute. If you are using them currently, the output spatial shape will be following when ceil_mode is enabled:
 ```
 VALID: output_spatial_shape[i] = ceil((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) + 1) / strides_spatial_shape[i])
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = ceil(input_spatial_shape[i] / strides_spatial_shape[i])
 ```
 or when ceil_mode is disabled (https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D):
 ```
 VALID: output_spatial_shape[i] = floor((input_spatial_shape[i] - ((kernel_spatial_shape[i] - 1) * dilations[i] + 1)) / strides_spatial_shape[i]) + 1
 SAME_UPPER or SAME_LOWER: output_spatial_shape[i] = floor((input_spatial_shape[i] - 1) / strides_spatial_shape[i]) + 1
 ```
 And pad shape will be following if `SAME_UPPER` or `SAME_LOWER`:
 ```
 pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial_shape[i] + ((kernel_spatial_shape[i] - 1) * dilations[i] + 1) - input_spatial_shape[i]
 ```
 The output of each pooling window is divided by the number of elements (exclude pad when attribute count_include_pad is zero).
 )DOC";

} // namespace

std::string MakeAveragePoolDoc(int since_version) {
  switch (since_version) {
  case 1:
    return kAveragePoolDocOpset1;
  case 7:
    return kAveragePoolDocOpset7;
  case 10:
  case 11:
    return kAveragePoolDocOpset10;
  case 19:
    return kAveragePoolDocOpset19;
  case 22:
    return kAveragePoolDocOpset22;
  default:
    return "";
  }
}

namespace {

constexpr const char *kRNNDoc = R"DOC(
Computes an one-layer simple RNN. This operator is usually supported
via some custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `t` - time step (t-1 means previous time step)
* `Wi` - W parameter weight matrix for input gate
* `Ri` - R recurrence weight matrix for input gate
* `Wbi` - W parameter bias vector for input gate
* `Rbi` - R parameter bias vector for input gate
* `WBi` - W parameter weight matrix for backward input gate
* `RBi` - R recurrence weight matrix for backward input gate
* `WBbi` - WR bias vectors for backward input gate
* `RBbi` - RR bias vectors for backward input gate
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Tanh):

* Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
)DOC";

constexpr const char *kGRUDoc = R"DOC(
Computes an one-layer GRU. This operator is usually supported via some custom
implementation such as CuDNN.

Notations:

* `X` - input tensor
* `z` - update gate
* `r` - reset gate
* `h` - hidden gate
* `t` - time step (t-1 means previous time step)
* `W[zrh]` - W parameter weight matrix for update, reset, and hidden gates
* `R[zrh]` - R recurrence weight matrix for update, reset, and hidden gates
* `Wb[zrh]` - W bias vectors for update, reset, and hidden gates
* `Rb[zrh]` - R bias vectors for update, reset, and hidden gates
* `WB[zrh]` - W parameter weight matrix for backward update, reset, and hidden gates
* `RB[zrh]` - R recurrence weight matrix for backward update, reset, and hidden gates
* `WBb[zrh]` - W bias vectors for backward update, reset, and hidden gates
* `RBb[zrh]` - R bias vectors for backward update, reset, and hidden gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE:
  Below are optional

* Affine(x)              - alpha * x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha * Tanh(beta * x)
* HardSigmoid(x)         - min(max(alpha * x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha * (e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh):

* zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
* rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
* ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) # default, when linear_before_reset = 0
* ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) # when linear_before_reset != 0
* Ht = (1 - zt) (.) ht + zt (.) Ht-1
)DOC";

constexpr const char *kLSTMDoc = R"DOC(
Computes an one-layer LSTM. This operator is usually supported via some
custom implementation such as CuDNN.

Notations:

* `X` - input tensor
* `i` - input gate
* `o` - output gate
* `f` - forget gate
* `c` - cell gate
* `t` - time step (t-1 means previous time step)
* `W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
* `R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
* `Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
* `Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
* `P[iof]`  - P peephole weight vector for input, output, and forget gates
* `WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
* `RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
* `WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
* `RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
* `PB[iof]`  - P peephole weight vector for backward input, output, and forget gates
* `H` - Hidden state
* `num_directions` - 2 if direction == bidirectional else 1

Activation functions:

* Relu(x)                - max(0, x)
* Tanh(x)                - (1 - e^{-2x})/(1 + e^{-2x})
* Sigmoid(x)             - 1/(1 + e^{-x})

NOTE: Below are optional

* Affine(x)              - alpha*x + beta
* LeakyRelu(x)           - x if x >= 0 else alpha * x
* ThresholdedRelu(x)     - x if x >= alpha else 0
* ScaledTanh(x)          - alpha*Tanh(beta*x)
* HardSigmoid(x)         - min(max(alpha*x + beta, 0), 1)
* Elu(x)                 - x if x >= 0 else alpha*(e^x - 1)
* Softsign(x)            - x/(1 + |x|)
* Softplus(x)            - log(1 + e^x)

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):

* it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
* ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
* ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
* Ct = ft (.) Ct-1 + it (.) ct
* gt = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
* Ht = gt (.) h(Ct)
)DOC";

bool IsSupportedRecurrentVersion(int since_version, bool include_v3) {
  switch (since_version) {
  case 1:
  case 7:
  case 14:
  case 22:
    return true;
  case 3:
    return include_v3;
  default:
    return false;
  }
}

} // namespace

std::string MakeRNNDoc(int since_version) {
  return IsSupportedRecurrentVersion(since_version, /*include_v3=*/false) ? kRNNDoc : "";
}

std::string MakeGRUDoc(int since_version) {
  return IsSupportedRecurrentVersion(since_version, /*include_v3=*/true) ? kGRUDoc : "";
}

std::string MakeLSTMDoc(int since_version) {
  return IsSupportedRecurrentVersion(since_version, /*include_v3=*/false) ? kLSTMDoc : "";
}

namespace {

constexpr const char *kBatchNormalizationDocOpset1And6 = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)
    )DOC";

constexpr const char *kBatchNormalizationDocOpset7 = R"DOC(
    Carries out batch normalization as described in the paper
    https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
    there are multiple cases for the number of outputs, which we list below:

    Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
    Output case #2: Y (test mode)
        )DOC";

constexpr const char *kBatchNormalizationDocOpset9 = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
there are multiple cases for the number of outputs, which we list below:

Output case #1: Y, mean, var, saved_mean, saved_var (training mode)
Output case #2: Y (test mode)

For previous (depreciated) non-spatial cases, implementers are suggested
to flatten the input shape to (N x C*D1*D2 ..*Dn) before a BatchNormalization Op.
)DOC";

constexpr const char *kBatchNormalizationDocOpset14 = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

Output case #1: Y, running_mean, running_var (training_mode=True)
Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B

where:

current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)

Notice that ReduceVar refers to the population variance, and it equals to
sum(sqrd(x_i - x_avg)) / N
where N is the population size (this formula does not use sample size N - 1).

```

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementers are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
)DOC";

constexpr const char *kBatchNormalizationDocOpset15 = R"DOC(
Carries out batch normalization as described in the paper
https://arxiv.org/abs/1502.03167. Depending on the mode it is being run,
There are five required inputs 'X', 'scale', 'B', 'input_mean' and
'input_var'.
Note that 'input_mean' and 'input_var' are expected to be the estimated
statistics in inference mode (training_mode=False, default),
and the running statistics in training mode (training_mode=True).
There are multiple cases for the number of outputs, which we list below:

* Output case #1: Y, running_mean, running_var (training_mode=True)
* Output case #2: Y (training_mode=False)

When training_mode=False, extra outputs are invalid.
The outputs are updated as follows when training_mode=True:
```
running_mean = input_mean * momentum + current_mean * (1 - momentum)
running_var = input_var * momentum + current_var * (1 - momentum)

Y = (X - current_mean) / sqrt(current_var + epsilon) * scale + B
```
where:
```
current_mean = ReduceMean(X, axis=all_except_channel_index)
current_var =  ReduceVar(X, axis=all_except_channel_index)
```
Notice that `ReduceVar` refers to the population variance, and it equals to
`sum(sqrd(x_i - x_avg)) / N`
where `N` is the population size (this formula does not use sample size `N - 1`).

The computation of ReduceMean and ReduceVar uses float to avoid overflow for float16 inputs.

When training_mode=False:
```
Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B
```

For previous (depreciated) non-spatial cases, implementers are suggested
to flatten the input shape to (N x C * D1 * D2 * ... * Dn) before a BatchNormalization Op.
)DOC";

constexpr const char *kAttentionDocOpset23 = R"DOC(

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal attribute`, only one of which can be provided.
1) If `is_causal` is set to `1`, the attention masking is a lower triangular matrix when the mask is a square matrix. The attention masking has the form of the upper left causal bias due to the alignment.
2) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
  softcap (if provided)     |
            |               |
 at_mask---Add              |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```

)DOC";

constexpr const char *kAttentionDocOpset24 = R"DOC(

Computes scaled dot product attention on query, key and value tensors, using an optional attention mask if passed.

This operator covers self and cross variants of the attention operation based on sequence lengths of K, Q and V.

For self attention, `kv_sequence_length` equals to `q_sequence_length`.

For cross attention, query and key might have different lengths.

This operator also covers the 3 following variants based on the number of heads:
1) Multi-headed Attention (MHA): Described in the paper https://arxiv.org/pdf/1706.03762, `q_num_heads = kv_num_heads`.
2) Group-query Attention (GQA): Described in the paper https://arxiv.org/pdf/2305.13245, `q_num_heads > kv_num_heads`, `q_num_heads % kv_num_heads == 0`.
3) Multi-query Attention (MQA): Described in the paper https://arxiv.org/pdf/1911.02150, `q_num_heads > kv_num_heads`, `kv_num_heads=1`.

Attention bias to be added is calculated based on `attn_mask` input and `is_causal` attribute:
1) `attn_mask`: A boolean mask where a value of `True` indicates that the element should take part in attention or a float mask of the same type as query, key, value that is added to the attention score.
2) If `is_causal` is set to `1`, attention scores above the diagonal are masked out, regardless of the `attn_mask` input.

With respect to KV cache update, this operator allows the following two use cases:

1) Cache update happens inside the Attention operator. In this case, the `K` and `V` inputs contain only the incoming
tokens for the current autoregressive step, and the four optional inputs/outputs past and present key and value are
all needed. The Attention op performs a Concat operation on the past and incoming key and value to form the present
key and value, respectively. Note that this only works correctly for the special case where the past key and value
do not contain padded tokens.
2) Cache update happens outside the Attention operator (for example, through the `TensorScatter` operator). In this
case, the `K` and `V` inputs correspond to the entire cache tensor, so the four optional inputs/outputs past and
present key and value should not be used. An additional input `nonpad_kv_seqlen` of shape (batch_size,) may be
provided to indicate the number of non-padding tokens in each sample of the batch to save unnecessary computation.
Here, the kv_sequence dimension of `attn_mask` can be shorter than `K` and `V`, but still needs to be at least as long
as the maximum value of `nonpad_kv_seqlen`.

Both past and present state key/values are optional. They shall be used together, and not allowed to use only one of them.
The following pattern is applied to the Q, K and V inputs after appropriate reshaping of K and V inputs based on sequence lengths and num heads provided:

```
  The following pattern is applied by this operator:
      Q          K          V
      |          |          |
Q*sqrt(scale) K*sqrt(scale) |
      |          |          |
      |       Transpose     |
      |          |          |
      ---MatMul---          |
            |               |
  softcap (if provided)     |
            |               |
 at_mask---Add              |
            |               |
         Softmax            |
            |               |
            -----MatMul------
                   |
                   Y
```

)DOC";

} // namespace

std::string MakeBatchNormalizationDoc(int since_version) {
  switch (since_version) {
  case 1:
  case 6:
    return kBatchNormalizationDocOpset1And6;
  case 7:
    return kBatchNormalizationDocOpset7;
  case 9:
    return kBatchNormalizationDocOpset9;
  case 14:
    return kBatchNormalizationDocOpset14;
  case 15:
    return kBatchNormalizationDocOpset15;
  default:
    return "";
  }
}

std::string MakeAttentionDoc(int since_version) {
  switch (since_version) {
  case 23:
    return kAttentionDocOpset23;
  case 24:
    return kAttentionDocOpset24;
  default:
    return "";
  }
}

namespace {

constexpr const char *kGlobalAveragePoolDoc = R"DOC(
 GlobalAveragePool consumes an input tensor X and applies average pooling across
 the values in the same channel. This is equivalent to AveragePool with kernel size
 equal to the spatial dimension of input tensor.)DOC";

constexpr const char *kGlobalMaxPoolDoc = R"DOC(
 GlobalMaxPool consumes an input tensor X and applies max pooling across
 the values in the same channel. This is equivalent to MaxPool with kernel size
 equal to the spatial dimension of input tensor.)DOC";

constexpr const char *kGlobalLpPoolDoc = R"DOC(
 GlobalLpPool consumes an input tensor X and applies lp pool pooling across
 the values in the same channel. This is equivalent to LpPool with kernel size
 equal to the spatial dimension of input tensor.)DOC";

} // namespace

std::string MakeGlobalAveragePoolDoc(int since_version) {
  switch (since_version) {
  case 1:
  case 22:
    return kGlobalAveragePoolDoc;
  default:
    return "";
  }
}

std::string MakeGlobalMaxPoolDoc(int since_version) {
  switch (since_version) {
  case 1:
  case 22:
    return kGlobalMaxPoolDoc;
  default:
    return "";
  }
}

std::string MakeGlobalLpPoolDoc(int since_version) {
  switch (since_version) {
  case 1:
  case 2:
  case 22:
    return kGlobalLpPoolDoc;
  default:
    return "";
  }
}

namespace {

constexpr const char *kDeformConvDoc = R"DOC(
Performs deformable convolution as described in https://arxiv.org/abs/1703.06211 and https://arxiv.org/abs/1811.11168.
This operator specification supports the general N-D case. Note that most common use cases have 2D or 3D data.
)DOC";

} // namespace

std::string MakeDeformConvDoc(int since_version) {
  switch (since_version) {
  case 19:
  case 22:
    return kDeformConvDoc;
  default:
    return "";
  }
}

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

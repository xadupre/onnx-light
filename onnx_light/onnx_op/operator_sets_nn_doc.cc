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

} // namespace nn
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

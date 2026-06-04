// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_generator_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace generator {

std::string MakeBernoulliDoc() {
  return R"DOC(
Draws binary random numbers (0 or 1) from a Bernoulli distribution. The input tensor should be a tensor
containing probabilities p (a value in the range [0,1]) to be used for drawing the binary random number,
where an output of 1 is produced with probability p and an output of 0 is produced with probability (1-p).

This operator is non-deterministic and may not produce the same values in different
implementations (even if a seed is specified).
)DOC";
}

std::string MakeConstantDoc(int since_version) {
  if (since_version == 1 || since_version == 9) {
    return "A constant tensor.";
  }
  if (since_version == 11) {
    return R"DOC(
A constant tensor. Exactly one of the two attributes, either value or sparse_value,
must be specified.
)DOC";
  }
  return R"DOC(
This operator produces a constant tensor. Exactly one of the provided attributes, either value, sparse_value,
or value_* must be specified.
)DOC";
}

std::string MakeConstantOfShapeDoc(int /*since_version*/) {
  return R"DOC(
Generate a tensor with given value and shape.
)DOC";
}

std::string MakeEyeLikeDoc(int /*since_version*/) {
  return R"DOC(
Generate a 2D tensor (matrix) with ones on the diagonal and zeros everywhere else. Only 2D
tensors are supported, i.e. input T1 must be of rank 2. The shape of the output tensor is the
same as the input tensor. The data type can be specified by the 'dtype' argument. If
'dtype' is not specified, then the type of input tensor is used. By default, the main diagonal
is populated with ones, but attribute 'k' can be used to populate upper or lower diagonals.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";
}

std::string MakeRandomNormalDoc() {
  return R"DOC(
Generate a tensor with random values drawn from a normal distribution. The shape
of the tensor is specified by the `shape` argument and the parameter of the normal distribution
specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";
}

std::string MakeMultinomialDoc() {
  return R"DOC(
Generate a tensor of samples from a multinomial distribution according to the probabilities
of each of the possible outcomes.
)DOC";
}

std::string MakeRandomNormalLikeDoc() {
  return R"DOC(
Generate a tensor with random values drawn from a normal distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the normal distribution are specified by `mean` and `scale`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message, and be valid as an output type.
)DOC";
}

std::string MakeRandomUniformDoc() {
  return R"DOC(
Generate a tensor with random values drawn from a uniform distribution. The shape
of the tensor is specified by the `shape` argument and the range by `low` and `high`.

The data type is specified by the 'dtype' argument. The 'dtype' argument must
be one of the data types specified in the 'DataType' enum field in the
TensorProto message.
)DOC";
}

std::string MakeRandomUniformLikeDoc() {
  return R"DOC(
Generate a tensor with random values drawn from a uniform distribution.
The shape of the output tensor is copied from the shape of the input tensor,
and the parameters of the uniform distribution are specified by `low` and `high`.

The data type is specified by the 'dtype' argument, or copied from the input tensor if not provided.
The 'dtype' argument must be one of the data types specified in the 'DataType' enum field in the
TensorProto message and be valid as an output type.
)DOC";
}

std::string MakeRangeDoc() {
  return R"DOC(
Generate a tensor containing a sequence of numbers that begin at `start` and extends by increments of `delta`
up to `limit` (exclusive).

The number of elements in the output of range is computed as below:

```
number_of_elements = max( ceil( (limit - start) / delta ) , 0 )
```

The pseudocode determining the contents of the output is shown below:

```
for(int i=0; i<number_of_elements; ++i) {
  output[i] =  start + (i * delta);
}
```

Example 1

```
Inputs: start = 3, limit = 9, delta = 3
Output: [3, 6]
```

Example 2

```
Inputs: start = 10, limit = 4, delta = -2
Output: [10, 8, 6]
```
)DOC";
}

} // namespace generator
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

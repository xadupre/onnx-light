// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_tensor_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace tensor {

std::string MakeCastDoc(int since_version) {
  if (since_version == 1 || since_version == 6) {
    return R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
NOTE: Casting to and from strings is not supported yet.
)DOC";
  }
  return R"DOC(
The operator casts the elements of a given input tensor to a data type
specified by the 'to' argument and returns an output tensor of the same size in
the converted type. The 'to' argument must be one of the data types specified
in the 'DataType' enum field in the TensorProto message.
)DOC";
}

std::string MakeCastInputTypeConstraintDescription(int since_version) {
  if (since_version == 1 || since_version == 6) {
    return "Constrain input types. Casting from strings and complex are not supported.";
  }
  return "Constrain input types. Casting from complex is not supported.";
}

std::string MakeCastOutputTypeConstraintDescription(int since_version) {
  if (since_version == 1 || since_version == 6) {
    return "Constrain output types. Casting to strings and complex are not supported.";
  }
  return "Constrain output types. Casting to complex is not supported.";
}

std::string MakeCastLikeDoc(int since_version) {
  (void)since_version;
  return R"DOC(
The operator casts the elements of a given input tensor (the first input) to
the same data type as the elements of the second input tensor.
See documentation of the Cast operator for further details.
)DOC";
}

std::string MakeCastLikeInputTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input types. Casting from complex is not supported.";
}

std::string MakeCastLikeOutputTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain output types. Casting to complex is not supported.";
}

std::string MakeAffineGridDoc(int since_version) {
  (void)since_version;
  return R"DOC(
Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
(https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
```
[r00, r01, r02, t0]   [x]   [x']
[r10, r11, r12, t1] * [y] = [y']
[r20, r21, r22, t2]   [z]   [z']
[0,   0,   0,   1 ]   [1]   [1 ]
```
where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.

Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
The output `grid` contains positions in the output space.

When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
```
v            v            v            v
|-------------------|------------------|
-1                  0                  1
```
When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
```
    v        v         v         v
|------------------|-------------------|
-1                 0                   1
```
)DOC";
}

std::string MakeAffineGridGridTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain grid types to float tensors.";
}

std::string MakeAffineGridSizeTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain size's type to int64 tensors.";
}

std::string MakeConcatDoc(int since_version) {
  if (since_version == 1) {
    return R"DOC(Concatenate a list of tensors into a single tensor)DOC";
  }
  if (since_version == 4) {
    return R"DOC(Concatenate a list of tensors into a single tensor)DOC";
  }
  return R"DOC(Concatenate a list of tensors into a single tensor. All input tensors must have the same shape, except for the dimension size of the axis to concatenate on.)DOC";
}

std::string MakeConcatTypeConstraintDescription(int since_version) {
  if (since_version == 1) {
    return "Constrain output types to float tensors.";
  }
  return "Constrain output types to any tensor type.";
}

std::string MakeExpandDoc(int since_version) {
  (void)since_version;
  return R"DOC(
Broadcast the input tensor following the given shape and the broadcast rule.
The broadcast rule is similar to numpy.array(input) * numpy.ones(shape):
Dimensions are right alignment;
Two corresponding dimensions must have the same value, or one of them is equal to 1.
Also, this operator is similar to numpy.broadcast_to(input, shape),
but the major difference is numpy.broadcast_to() does not allow shape to be smaller than input.size().
It is possible that the output.shape is not equal to shape, when some dimensions in shape is equal to 1,
or the shape.ndim < input.shape.ndim.
)DOC";
}

std::string MakeExpandTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensors.";
}

std::string MakeTransposeDoc(int since_version) {
  (void)since_version;
  return R"DOC(
Returns a transpose of the input tensor. (Similar to `numpy.transpose`).
The optional attribute `perm` must be a permutation of the dimensions of
the input tensor. Axis `i` of the output tensor corresponds to the axis
`perm[i]` of the input tensor.
For example, when perm=(1, 0, 2), given an input tensor of shape (1, 2, 3),
the output shape will be (2, 1, 3).
When perm=(1, 2, 0), given an input tensor of shape (1, 2, 3),
the output shape will be (2, 3, 1).
If the attribute `perm` is omitted, its default value is `(n-1, ..., 0)`,
where `n` is the rank of the input tensor.
)DOC";
}

std::string MakeTransposeTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensor types.";
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

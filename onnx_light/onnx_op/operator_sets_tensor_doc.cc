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

std::string MakeBitCastDoc() {
  return R"DOC(
Reinterprets the binary representation of a tensor as a different data type,
specified by the 'to' attribute. Unlike Cast, BitCast preserves the exact bit
pattern without any value conversion.

The target data type must have the same bit-width as the input data type.
The output tensor has the same shape as the input tensor.
All types except string are supported. Implementations must treat the
underlying bytes as little endian.
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

std::string MakeGridSampleDoc(int since_version) {
  if (since_version <= 16) {
    return R"DOC(
Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from `grid`.
Currently, only spatial (4-D) inputs are supported. For input `X` with shape (N, C, H, W) and `grid` with shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out).

The tensor `X` contains values at centers of square pixels in a H by W 2-dimensional image.
The tensor `grid` describes normalized positions where the output `Y` is to be computed
using a specified interpolation method (the mode) and a padding mode (for grid positions falling outside the 2-dimensional image).

Elements in `grid[N, H_out, W_out]` are size-2 vectors specifying positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[N, C, H_out, W_out]`.

The GridSample operator is often used in doing grid generator and sampler in the [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/master/generated/torch.nn.functional.grid_sample.html#torch-nn-functional-grid-sample).
)DOC";
  }
  return R"DOC(
Given an input `X` and a flow-field `grid`, computes the output `Y` using `X` values and pixel locations from the `grid`.
For spatial input `X` with shape (N, C, H, W), the `grid` will have shape (N, H_out, W_out, 2),
the output `Y` will have shape (N, C, H_out, W_out). For volumetric input `X` with shape (N, C, D, H, W),
the `grid` will have shape (N, D_out, H_out, W_out, 3), the output `Y` will have shape (N, C, D_out, H_out, W_out).
More generally, for an input `X` of rank r+2 with shape (N, C, d1, d2, ..., dr),
the `grid` will have shape (N, D1_out, D2_out, ..., Dr_out, r), the output `Y` will have shape (N, C, D1_out, D2_out, ..., Dr_out).

The tensor `X` contains values at centers of square pixels (voxels, etc) locations such as (n, c, d1_in, d2_in, ..., dr_in).
The (n, d1_out, d2_out, ..., dr_out, :) values from the tensor `grid` are the normalized positions for interpolating the values
at the (n, c, d1_out, d2_out, ..., dr_out) locations from the output tensor `Y` using a specified interpolation method (the mode)
and a padding mode (for `grid` positions falling outside the 2-dimensional image).

For example, the values in `grid[n, h_out, w_out, :]` are size-2 vectors specifying normalized positions in the 2-dimensional space of `X`.
They are used to interpolate output values of `Y[n, c, h_out, w_out]`.

The GridSample operator is often used in doing grid generator and sampler in the
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025).
See also in [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html).
)DOC";
}

std::string MakeGridSampleInputTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input `X` and output `Y` types to all tensor types.";
}

std::string MakeGridSampleGridTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain grid types to float tensors.";
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

std::string MakeSqueezeDoc(int since_version) {
  if (since_version <= 11) {
    return R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes a  parameter `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
)DOC";
  }
  return R"DOC(
Remove single-dimensional entries from the shape of a tensor.
Takes an input `axes` with a list of axes to squeeze.
If `axes` is not provided, all the single dimensions will be removed from
the shape. If an axis is selected with shape entry not equal to one, an error is raised.
)DOC";
}

std::string MakeSqueezeTypeConstraintDescription(int since_version) {
  if (since_version >= 25) {
    return "Constrain input and output types to all tensor types up to IRv13.";
  }
  if (since_version >= 24) {
    return "Constrain input and output types to all tensor types up to IRv12.";
  }
  if (since_version >= 23) {
    return "Constrain input and output types to all tensor types up to IRv11.";
  }
  if (since_version >= 21) {
    return "Constrain input and output types to all tensor types up to IRv10.";
  }
  return "Constrain input and output types to all tensor types.";
}

std::string MakeUnsqueezeDoc(int since_version) {
  if (since_version <= 11) {
    return R"DOC(
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required argument `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing the same data as `data` but with shape [1, 3, 4, 5, 1].

The attribute `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.

)DOC";
  }
  return R"DOC(
Insert single-dimensional entries to the shape of an input tensor (`data`).
Takes one required input `axes` - which contains a list of dimension indices and this operator will insert a dimension of value `1` into the corresponding index of the output tensor (`expanded`).

For example:
  Given an input tensor (`data`) of shape [3, 4, 5], then
  Unsqueeze(data, axes=[0, 4]) outputs a tensor (`expanded`) containing the same data as `data` but with shape [1, 3, 4, 5, 1].

The `axes` should not contain any duplicate entries. It is an error if it contains duplicates.
The rank of the output tensor (`output_rank`) is the rank of the input tensor (`data`) plus the number of values in `axes`.
Each value in `axes` should be within the (inclusive) range [-output_rank , output_rank - 1].
The order of values in `axes` does not matter and can come in any order.

)DOC";
}

std::string MakeUnsqueezeTypeConstraintDescription(int since_version) {
  if (since_version >= 25) {
    return "Constrain input and output types to all tensor types up to IRv13.";
  }
  if (since_version >= 24) {
    return "Constrain input and output types to all tensor types up to IRv12.";
  }
  if (since_version >= 23) {
    return "Constrain input and output types to all tensor types up to IRv11.";
  }
  if (since_version >= 21) {
    return "Constrain input and output types to all tensor types up to IRv10.";
  }
  return "Constrain input and output types to all tensor types.";
}

std::string MakeNonZeroDoc(int since_version) {
  (void)since_version;
  return R"DOC(
    Returns the indices of the elements that are non-zero
    (in row-major order - by dimension).
    NonZero behaves similar to numpy.nonzero:
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.nonzero.html,
    but for scalar input, NonZero produces output shape (0, N) instead of (1, N), which is different from Numpy's behavior.
)DOC";
}

std::string MakeNonZeroTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain to all tensor types.";
}

std::string MakeShapeDoc(int since_version) {
  if (since_version <= 13) {
    return R"DOC(
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
)DOC";
  }
  return R"DOC(
Takes a tensor as input and outputs an 1D int64 tensor containing the shape of the input tensor.
Optional attributes start and end can be used to compute a slice of the input tensor's shape.
If start axis is omitted, the slice starts from axis 0.
The end axis, if specified, is exclusive (and the returned value will not include the size of that axis).
If the end axis is omitted, the axes upto the last one will be included.
Negative axes indicate counting back from the last axis.
Note that axes will be clamped to the range [0, r], where r is the
rank of the input tensor if they are out-of-range (after adding r in the case of
negative axis). Thus, specifying any end value > r is equivalent to specifying an end
value of r, and specifying any start value < -r is equivalent to specifying a start
value of 0. If start > end, the result will be an empty shape.

Examples:

```
Input tensor with shape: [2, 3, 4]
No attributes specified.
Output: [2, 3, 4]
```

```
Input tensor with shape: [2, 3, 4]
start: -1
Output: [4]
```

```
Input tensor with shape: [2, 3, 4]
end: -1
Output: [2, 3]
```

```
Input tensor with shape: [2, 3, 4]
start: 1
end: 2
Output: [3]
```
)DOC";
}

std::string MakeShapeTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Input tensor can be of arbitrary type.";
}

std::string MakeTileDoc(int since_version) {
  (void)since_version;
  return R"DOC(Constructs a tensor by tiling a given tensor.
This is the same as function `tile` in Numpy, but no broadcast.
For example A = [[1, 2], [3, 4]], B = [1, 2], tile(A, B) = [[1, 2, 1, 2], [3, 4, 3, 4]]
)DOC";
}

std::string MakeTileTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensor types.";
}

std::string MakeUpsampleDoc(int since_version) {
  if (since_version == 1) {
    return R"DOC(
Upsample the input tensor.
The width and height of the output tensor are:
  output_width = floor(input_width * width_scale),
  output_height = floor(input_height * height_scale).
Example:
  Given `data` tensor, width_scale, height_scale, mode,
  Upsample the input 4-D tensor in nearest mode:
  data = [[[
      [1, 2],
      [3, 4]
  ]]]
  width_scale = 2
  height_scale = 2
  mode = "nearest"
  output = [[[
      [1, 1, 2, 2],
      [1, 1, 2, 2],
      [3, 3, 4, 4],
      [3, 3, 4, 4]
  ]]]
)DOC";
  }
  return R"DOC(
Upsample the input tensor.
Each dimension value of the output tensor is:
  output_dimension = floor(input_dimension * scale).
)DOC";
}

std::string MakeUpsampleTypeConstraintDescription(int since_version) {
  if (since_version == 1) {
    return "Constrain output types to bool, int32, int64, float16, float, double tensors.";
  }
  if (since_version == 7) {
    return "Constrain input and output types to all tensor types.";
  }
  return "Constrain input 'X' and output 'Y' to all tensor types.";
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

std::string MakeDepthToSpaceDoc(int since_version) {
  if (since_version <= 1) {
    return R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions.
)DOC";
  }
  if (since_version <= 11) {
    return R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])

tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])

y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])


In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

b, c, h, w = x.shape

tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])

tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])

y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])

)DOC";
  }
  return R"DOC(DepthToSpace rearranges (permutes) data from depth into blocks of spatial data.
This is the reverse transformation of SpaceToDepth. More specifically, this op outputs a copy of
the input tensor where values from the depth dimension are moved in spatial blocks to the height
and width dimensions. By default, `mode` = `DCR`.
In the DCR mode, elements along the depth dimension from the input tensor are rearranged in the
following order: depth, column, and then row. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, blocksize, blocksize, c // (blocksize**2), h, w])
tmp = np.transpose(tmp, [0, 3, 4, 1, 5, 2])
y = np.reshape(tmp, [b, c // (blocksize**2), h * blocksize, w * blocksize])
```

In the CRD mode, elements along the depth dimension from the input tensor are rearranged in the
following order: column, row, and the depth. The output y is computed from the input x as below:

```
b, c, h, w = x.shape
tmp = np.reshape(x, [b, c // (blocksize ** 2), blocksize, blocksize, h, w])
tmp = np.transpose(tmp, [0, 1, 4, 2, 5, 3])
y = np.reshape(tmp, [b, c // (blocksize ** 2), h * blocksize, w * blocksize])
```
)DOC";
}

std::string MakeDepthToSpaceTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensor types.";
}

std::string MakeSpaceToDepthDoc(int since_version) {
  (void)since_version;
  return R"DOC(SpaceToDepth rearranges blocks of spatial data into depth. More specifically,
this op outputs a copy of the input tensor where values from the height and width dimensions
are moved to the depth dimension.
)DOC";
}

std::string MakeSpaceToDepthTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensor types.";
}

std::string MakeGatherDoc(int since_version) {
  if (since_version <= 1) {
    return R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).
Example 1:
```
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
```
Example 2:
```
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [[1.0, 1.9]],
      [[2.3, 3.9]],
      [[4.5, 5.9]],
  ]
```
)DOC";
  }
  if (since_version <= 11) {
    return R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

axis = 0 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]

```
  data = [
      [1.0, 1.2],
      [2.3, 3.4],
      [4.5, 5.7],
  ]
  indices = [
      [0, 1],
      [1, 2],
  ]
  output = [
      [
          [1.0, 1.2],
          [2.3, 3.4],
      ],
      [
          [2.3, 3.4],
          [4.5, 5.7],
      ],
  ]
```
axis = 1 :

Let
k = indices[i_{0}, ..., i_{q-1}]
Then
output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]

```
  data = [
      [1.0, 1.2, 1.9],
      [2.3, 3.4, 3.9],
      [4.5, 5.7, 5.9],
  ]
  indices = [
      [0, 2],
  ]
  axis = 1,
  output = [
      [[1.0, 1.9]],
      [[2.3, 3.9]],
      [[4.5, 5.9]],
  ]
```
)DOC";
  }
  return R"DOC(
Given `data` tensor of rank r >= 1, and `indices` tensor of rank q, gather
entries of the axis dimension of `data` (by default outer-most one as axis=0) indexed by `indices`, and concatenates
them in an output tensor of rank q + (r - 1).

It is an indexing operation that indexes into the input `data` along a single (specified) axis.
Each entry in `indices` produces a `r-1` dimensional slice of the input tensor.
The entire operation produces, conceptually, a `q`-dimensional tensor of `r-1` dimensional slices,
which is arranged into a `q + (r-1)`-dimensional tensor, with the `q` dimensions taking the
place of the original `axis` that is being indexed into.

The following few examples illustrate how `Gather` works for specific shapes of `data`,
`indices`, and given value of `axis`:
| data shape | indices shape | axis | output shape | output equation |
| --- | --- | --- | --- | --- |
| (P, Q) | ( )  (a scalar)   | 0 | (Q)       | output[q] = data[indices, q] |
| (P, Q, R) | ( )  (a scalar)   | 1 | (P, R)       | output[p, r] = data[p, indices, r] |
| (P, Q) | (R, S) | 0 | (R, S, Q) | output[r, s, q] = data[ [indices[r, s], q] |
| (P, Q) | (R, S) | 1 | (P, R, S) | output[p, r, s] = data[ p, indices[r, s]] |

More generally, if `axis = 0`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[i_{0}, ..., i_{q-1}, j_{0}, ..., j_{r-2}] = input[k , j_{0}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2],
    [2.3, 3.4],
    [4.5, 5.7],
]
indices = [
    [0, 1],
    [1, 2],
]
output = [
    [
        [1.0, 1.2],
        [2.3, 3.4],
    ],
    [
        [2.3, 3.4],
        [4.5, 5.7],
    ],
]
```

If `axis = 1`, let `k = indices[i_{0}, ..., i_{q-1}]`
then `output[j_{0}, i_{0}, ..., i_{q-1}, j_{1}, ..., j_{r-2}] = input[j_{0}, k, j_{1}, ..., j_{r-2}]`:

```
data = [
    [1.0, 1.2, 1.9],
    [2.3, 3.4, 3.9],
    [4.5, 5.7, 5.9],
]
indices = [
    [0, 2],
]
axis = 1,
output = [
        [[1.0, 1.9]],
        [[2.3, 3.9]],
        [[4.5, 5.9]],
]
```
)DOC";
}

std::string MakeGatherElementsDoc(int since_version) {
  if (since_version <= 11) {
    return R"DOC(

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
  out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
  out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
  out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
  data = [
      [1, 2],
      [3, 4],
  ]
  indices = [
      [0, 0],
      [1, 0],
  ]
  axis = 1
  output = [
      [
        [1, 1],
        [4, 3],
      ],
  ]
```
Example 2:
```
  data = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
  ]
  indices = [
      [1, 2, 0],
      [2, 0, 0],
  ]
  axis = 0
  output = [
      [
        [4, 8, 3],
        [7, 2, 3],
      ],
  ]
```
)DOC";
  }
  return R"DOC(

GatherElements takes two inputs `data` and `indices` of the same rank r >= 1
and an optional attribute `axis` that identifies an axis of `data`
(by default, the outer-most axis, that is axis 0). It is an indexing operation
that produces its output by indexing into the input data tensor at index
positions determined by elements of the `indices` tensor.
Its output shape is the same as the shape of `indices` and consists of one value
(gathered from the `data`) for each element in `indices`.

For instance, in the 3-D case (r = 3), the output produced is determined
by the following equations:
```
out[i][j][k] = input[index[i][j][k]][j][k] if axis = 0,
out[i][j][k] = input[i][index[i][j][k]][k] if axis = 1,
out[i][j][k] = input[i][j][index[i][j][k]] if axis = 2,
```

This operator is also the inverse of ScatterElements. It is similar to Torch's gather operation.

Example 1:
```
data = [
    [1, 2],
    [3, 4],
]
indices = [
    [0, 0],
    [1, 0],
]
axis = 1
output = [
    [1, 1],
    [4, 3],
]
```
Example 2:
```
data = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]
indices = [
    [1, 2, 0],
    [2, 0, 0],
]
axis = 0
output = [
    [4, 8, 3],
    [7, 2, 3],
]
```
)DOC";
}

std::string MakeGatherNDDoc(int since_version) {
  if (since_version <= 11) {
    return R"DOC(
Given `data` tensor of rank `r` >= 1, and `indices` tensor of rank `q` >= 1, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r` (inclusive)

3) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r` => error condition

2) If `indices_shape[-1] == r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `r`. Let us think of each such `r` ranked tensor as `indices_slice`.
   Each *scalar value* corresponding to `data[indices_slice]` is filled into the corresponding location of the `(q-1)`-dimensional tensor
   to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r`, since the rank of `indices` is `q`, `indices` can be thought of as a `(q-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r`. Let us think of each such tensors as `indices_slice`.
   Each *tensor slice* corresponding to `data[indices_slice , :]` is filled into the corresponding location of the `(q-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, and 4 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

)DOC";
  }
  if (since_version <= 12) {
    return R"DOC(
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

`Example 1`

  batch_dims = 0

  data    = [[0,1],[2,3]]   # data_shape = [2, 2]

  indices = [[0,0],[1,1]]   # indices_shape = [2, 2]

  output  = [0,3]           # output_shape = [2]

`Example 2`

  batch_dims = 0

  data    = [[0,1],[2,3]]  # data_shape = [2, 2]

  indices = [[1],[0]]      # indices_shape = [2, 1]

  output  = [[2,3],[0,1]]  # output_shape = [2, 2]

`Example 3`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]

  output  = [[2,3],[4,5]]                 # output_shape = [2, 2]

`Example 4`

  batch_dims = 0

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]

  output  = [[[2,3]],[[4,5]]]             # output_shape = [2, 1, 2]

`Example 5`

  batch_dims = 1

  data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape = [2, 2, 2]

  indices = [[1],[0]]             # indices_shape = [2, 1]

  output  = [[2,3],[4,5]]             # output_shape = [2, 2]


)DOC";
  }
  return R"DOC(
Given `data` tensor of rank `r` >= 1, `indices` tensor of rank `q` >= 1, and `batch_dims` integer `b`, this operator gathers
slices of `data` into an output tensor of rank `q + r - indices_shape[-1] - 1 - b`.

`indices` is an q-dimensional integer tensor, best thought of as a `(q-1)`-dimensional tensor of index-tuples into `data`,
where each element defines a slice of `data`

`batch_dims` (denoted as `b`) is an integer indicating the number of batch dimensions, i.e the leading `b` number of dimensions of
`data` tensor and `indices` are representing the batches, and the gather starts from the `b+1` dimension.

Some salient points about the inputs' rank and shape:

1) r >= 1 and q >= 1 are to be honored. There is no dependency condition to be met between ranks `r` and `q`

2) The first `b` dimensions of the shape of `indices` tensor and `data` tensor must be equal.

3) b < min(q, r) is to be honored.

4) The `indices_shape[-1]` should have a value between 1 (inclusive) and rank `r-b` (inclusive)

5) All values in `indices` are expected to be within bounds [-s, s-1] along axis of size `s` (i.e.) `-data_shape[i] <= indices[...,i] <= data_shape[i] - 1`.
   It is an error if any of the index values are out of bounds.

The output is computed as follows:

The output tensor is obtained by mapping each index-tuple in the `indices` tensor to the corresponding slice of the input `data`.

1) If `indices_shape[-1] > r-b` => error condition

2) If `indices_shape[-1] == r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensors
   containing 1-D tensors of dimension `r-b`, where `N` is an integer equals to the product of 1 and all the elements in the batch dimensions
   of the indices_shape. Let us think of each such `r-b` ranked tensor as `indices_slice`. Each *scalar value* corresponding to `data[0:b-1,indices_slice]`
   is filled into the corresponding location of the `(q-b-1)`-dimensional tensor to form the `output` tensor (Example 1 below)

3) If `indices_shape[-1] < r-b`, since the rank of `indices` is `q`, `indices` can be thought of as `N` `(q-b-1)`-dimensional tensor
   containing 1-D tensors of dimension `< r-b`. Let us think of each such tensors as `indices_slice`. Each *tensor slice* corresponding
   to `data[0:b-1, indices_slice , :]` is filled into the corresponding location of the `(q-b-1)`-dimensional tensor
   to form the `output` tensor (Examples 2, 3, 4 and 5 below)

This operator is the inverse of `ScatterND`.

**Example 1**

```
batch_dims = 0
data    = [[0,1],[2,3]]   # data_shape    = [2, 2]
indices = [[0,0],[1,1]]   # indices_shape = [2, 2]
output  = [0,3]           # output_shape  = [2]
```

**Example 2**

```
batch_dims = 0
data    = [[0,1],[2,3]]  # data_shape    = [2, 2]
indices = [[1],[0]]      # indices_shape = [2, 1]
output  = [[2,3],[0,1]]  # output_shape  = [2, 2]
```

**Example 3**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[0,1],[1,0]]                 # indices_shape = [2, 2]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```

**Example 4**

```
batch_dims = 0
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[[0,1]],[[1,0]]]             # indices_shape = [2, 1, 2]
output  = [[[2,3]],[[4,5]]]             # output_shape  = [2, 1, 2]
```

**Example 5**

```
batch_dims = 1
data    = [[[0,1],[2,3]],[[4,5],[6,7]]] # data_shape    = [2, 2, 2]
indices = [[1],[0]]                     # indices_shape = [2, 1]
output  = [[2,3],[4,5]]                 # output_shape  = [2, 2]
```
)DOC";
}

std::string MakeTriluDoc(int since_version) {
  (void)since_version;
  return R"DOC(
Given a 2-D matrix or batches of 2-D matrices, returns the upper or lower triangular part of the tensor(s).
The attribute "upper" determines whether the upper or lower part is retained. If set to true,
the upper triangular matrix is retained. Lower triangular matrix is retained otherwise.
Default value for the "upper" attribute is true.
Trilu takes one input tensor of shape [*, N, M], where * is zero or more batch dimensions. The upper triangular part consists
of the elements on and above the given diagonal (k). The lower triangular part consists of elements on and below the diagonal.
All other elements in the matrix are set to zero.
If k = 0, the triangular part on and above/below the main diagonal is retained.
If upper is set to true, a positive k retains the upper triangular matrix excluding the main diagonal and (k-1) diagonals above it.
A negative k value retains the main diagonal and |k| diagonals below it.
If upper is set to false, a positive k retains the lower triangular matrix including the main diagonal and k diagonals above it.
A negative k value excludes the main diagonal and (|k|-1) diagonals below it.
)DOC";
}

std::string MakeTriluTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensor types.";
}

std::string MakeReverseSequenceDoc(int since_version) {
  (void)since_version;
  return R"DOC(
Reverse batch of sequences having different lengths specified by `sequence_lens`.

For each slice i iterating on batch axis, the operator reverses the first sequence_lens[i] elements on time axis,
and copies elements whose index's beyond sequence_lens[i] to the output. So the output slice i contains reversed
sequences on the first sequence_lens[i] elements, then have original values copied for the other elements.

Example 1:
  input = [[0.0, 4.0, 8.0,  12.0],
           [1.0, 5.0, 9.0,  13.0],
           [2.0, 6.0, 10.0, 14.0],
           [3.0, 7.0, 11.0, 15.0]]
  sequence_lens = [4, 3, 2, 1]
  time_axis = 0
  batch_axis = 1

  output = [[3.0, 6.0, 9.0,  12.0],
            [2.0, 5.0, 8.0,  13.0],
            [1.0, 4.0, 10.0, 14.0],
            [0.0, 7.0, 11.0, 15.0]]

Example 2:
  input = [[0.0,  1.0,  2.0,  3.0 ],
           [4.0,  5.0,  6.0,  7.0 ],
           [8.0,  9.0,  10.0, 11.0],
           [12.0, 13.0, 14.0, 15.0]]
  sequence_lens = [1, 2, 3, 4]
  time_axis = 1
  batch_axis = 0

  output = [[0.0,  1.0,  2.0,  3.0 ],
            [5.0,  4.0,  6.0,  7.0 ],
            [10.0, 9.0,  8.0,  11.0],
            [15.0, 14.0, 13.0, 12.0]]
)DOC";
}

std::string MakeReverseSequenceTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Input and output types can be of any tensor type.";
}

std::string MakeCompressDoc(int since_version) {
  if (since_version <= 9) {
    return R"DOC(
Selects slices from an input tensor along a given axis value passed.

In case axis is not provided, input is flattened before elements being selected.

Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
)DOC";
  }
  return R"DOC(
Selects slices from an input tensor along a given axis value passed.

In case axis is not provided, input is flattened before elements being selected.
Positive value means counting dimensions from the front.
Negative value means counting dimensions from the back.
Compress behaves like numpy.compress: https://docs.scipy.org/doc/numpy/reference/generated/numpy.compress.html
)DOC";
}

std::string MakeCompressTypeConstraintDescription(int since_version) {
  (void)since_version;
  return "Constrain input and output types to all tensor types.";
}

std::string MakeSplitDoc(int since_version) {
  if (since_version == 1) {
    return R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. The lengths of the split can be specified using argument 'axis' or
optional second input blob to the operator. Otherwise, the tensor is split
to equal sized parts.
)DOC";
  }
  if (since_version == 2 || since_version == 11) {
    return R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using argument 'split'.
Otherwise, the tensor is split to equal sized parts.
)DOC";
  }
  if (since_version == 13) {
    return R"DOC(Split a tensor into a list of tensors, along the specified
'axis'. Lengths of the parts can be specified using input 'split'.
Otherwise, the tensor is split to equal sized parts.
)DOC";
  }
  return R"DOC(Split a tensor into a list of tensors, along the specified 'axis'.
Either input 'split' or the attribute 'num_outputs' should be specified, but not both.
If the attribute 'num_outputs' is specified, then the tensor is split into equal sized parts.
If the tensor is not evenly splittable into `num_outputs`, the last chunk will be smaller.
If the input 'split' is specified, it indicates the sizes of each output in the split.
)DOC";
}

std::string MakeSplitTypeConstraintDescription(int since_version) {
  if (since_version == 1) {
    return "Constrain input types to float tensors.";
  }
  return "Constrain input and output types to all tensor types.";
}

} // namespace tensor
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

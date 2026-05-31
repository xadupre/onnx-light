// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_traditionalml_doc.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_op {
namespace traditionalml {

std::string MakeArrayFeatureExtractorDoc() {
  return R"DOC(
Select elements of the input tensor based on the indices passed.<br>
The indices are applied to the last axes of the tensor.
)DOC";
}

std::string MakeBinarizerDoc() {
  return R"DOC(
Maps the values of the input tensor to either 0 or 1, element-wise, based on the outcome of a comparison against a threshold value.
)DOC";
}

std::string MakeLabelEncoderDoc() {
  return R"DOC(
Maps each element in the input tensor to another value.<br>
The mapping is determined by the two parallel attributes, 'keys_*' and
'values_*' attribute. The i-th value in the specified 'keys_*' attribute
would be mapped to the i-th value in the specified 'values_*' attribute. It
implies that input's element type and the element type of the specified
'keys_*' should be identical while the output type is identical to the
specified 'values_*' attribute. Note that the 'keys_*' and 'values_*' attributes
must have the same length. If an input element can not be found in the
specified 'keys_*' attribute, the 'default_*' that matches the specified
'values_*' attribute may be used as its output value. The type of the 'default_*'
attribute must match the 'values_*' attribute chosen. <br>
Let's consider an example which maps a string tensor to an integer tensor.
Assume and 'keys_strings' is ["Amy", "Sally"], 'values_int64s' is [5, 6],
and 'default_int64' is '-1'.  The input ["Dori", "Amy", "Amy", "Sally",
"Sally"] would be mapped to [-1, 5, 5, 6, 6].<br>
Since this operator is an one-to-one mapping, its input and output shapes
are the same. Notice that only one of 'keys_*'/'values_*' can be set.<br>
Float keys with value 'NaN' match any input 'NaN' value regardless of bit
value. If a key is repeated, the last key takes precedence.
)DOC";
}

std::string MakeOneHotEncoderDoc() {
  return R"DOC(
    Replace each input element with an array of ones and zeros, where a single
    one is placed at the index of the category that was passed in. The total category count
    will determine the size of the extra dimension of the output array Y.<br>
    For example, if we pass a tensor with a single value of 4, and a category count of 8,
    the output will be a tensor with ``[0,0,0,0,1,0,0,0]``.<br>
    This operator assumes every input feature is from the same set of categories.<br>
    If the input is a tensor of float, int32, or double, the data will be cast
    to integers and the cats_int64s category list will be used for the lookups.
)DOC";
}

std::string MakeLinearClassifierDoc() {
  return R"DOC(
    Linear classifier
)DOC";
}

std::string MakeLinearRegressorDoc() {
  return R"DOC(
    Generalized linear regression evaluation.<br>
    If targets is set to 1 (default) then univariate regression is performed.<br>
    If targets is set to M then M sets of coefficients must be passed in as a sequence
    and M results will be output for each input n in N.<br>
    The coefficients array is of length n, and the coefficients for each target are contiguous.
    Intercepts are optional but if provided must match the number of targets.
)DOC";
}

std::string MakeScalerDoc() {
  return R"DOC(
    Rescale input data, for example to standardize features by removing the mean and scaling to unit variance.
)DOC";
}

std::string MakeSVMClassifierDoc() {
  return R"DOC(
    Support Vector Machine classifier
)DOC";
}

std::string MakeSVMRegressorDoc() {
  return R"DOC(
    Support Vector Machine regression prediction and one-class SVM anomaly detection.
)DOC";
}

std::string MakeZipMapDoc() {
  return R"DOC(
Creates a map from the input and the attributes.<br>
The values are provided by the input tensor, while the keys are specified by the attributes.
Must provide keys in either classlabels_strings or classlabels_int64s (but not both).<br>
The columns of the tensor correspond one-by-one to the keys specified by the attributes. There must be as many columns as keys.<br>
)DOC";
}

std::string MakeTreeEnsembleClassifierDoc(int since_version) {
  switch (since_version) {
  case 1:
    return R"DOC(
    Tree Ensemble classifier.  Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with ``class_`` are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
)DOC";
  case 3:
    return R"DOC(
    Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with ``class_`` are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
)DOC";
  case 5:
    return R"DOC(
    This operator is DEPRECATED. Please use TreeEnsemble with provides similar functionality.
    In order to determine the top class, the ArgMax node can be applied to the output of TreeEnsemble.
    To encode class labels, use a LabelEncoder operator.
    Tree Ensemble classifier. Returns the top class for each of N inputs.<br>
    The attributes named 'nodes_X' form a sequence of tuples, associated by
    index into the sequences, which must all be of equal length. These tuples
    define the nodes.<br>
    Similarly, all fields prefixed with ``class_`` are tuples of votes at the leaves.
    A leaf may have multiple votes, where each vote is weighted by
    the associated class_weights index.<br>
    One and only one of classlabels_strings or classlabels_int64s
    will be defined. The class_ids are indices into this list.
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
)DOC";
  default:
    return std::string();
  }
}

std::string MakeTreeEnsembleRegressorDoc(int since_version) {
  switch (since_version) {
  case 1:
    return R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";
  case 3:
    return R"DOC(
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";
  case 5:
    return R"DOC(
    This operator is DEPRECATED. Please use TreeEnsemble instead which provides the same
    functionality.<br>
    Tree Ensemble regressor.  Returns the regressed values for each input in N.<br>
    All args with nodes_ are fields of a tuple of tree nodes, and
    it is assumed they are the same length, and an index i will decode the
    tuple across these inputs.  Each node id can appear only once
    for each tree id.<br>
    All fields prefixed with target_ are tuples of votes at the leaves.<br>
    A leaf may have multiple votes, where each vote is weighted by
    the associated target_weights index.<br>
    All fields ending with <i>_as_tensor</i> can be used instead of the
    same parameter without the suffix if the element type is double and not float.
    All trees must have their node ids start at 0 and increment by 1.<br>
    Mode enum is BRANCH_LEQ, BRANCH_LT, BRANCH_GTE, BRANCH_GT, BRANCH_EQ, BRANCH_NEQ, LEAF
)DOC";
  default:
    return std::string();
  }
}

std::string MakeTreeEnsembleDoc() {
  return R"DOC(
    Tree Ensemble operator.  Returns the regressed values for each input in a batch.
    Inputs have dimensions `[N, F]` where `N` is the input batch size and `F` is the number of input features.
    Outputs have dimensions `[N, num_targets]` where `N` is the batch size and `num_targets` is the number of targets, which is a configurable attribute.

    The encoding of this attribute is split along interior nodes and the leaves of the trees. Notably, attributes with the prefix `nodes_*` are associated with interior nodes, and attributes with the prefix `leaf_*` are associated with leaves.
    The attributes `nodes_*` must all have the same length and encode a sequence of tuples, as defined by taking all the `nodes_*` fields at a given position.

    All fields prefixed with `leaf_*` represent tree leaves, and similarly define tuples of leaves and must have identical length.

    This operator can be used to implement both the previous `TreeEnsembleRegressor` and `TreeEnsembleClassifier` nodes.
    The `TreeEnsembleRegressor` node maps directly to this node and requires changing how the nodes are represented.
    The `TreeEnsembleClassifier` node can be implemented by adding a `ArgMax` node after this node to determine the top class.
    To encode class labels, a `LabelEncoder` or `GatherND` operator may be used.
)DOC";
}

} // namespace traditionalml
} // namespace onnx_op
} // namespace ONNX_LIGHT_NAMESPACE

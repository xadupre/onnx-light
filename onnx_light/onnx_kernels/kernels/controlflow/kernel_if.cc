// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_kernels/kernels/controlflow/include_controlflow_kernels.h"

#include "onnx_kernels/run_nodes.h"
#include "onnx_kernels/runtime_context.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_kernels {
namespace kernel {

Tensor If::operator()(const Tensor &cond, const Tensor &then_value,
                      const Tensor &else_value) const {
  // Allocate an output matching either branch's type/shape (both must agree;
  // the in-place overload enforces this); the in-place overload writes into
  // ``out.data`` below.
  Tensor out("", then_value.data_type, then_value.shape,
             std::vector<uint8_t>(then_value.size_bytes()));
  (*this)(cond, then_value, else_value, out);
  return out;
}

void If::operator()(const Tensor &cond, const Tensor &then_value, const Tensor &else_value,
                    Tensor &output) const {
  EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL,
                      "kernel::If: 'cond' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "kernel::If: 'cond' must contain a single element.");
  EXT_ENFORCE_INVALID(then_value.data_type == else_value.data_type,
                      "kernel::If: 'then_value' and 'else_value' must have the same data type.");
  EXT_ENFORCE_INVALID(then_value.shape == else_value.shape,
                      "kernel::If: 'then_value' and 'else_value' must have the same shape.");
  EXT_ENFORCE_INVALID(
      output.data_type == then_value.data_type,
      "kernel::If preallocated output must have the same data type as the branches.");
  EXT_ENFORCE_INVALID(output.shape == then_value.shape,
                      "kernel::If preallocated output shape must match the branch shape.");
  EXT_ENFORCE_INVALID(output.data.size() == then_value.size_bytes(),
                      "kernel::If preallocated output buffer has unexpected size in bytes.");

  const bool taken = cond.bytes()[0] != 0;
  const Tensor &src = taken ? then_value : else_value;
  if (src.size_bytes() > 0) {
    std::memcpy(output.data.data(), src.bytes(), src.size_bytes());
  }
}

std::vector<Tensor> If::operator()(const Tensor &cond, const GraphProto &then_branch,
                                   const GraphProto &else_branch, RuntimeContext &rt) const {
  EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL,
                      "kernel::If: 'cond' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "kernel::If: 'cond' must contain a single element.");
  EXT_ENFORCE_INVALID(then_branch.output_size() == else_branch.output_size(),
                      "kernel::If: 'then_branch' and 'else_branch' must declare the same number "
                      "of outputs.");

  const bool taken = cond.bytes()[0] != 0;
  const GraphProto &branch = taken ? then_branch : else_branch;

  // Run the selected subgraph in a fresh child context whose tensor map and
  // model-local function registry are inherited from the caller's context
  // so the subgraph can read outer-scope values, while writes produced by
  // the subgraph remain local and do not pollute ``rt``.
  RuntimeContext child(rt.kernel_ctx());
  child.functions() = rt.functions();
  child.tensors() = rt.tensors();
  RunGraph(branch, child);

  std::vector<Tensor> outputs;
  outputs.reserve(static_cast<size_t>(branch.output_size()));
  for (size_t i = 0; i < branch.output().size(); ++i) {
    const std::string out_name = branch.output()[i].name().as_string();
    EXT_ENFORCE_INVALID(!out_name.empty(), "kernel::If: a subgraph output has an empty name.");
    auto it = child.tensors().find(out_name);
    EXT_ENFORCE_INVALID(it != child.tensors().end(),
                        "kernel::If: subgraph output '" + out_name +
                            "' was not produced by the selected branch.");
    outputs.push_back(std::move(it->second));
  }
  return outputs;
}

} // namespace kernel
} // namespace onnx_kernels
} // namespace ONNX_LIGHT_NAMESPACE

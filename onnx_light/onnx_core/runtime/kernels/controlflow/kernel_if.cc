// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/controlflow/include_controlflow_kernels.h"

#include "onnx_core/runtime/kernels/run_nodes.h"
#include "onnx_core/runtime/runtime_context.h"

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

Tensor If::operator()(const Tensor &cond, const Tensor &then_value,
                      const Tensor &else_value) const {
  EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL,
                      "kernel::If: 'cond' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "kernel::If: 'cond' must contain a single element.");
  EXT_ENFORCE_INVALID(then_value.data_type == else_value.data_type,
                      "kernel::If: 'then_value' and 'else_value' must have the same data type.");
  EXT_ENFORCE_INVALID(then_value.shape == else_value.shape,
                      "kernel::If: 'then_value' and 'else_value' must have the same shape.");
  const bool taken = cond.bytes()[0] != 0;
  return taken ? then_value : else_value;
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
  EXT_ENFORCE_INVALID(output.size_bytes() == then_value.size_bytes(),
                      "kernel::If preallocated output buffer has unexpected size in bytes.");

  const bool taken = cond.bytes()[0] != 0;
  const Tensor &src = taken ? then_value : else_value;
  if (src.size_bytes() > 0) {
    std::memcpy(output.mutable_bytes(), src.bytes(), src.size_bytes());
  }
}

Tensors If::operator()(RuntimeContext &rt, const Tensor &cond, const GraphProto &then_branch,
                       const GraphProto &else_branch) const {
  EXT_ENFORCE_INVALID(cond.data_type == DataType::BOOL,
                      "kernel::If: 'cond' must be a BOOL tensor.");
  EXT_ENFORCE_INVALID(cond.element_count() == 1,
                      "kernel::If: 'cond' must contain a single element.");
  EXT_ENFORCE_INVALID(then_branch.output_size() == else_branch.output_size(),
                      "kernel::If: 'then_branch' and 'else_branch' must declare the same number "
                      "of outputs.");

  const bool taken = cond.bytes()[0] != 0;
  const GraphProto &branch = taken ? then_branch : else_branch;
  const std::string branch_name = taken ? "then_branch" : "else_branch";

  // Build the session over the selected branch only; the branch itself is
  // only needed here, not kept around afterwards. Unlike Loop / Scan, If
  // selects and runs a branch exactly once per invocation, so there is no
  // repeated iteration to amortize the session over, but the same
  // construct-then-run separation keeps the control-flow kernels
  // consistent with one another.
  SubgraphSession session(rt, branch);
  return session.Run({}, rt, branch_name);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

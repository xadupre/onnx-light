// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/gradient/grad_common.h"

namespace ONNX_LIGHT_NAMESPACE::core::gradient {

void AccumulateGrad(const std::string &contrib_name, std::string &acc_name, int &counter,
                    FunctionProto &func) {
  if (acc_name.empty()) {
    acc_name = contrib_name;
  } else {
    std::string new_acc = "grad_acc_" + std::to_string(counter++);
    func.add_node("Add", {acc_name, contrib_name}, {new_acc});
    acc_name = new_acc;
  }
}

std::string NewGradName(const std::string &prefix, int &counter) {
  return prefix + "_" + std::to_string(counter++);
}

} // namespace ONNX_LIGHT_NAMESPACE::core::gradient

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/kernel_context.h"

#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_light_helpers.h"

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

// Default implementation for kernels that are not runnable through the dispatch
// path (e.g. helper kernels used only via their ``operator()``). Every
// dispatch-registered kernel overrides this.
void KernelBase::Run(RuntimeContext & /*rt*/) {
  EXT_THROW_INVALID("KernelBase::Run: this kernel does not implement Run().");
}

KernelTuningKey KernelBase::TuningKey(int32_t) const { return {}; }

void KernelBase::Configure(const KernelTuningParameters &) {
  EXT_THROW_INVALID("KernelBase::Configure is not implemented for this tunable kernel.");
}

} // namespace ONNX_LIGHT_NAMESPACE::core::runtime

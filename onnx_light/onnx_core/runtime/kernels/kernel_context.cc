// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/runtime/kernels/kernel_context.h"

#include "onnx_core/runtime/tuning/kernel_tuning.h"
#include "onnx_light_helpers.h"

#include <atomic>

namespace ONNX_LIGHT_NAMESPACE::core::runtime {

namespace {

std::atomic<uint64_t> kernel_construction_count{0};
std::atomic<int64_t> live_kernel_instance_count{0};

void RecordKernelConstruction() {
  kernel_construction_count.fetch_add(1, std::memory_order_relaxed);
  live_kernel_instance_count.fetch_add(1, std::memory_order_relaxed);
}

} // namespace

KernelBase::KernelBase(const KernelContext &ctx) : ctx_(ctx) { RecordKernelConstruction(); }

KernelBase::KernelBase(const KernelBase &other) : ctx_(other.ctx_), node_(other.node_) {
  RecordKernelConstruction();
}

KernelBase::KernelBase(KernelBase &&other) noexcept
    : ctx_(std::move(other.ctx_)), node_(other.node_) {
  RecordKernelConstruction();
}

KernelBase::~KernelBase() { live_kernel_instance_count.fetch_sub(1, std::memory_order_relaxed); }

uint64_t KernelBase::ConstructionCountForTesting() {
  return kernel_construction_count.load(std::memory_order_relaxed);
}

int64_t KernelBase::LiveInstanceCountForTesting() {
  return live_kernel_instance_count.load(std::memory_order_relaxed);
}

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

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <onnxruntime_lite_custom_op.h>

#include "onnx_core/runtime/tuning/cpu_executor.h"
#include "onnx_extensions/kernels/kernels/tensor/include_tensor_kernels.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#if ORT_API_VERSION < 17
#error "This example requires ONNX Runtime 1.17 or newer for KernelContext::ParallelFor."
#endif

namespace {

namespace runtime = ONNX_LIGHT_NAMESPACE::core::runtime;
namespace kernels = ONNX_LIGHT_NAMESPACE::onnx_kernels;

constexpr const char *kDomain = "com.example";

struct OrtBlockContext {
  void *block_context;
  runtime::CpuParallelBlockFn block_function;
};

void RunOrtBlock(void *context, size_t block_index) noexcept {
  auto &block = *static_cast<OrtBlockContext *>(context);
  block.block_function(block.block_context, static_cast<int64_t>(block_index));
}

void DispatchWithOrt(void *context, int64_t num_blocks, void *block_context,
                     runtime::CpuParallelBlockFn block_function) {
  OrtBlockContext ort_block{block_context, block_function};
  Ort::KernelContext(static_cast<OrtKernelContext *>(context))
      .ParallelFor(&RunOrtBlock, static_cast<size_t>(num_blocks), 0, &ort_block);
}

uint32_t ExternalParticipantLimit() noexcept {
  return std::max(1u, std::thread::hardware_concurrency());
}

class OrtAffineGridKernel {
public:
  OrtAffineGridKernel(const OrtApi *api, const OrtKernelInfo *info)
      : kernel_(runtime::KernelContext{runtime::DefaultOpset(20)}) {
    OrtStatus *status = api->KernelInfoGetAttribute_int64(info, "align_corners", &align_corners_);
    if (status != nullptr) {
      const std::string message = api->GetErrorMessage(status);
      if (message.find("No attribute with name:") != std::string::npos) {
        api->ReleaseStatus(status);
      } else {
        Ort::ThrowOnError(status);
      }
    }
    if (align_corners_ != 0 && align_corners_ != 1) {
      ORT_CXX_API_THROW("align_corners must be 0 or 1.", ORT_INVALID_ARGUMENT);
    }
  }

  Ort::Status Compute(OrtKernelContext *context, const Ort::Custom::Tensor<float> &theta,
                      const Ort::Custom::Tensor<int64_t> &size,
                      Ort::Custom::Tensor<float> &output) {
    runtime::Tensor onnx_theta = runtime::Tensor::Borrow(
        "theta", static_cast<int32_t>(runtime::DataType::FLOAT), theta.Shape(),
        reinterpret_cast<const uint8_t *>(theta.Data()), theta.SizeInBytes());
    runtime::Tensor onnx_size = runtime::Tensor::Borrow(
        "size", static_cast<int32_t>(runtime::DataType::INT64), size.Shape(),
        reinterpret_cast<const uint8_t *>(size.Data()), size.SizeInBytes());
    const runtime::Shape output_shape =
        kernels::kernel::AffineGrid::ComputeOutputShape(onnx_theta, onnx_size);
    float *output_data = output.Allocate(output_shape);
    runtime::Tensor onnx_output =
        runtime::Tensor::Borrow("grid", static_cast<int32_t>(runtime::DataType::FLOAT),
                                output_shape, reinterpret_cast<const uint8_t *>(output_data),
                                static_cast<size_t>(output.NumberOfElement()) * sizeof(float));

    std::unique_ptr<runtime::CpuExecutor> executor =
        runtime::CpuExecutor::CreateExternal(ExternalParticipantLimit(), context, &DispatchWithOrt);
    const runtime::CpuExecutorScope executor_scope(executor.get());
    kernel_(onnx_theta, onnx_size, kernels::kernel::AffineGrid::Attributes{align_corners_},
            onnx_output);
    return Ort::Status{nullptr};
  }

private:
  int64_t align_corners_ = 0;
  kernels::kernel::AffineGrid kernel_;
};

} // namespace

extern "C" {

ORT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                     const OrtApiBase *api_base) {
  const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
  Ort::InitApi(api);
  static const std::unique_ptr<Ort::Custom::OrtLiteCustomOp> affine_grid{
      Ort::Custom::CreateLiteCustomOp<OrtAffineGridKernel>("AffineGrid", "CPUExecutionProvider")};
  static std::mutex mutex;
  static OrtCustomOpDomain *domain = nullptr;
  std::lock_guard<std::mutex> lock(mutex);
  if (domain == nullptr) {
    OrtStatus *status = api->CreateCustomOpDomain(kDomain, &domain);
    if (status != nullptr) {
      return status;
    }
    status = api->CustomOpDomain_Add(domain, affine_grid.get());
    if (status != nullptr) {
      api->ReleaseCustomOpDomain(domain);
      domain = nullptr;
      return status;
    }
  }
  return api->AddCustomOpDomain(options, domain);
}

} // extern "C"

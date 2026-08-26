// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#if ORT_API_VERSION < 17
#error "This example requires ONNX Runtime 1.17 or newer for KernelContext::ParallelFor."
#endif

namespace {

constexpr const char *kDomain = "com.example";

struct AffineGridTask {
  const float *theta;
  float *output;
  int64_t depth;
  int64_t height;
  int64_t width;
  bool align_corners;
  bool is_3d;
};

float NormalizedCoordinate(int64_t index, int64_t size, bool align_corners) noexcept {
  if (size <= 1) {
    return align_corners ? -1.0f : 0.0f;
  }
  if (align_corners) {
    return -1.0f + 2.0f * static_cast<float>(index) / static_cast<float>(size - 1);
  }
  return -1.0f + (2.0f * static_cast<float>(index) + 1.0f) / static_cast<float>(size);
}

void ComputeRow(void *opaque, size_t row_index) noexcept {
  const auto &task = *static_cast<const AffineGridTask *>(opaque);
  const int64_t row = static_cast<int64_t>(row_index);

  if (!task.is_3d) {
    const int64_t batch = row / task.height;
    const int64_t y = row % task.height;
    const float normalized_y = NormalizedCoordinate(y, task.height, task.align_corners);
    const float *matrix = task.theta + batch * 6;
    float *output_row = task.output + row * task.width * 2;
    for (int64_t x = 0; x < task.width; ++x) {
      const float normalized_x = NormalizedCoordinate(x, task.width, task.align_corners);
      output_row[x * 2] = matrix[0] * normalized_x + matrix[1] * normalized_y + matrix[2];
      output_row[x * 2 + 1] = matrix[3] * normalized_x + matrix[4] * normalized_y + matrix[5];
    }
    return;
  }

  const int64_t rows_per_batch = task.depth * task.height;
  const int64_t batch = row / rows_per_batch;
  const int64_t spatial_row = row % rows_per_batch;
  const int64_t z = spatial_row / task.height;
  const int64_t y = spatial_row % task.height;
  const float normalized_z = NormalizedCoordinate(z, task.depth, task.align_corners);
  const float normalized_y = NormalizedCoordinate(y, task.height, task.align_corners);
  const float *matrix = task.theta + batch * 12;
  float *output_row = task.output + row * task.width * 3;
  for (int64_t x = 0; x < task.width; ++x) {
    const float normalized_x = NormalizedCoordinate(x, task.width, task.align_corners);
    for (int64_t axis = 0; axis < 3; ++axis) {
      const float *matrix_row = matrix + axis * 4;
      output_row[x * 3 + axis] = matrix_row[0] * normalized_x + matrix_row[1] * normalized_y +
                                 matrix_row[2] * normalized_z + matrix_row[3];
    }
  }
}

void Require(bool condition, const char *message) {
  if (!condition) {
    ORT_CXX_API_THROW(message, ORT_INVALID_ARGUMENT);
  }
}

class AffineGridKernel {
public:
  AffineGridKernel(const OrtApi &, const OrtKernelInfo *info) : align_corners_(false) {
    try {
      const int64_t value = Ort::ConstKernelInfo(info).GetAttribute<int64_t>("align_corners");
      Require(value == 0 || value == 1, "align_corners must be 0 or 1.");
      align_corners_ = value != 0;
    } catch (const Ort::Exception &exception) {
      const std::string message = exception.what();
      if (message.find("No attribute with name:") == std::string::npos) {
        throw;
      }
    }
  }

  void Compute(OrtKernelContext *context) {
    Ort::KernelContext kernel_context(context);
    const Ort::ConstValue theta_value = kernel_context.GetInput(0);
    const Ort::ConstValue size_value = kernel_context.GetInput(1);
    const std::vector<int64_t> theta_shape = theta_value.GetTensorTypeAndShapeInfo().GetShape();
    const std::vector<int64_t> size_shape = size_value.GetTensorTypeAndShapeInfo().GetShape();

    Require(theta_shape.size() == 3, "theta must have rank 3.");
    Require(size_shape.size() == 1 && (size_shape[0] == 4 || size_shape[0] == 5),
            "size must be a rank-1 tensor containing 4 or 5 values.");

    const bool is_3d = size_shape[0] == 5;
    Require((!is_3d && theta_shape[1] == 2 && theta_shape[2] == 3) ||
                (is_3d && theta_shape[1] == 3 && theta_shape[2] == 4),
            "theta must have shape [N,2,3] for 2D or [N,3,4] for 3D.");

    const int64_t *size = size_value.GetTensorData<int64_t>();
    Require(size[0] == theta_shape[0], "size[0] must match theta's batch dimension.");
    Require(size[0] >= 0, "The batch dimension must not be negative.");
    for (int64_t index = 2; index < size_shape[0]; ++index) {
      Require(size[index] >= 0, "Spatial dimensions must not be negative.");
    }

    std::vector<int64_t> output_shape;
    if (is_3d) {
      output_shape = {size[0], size[2], size[3], size[4], 3};
    } else {
      output_shape = {size[0], size[2], size[3], 2};
    }
    Ort::UnownedValue output_value = kernel_context.GetOutput(0, output_shape);

    const int64_t depth = is_3d ? size[2] : 1;
    const int64_t height = is_3d ? size[3] : size[2];
    const int64_t width = is_3d ? size[4] : size[3];
    const int64_t maximum = std::numeric_limits<int64_t>::max();
    Require(depth == 0 || size[0] <= maximum / depth, "The output row count is too large.");
    const int64_t batch_depth = size[0] * depth;
    Require(height == 0 || batch_depth <= maximum / height, "The output row count is too large.");
    const int64_t row_count = batch_depth * height;
    Require(static_cast<uint64_t>(row_count) <= std::numeric_limits<size_t>::max(),
            "The output row count does not fit size_t.");
    if (row_count == 0 || width == 0) {
      return;
    }

    AffineGridTask task{
        theta_value.GetTensorData<float>(),
        output_value.GetTensorMutableData<float>(),
        depth,
        height,
        width,
        align_corners_,
        is_3d,
    };

    // num_batch == 0 asks ONNX Runtime to choose the partitioning. The work is
    // scheduled on the operator's intra-op pool; this library creates no threads.
    kernel_context.ParallelFor(ComputeRow, static_cast<size_t>(row_count), 0, &task);
  }

private:
  bool align_corners_;
};

struct AffineGridCustomOp : Ort::CustomOpBase<AffineGridCustomOp, AffineGridKernel> {
  void *CreateKernel(const OrtApi &api, const OrtKernelInfo *info) const {
    return std::make_unique<AffineGridKernel>(api, info).release();
  }

  const char *GetName() const { return "AffineGrid"; }
  const char *GetExecutionProviderType() const { return "CPUExecutionProvider"; }
  size_t GetInputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

void RetainDomain(Ort::CustomOpDomain &&domain) {
  static std::mutex mutex;
  static std::vector<Ort::CustomOpDomain> domains;
  std::lock_guard<std::mutex> lock(mutex);
  domains.push_back(std::move(domain));
}

} // namespace

extern "C" {

ORT_EXPORT OrtStatus *ORT_API_CALL RegisterCustomOps(OrtSessionOptions *options,
                                                     const OrtApiBase *api_base) {
  const OrtApi *api = api_base->GetApi(ORT_API_VERSION);
  Ort::InitApi(api);
  try {
    static const AffineGridCustomOp affine_grid;
    Ort::CustomOpDomain domain(kDomain);
    domain.Add(&affine_grid);
    Ort::UnownedSessionOptions session_options(options);
    session_options.Add(domain);
    RetainDomain(std::move(domain));
    return nullptr;
  } catch (const Ort::Exception &exception) {
    return api->CreateStatus(exception.GetOrtErrorCode(), exception.what());
  } catch (const std::exception &exception) {
    return api->CreateStatus(ORT_RUNTIME_EXCEPTION, exception.what());
  }
}

} // extern "C"

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " MODEL.onnx CUSTOM_OP_LIBRARY\n";
    return 2;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ort-affine-grid-example");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(4);
  options.RegisterCustomOpsLibrary(argv[2]);
  Ort::Session session(env, argv[1], options);

  std::array<float, 6> theta{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f};
  constexpr int64_t height = 256;
  constexpr int64_t width = 256;
  std::array<int64_t, 4> size{1, 1, height, width};
  const std::array<int64_t, 3> theta_shape{1, 2, 3};
  const std::array<int64_t, 1> size_shape{4};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<Ort::Value, 2> inputs{
      Ort::Value::CreateTensor<float>(memory, theta.data(), theta.size(), theta_shape.data(),
                                      theta_shape.size()),
      Ort::Value::CreateTensor<int64_t>(memory, size.data(), size.size(), size_shape.data(),
                                        size_shape.size()),
  };
  const std::array<const char *, 2> input_names{"theta", "size"};
  const std::array<const char *, 1> output_names{"grid"};
  std::vector<Ort::Value> outputs =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                  output_names.data(), output_names.size());

  const std::vector<int64_t> shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
  const std::vector<int64_t> expected_shape{1, height, width, 2};
  if (shape != expected_shape) {
    std::cerr << "Unexpected output shape.\n";
    return 1;
  }

  const float *actual = outputs[0].GetTensorData<float>();
  for (int64_t y = 0; y < height; ++y) {
    const float expected_y =
        -1.0f + (2.0f * static_cast<float>(y) + 1.0f) / static_cast<float>(height);
    for (int64_t x = 0; x < width; ++x) {
      const float expected_x =
          -1.0f + (2.0f * static_cast<float>(x) + 1.0f) / static_cast<float>(width);
      const size_t offset = static_cast<size_t>((y * width + x) * 2);
      if (std::abs(actual[offset] - expected_x) > 1e-6f ||
          std::abs(actual[offset + 1] - expected_y) > 1e-6f) {
        std::cerr << "Unexpected grid value at (" << y << ", " << x << ").\n";
        return 1;
      }
    }
  }

  std::cout << "AffineGrid custom op produced the expected [1, 256, 256, 2] grid.\n";
  return 0;
}

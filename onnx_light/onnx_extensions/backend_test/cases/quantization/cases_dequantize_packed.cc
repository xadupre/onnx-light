// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/cast_helper.h"
#include "onnx_extensions/backend_test/cases/quantization/include_quantization_cases.h"
#include "onnx_proto/onnx_helper.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void RegisterDequantizePackedCases(std::vector<TestCase> &registry, TestMode mode) {
  if (mode == TestMode::BENCHMARK)
    return;
  for (const std::string kind : {"q4_0", "q5_1", "iq4_nl", "mxfp4", "empty"}) {
    NodeProto node;
    node.set_op_type("DequantizePacked");
    node.add_input("x");
    node.add_output("y");
    AddAttribute<std::string>(node, "format", "gguf:" + (kind == "empty" ? "q2_k" : kind));
    const std::vector<int64_t> shape = kind == "q4_0"    ? std::vector<int64_t>{2, 32}
                                       : kind == "empty" ? std::vector<int64_t>{0, 256}
                                                         : std::vector<int64_t>{32};
    AddAttribute<std::vector<int64_t>>(node, "shape", shape);
    if (kind == "iq4_nl")
      AddAttribute<int64_t>(node, "output_dtype", TensorProto::FLOAT16);
    const std::string name =
        "test_dequantizepacked_" + (kind == "iq4_nl" ? "iq4_nl_float16" : kind);
    Expect(registry, std::move(node), name, {DefaultOpset(29)}, [kind, shape]() -> IoData {
      std::vector<uint8_t> bytes;
      std::vector<float> expected;
      if (kind == "q4_0") {
        bytes = {0, 0x38};
        for (int j = 0; j < 16; ++j) {
          bytes.push_back(j);
          expected.push_back((j - 8) * 0.5f);
        }
        expected.insert(expected.end(), 16, -4.0f);
        bytes.push_back(0);
        bytes.push_back(0xc0);
        bytes.insert(bytes.end(), 16, 0xf0);
        expected.insert(expected.end(), 16, 16.0f);
        expected.insert(expected.end(), 16, -14.0f);
      } else if (kind == "q5_1") {
        bytes = {0, 0x38, 0, 0xbc, 0, 0, 0xff, 0xff};
        for (int j = 0; j < 16; ++j) {
          bytes.push_back(j);
          expected.push_back(j * 0.5f - 1);
        }
        expected.insert(expected.end(), 16, 7.0f);
      } else if (kind == "iq4_nl") {
        bytes = {0, 0x38};
        constexpr int values[] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                  1,    13,   25,  38,  53,  69,  89,  113};
        for (int j = 0; j < 16; ++j) {
          bytes.push_back(j);
          expected.push_back(values[j] * 0.5f);
        }
        expected.insert(expected.end(), 16, -63.5f);
      } else if (kind == "mxfp4") {
        bytes = {127};
        expected = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6};
        for (int j = 0; j < 16; ++j)
          bytes.push_back(j);
        expected.insert(expected.end(), 16, 0.0f);
      }
      const int64_t length = bytes.size();
      Tensor x("", TensorProto::UINT8, {length}, std::move(bytes));
      Tensor y = kind == "iq4_nl" ? MakeFloat16Tensor("", shape, expected)
                                  : Tensor::FromFloat("", shape, expected);
      return IoData{{std::move(x)}, {std::move(y)}};
    });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/cases/tensor/include_tensor_cases.h"
#include "onnx_backend_test/test_case.h"
#include "onnx_kernels/kernels/tensor/include_tensor_kernels.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {
namespace onnx_backend_test {

// ---------------------------------------------------------------------------
// Tile — output[i_0, ..., i_{r-1}] = input[i_0 mod d_0, ..., i_{r-1} mod d_{r-1}]
// where d_k is the k-th input dim, and the output shape is
// input.shape[k] * repeats[k] (since opset 6 in the ai.onnx domain).
// Mirrors the upstream ONNX node tests in
// ``onnx/backend/test/case/node/tile.py``.
// ---------------------------------------------------------------------------

namespace {

NodeProto MakeTileNode() {
  NodeProto node;
  node.set_op_type("Tile");
  node.add_input("x");
  node.add_input("y");
  node.add_output("z");
  return node;
}

// Builds a 1-D INT64 ``repeats`` tensor from the given values.
Tensor MakeRepeatsTensor(const std::vector<int64_t> &repeats) {
  const std::vector<int64_t> shape = {static_cast<int64_t>(repeats.size())};
  std::vector<uint8_t> data(repeats.size() * sizeof(int64_t));
  std::memcpy(data.data(), repeats.data(), data.size());
  return Tensor("", DataType::INT64, shape, std::move(data));
}

} // namespace

void RegisterTileCases(std::vector<TestCase> &registry) {
  const OpsetId opset = DefaultOpset(13);
  const kernel::KernelContext ctx{opset};
  const kernel::Tile tile_kernel{ctx};

  // test_cc_tile_precomputed — matches the upstream ``test_tile_precomputed``
  // node test exactly (small deterministic case suitable for parity checks).
  //
  // input:    [[0, 1], [2, 3]]  (shape [2, 2])
  // repeats:  [2, 2]
  // output:   [[0, 1, 0, 1],
  //            [2, 3, 2, 3],
  //            [0, 1, 0, 1],
  //            [2, 3, 2, 3]]    (shape [4, 4])
  {
    const Tensor input = Tensor::FromFloat("", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
    const Tensor repeats = MakeRepeatsTensor({2, 2});
    const Tensor output = tile_kernel(input, repeats);
    Expect(MakeTileNode(), {input, repeats}, {output}, "test_cc_tile_precomputed", {opset},
           "backend-test", registry);
  }

  // test_cc_tile_1d — repeat a 1-D tensor along its only axis.
  //
  // input:   [1, 2, 3]
  // repeats: [3]
  // output:  [1, 2, 3, 1, 2, 3, 1, 2, 3]
  {
    const Tensor input = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
    const Tensor repeats = MakeRepeatsTensor({3});
    const Tensor output = tile_kernel(input, repeats);
    Expect(MakeTileNode(), {input, repeats}, {output}, "test_cc_tile_1d", {opset}, "backend-test",
           registry);
  }

  // test_cc_tile_repeats_one — repeating with all-ones leaves the tensor
  // unchanged (output shape == input shape).
  //
  // input:   [[1, 2], [3, 4]]
  // repeats: [1, 1]
  // output:  same as input
  {
    const Tensor input = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    const Tensor repeats = MakeRepeatsTensor({1, 1});
    const Tensor output = tile_kernel(input, repeats);
    Expect(MakeTileNode(), {input, repeats}, {output}, "test_cc_tile_repeats_one", {opset},
           "backend-test", registry);
  }
}

} // namespace onnx_backend_test
} // namespace ONNX_LIGHT_NAMESPACE

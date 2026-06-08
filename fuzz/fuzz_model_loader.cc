// Copyright (c) ONNX Project Contributors
// SPDX-License-Identifier: Apache-2.0
//
// libFuzzer harness for the ``ModelProto`` loader. Distinct from
// ``fuzz_checker`` in that it also pokes at the parsed graph's
// node/input/output lists before invoking the structural checker —
// catching bugs that would only manifest when the parsed object is
// actually walked. Mirrors the former
// ``onnx_light/fuzz/fuzz_model_loader.py``.

#include "onnx_lib/checker.h"
#include "onnx_proto/onnx.h"

#include <cstddef>
#include <cstdint>
#include <string>

using ONNX_LIGHT_NAMESPACE::ModelProto;
namespace checker = ONNX_LIGHT_NAMESPACE::checker;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  ModelProto model;
  try {
    model.ParseFromString(std::string(reinterpret_cast<const char *>(data), size));
  } catch (...) {
    return 0;
  }

  // Force the loader to materialise the graph and walk its top-level
  // repeated fields. ``volatile`` keeps the compiler from optimising the
  // reads away under -O2.
  try {
    if (model.has_graph()) {
      const auto &graph = model.ref_graph();
      volatile size_t n_nodes = graph.ref_node().size();
      volatile size_t n_inputs = graph.ref_input().size();
      volatile size_t n_outputs = graph.ref_output().size();
      (void)n_nodes;
      (void)n_inputs;
      (void)n_outputs;
    }
  } catch (...) {
    return 0;
  }

  try {
    checker::check_model(model);
  } catch (...) {
    // Expected on malformed inputs.
  }
  return 0;
}

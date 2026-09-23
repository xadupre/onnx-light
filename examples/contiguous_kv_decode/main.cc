// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/compute/raw_buffer_allocator.h"
#include "onnx_core/runtime/persistent_value_state.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include <chrono>
#include <cmath>
#include <iostream>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace ONNX_LIGHT_NAMESPACE::core::runtime;

namespace {

constexpr int64_t kWidth = 4;

ModelProto DecodeModel(int64_t heads) {
  ModelProto model;
  model.set_ir_version(10);
  model.add_opset_import()->set_version(24);
  auto *graph = model.mutable_graph();
  graph->set_name("contiguous_kv_decode");
  auto declare = [heads](ValueInfoProto *value, const char *name, bool variable_length) {
    value->set_name(name);
    auto *tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    auto *shape = tensor->mutable_shape();
    shape->add_dim()->set_dim_value(1);
    shape->add_dim()->set_dim_value(heads);
    if (variable_length)
      shape->add_dim();
    else
      shape->add_dim()->set_dim_value(1);
    shape->add_dim()->set_dim_value(kWidth);
  };
  for (const char *name : {"q", "k", "v"})
    declare(graph->add_input(), name, false);
  for (const char *name : {"past_key", "past_value"})
    declare(graph->add_input(), name, true);
  declare(graph->add_output(), "y", false);
  for (const char *name : {"present_key", "present_value"})
    declare(graph->add_output(), name, true);
  auto *node = graph->add_node();
  node->set_op_type("Attention");
  for (const char *name : {"q", "k", "v", "", "past_key", "past_value"})
    node->add_input(name);
  for (const char *name : {"y", "present_key", "present_value"})
    node->add_output(name);
  for (const auto &[input, output] :
       {std::pair{"past_key", "present_key"}, std::pair{"past_value", "present_value"}}) {
    auto *binding = graph->add_persistent_bindings();
    binding->set_input_name(input);
    binding->set_output_name(output);
  }
  return model;
}

RuntimeValue Filled(int64_t heads, int64_t length, float value) {
  return RuntimeValue(
      Tensor::FromFloat("", {1, heads, length, kWidth},
                        std::vector<float>(static_cast<size_t>(heads * length * kWidth), value)));
}

} // namespace

int main() {
  onnx_kernels::RegisterKernelFunctions();
  std::cout << "kv_heads,token,elapsed_ns,kv_allocations,kv_allocated_bytes,"
               "prefix_copied_bytes,append_copied_bytes,reused_buffers,"
               "io_live_bytes,io_peak_bytes,state_alias_verified\n";
  for (int64_t heads : {1, 2}) {
    const ModelProto model = DecodeModel(heads);
    SimpleRawBufferAllocator execution(64);
    auto io = IOArena::Create(32);
    RuntimeContext context(KernelContext(DefaultOpset(24)),
                           RuntimeContextOptions{.allocator = &execution,
                                                 .io_allocator = io.get(),
                                                 .events_enabled = true});
    PersistentValueState state(
        model, {{"past_key", Filled(heads, 0, 0)}, {"past_value", Filled(heads, 0, 0)}},
        RuntimeSessionOptions{.persistent_tensor_initial_capacity = 4});
    for (int step = 0; step < 20; ++step) {
      context.ClearEvents();
      RuntimeValueMap feeds{{"q", Filled(heads, 1, 0)},
                            {"k", Filled(heads, 1, 0)},
                            {"v", Filled(heads, 1, static_cast<float>(step + 1))}};
      const auto start = std::chrono::steady_clock::now();
      const auto output = state.Run(context, feeds);
      const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - start);
      const Tensor &y = output.at("y").tensor;
      for (int64_t i = 0; i < y.element_count(); ++i) {
        if (std::fabs(y.AsFloat()[i] - static_cast<float>(step + 2) / 2) > 1e-5f) {
          std::cerr << "Attention result differs from the uniform-attention mean.\n";
          return 1;
        }
      }
      const auto retained = state.Values();
      if (retained.at("past_key").tensor.bytes() != output.at("present_key").tensor.bytes() ||
          retained.at("past_value").tensor.bytes() != output.at("present_value").tensor.bytes()) {
        std::cerr << "State forwarding unexpectedly copied a KV payload.\n";
        return 1;
      }
      uint64_t allocations = 0, allocated_bytes = 0, prefix_copied = 0, append_copied = 0,
               reused = 0;
      for (const auto &event : context.events()) {
        if (event.action != RuntimeEventAction::kPersistentStorage)
          continue;
        allocations += event.storage_allocations;
        allocated_bytes += event.storage_allocated_bytes;
        prefix_copied += event.storage_prefix_copied_bytes;
        append_copied += event.storage_append_copied_bytes;
        reused += event.storage_reuse_count;
      }
      std::cout << heads << "," << step + 1 << "," << elapsed.count() << "," << allocations << ","
                << allocated_bytes << "," << prefix_copied << "," << append_copied << "," << reused
                << "," << io->TotalAllocatedSize() << "," << io->PeakAllocatedSize() << ",1\n";
    }
  }
}

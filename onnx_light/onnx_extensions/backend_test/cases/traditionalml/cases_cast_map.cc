// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/expect.h"
#include "onnx_core/runtime/kernels/random.h"
#include "onnx_core/runtime/memory/simple_map.h"
#include "onnx_extensions/backend_test/cases/traditionalml/include_traditionalml_cases.h"
#include "onnx_extensions/kernels/kernels/traditionalml/include_traditionalml_kernels.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

namespace {

void AddStringAttr(NodeProto &node, const char *name, const std::string &value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::STRING);
  attr->set_s(value);
}

void AddIntAttr(NodeProto &node, const char *name, int64_t value) {
  AttributeProto *attr = node.add_attribute();
  attr->set_name(name);
  attr->set_type(AttributeProto::AttributeType::INT);
  attr->set_i(value);
}

} // namespace

// ---------------------------------------------------------------------------
// CastMap — converts a ``map(int64, X)`` input into a 1-D output tensor whose
// length is either the number of keys (``DENSE``) or ``max_map`` (``SPARSE``).
// The output element type is controlled by the ``cast_to`` attribute. Mirrors
// the upstream ONNX ``ai.onnx.ml::CastMap`` operator (since opset 1).
//
// The map input is passed as a ``Map`` object in ``IoData::maps``; the runtime
// retrieves it by name from ``RuntimeContext::maps()``.
// ---------------------------------------------------------------------------
void RegisterCastMapCases(std::vector<TestCase> &registry, TestMode mode) {
  const OpsetId opset("ai.onnx.ml", 1);
  const OpsetId default_opset = DefaultOpset(13);

  if (mode == TestMode::BENCHMARK) {
    const int64_t count = 65536;
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_FLOAT");
    AddStringAttr(node, "map_form", "DENSE");
    Expect(registry, std::move(node), "test_cc_cast_map_benchmark", {default_opset, opset},
           [opset]() -> IoData {
             const KernelContext cast_map_ctx{opset};
             const onnx_kernels::kernel::CastMap cast_map{cast_map_ctx};

             std::vector<int64_t> keys(static_cast<size_t>(count));
             for (int64_t i = 0; i < count; ++i) {
               keys[static_cast<size_t>(i)] = i;
             }
             std::vector<float> values = Randn<float>({count}, 2001);
             Map x("x", Tensor::FromInt64("", {count}, keys),
                   Tensor::FromFloat("", {count}, values));
             Tensor y =
                 cast_map.template operator()<float, float>(keys, values, "TO_FLOAT", "DENSE", 0);
             return IoData{{}, {std::move(y)}, {std::move(x)}};
           });
    return;
  }

  // DENSE map(int64, float) -> tensor(float). Keys are not sorted on input;
  // the operator must sort them ascending in the output.
  {
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_FLOAT");
    AddStringAttr(node, "map_form", "DENSE");
    Expect(registry, std::move(node), "test_cc_cast_map_int64_float_dense", {default_opset, opset},
           [opset]() -> IoData {
             const KernelContext cast_map_ctx{opset};
             const onnx_kernels::kernel::CastMap cast_map{cast_map_ctx};

             const std::vector<int64_t> keys{2, 0, 1};
             const std::vector<float> values{2.5f, 0.5f, 1.5f};
             Map x("x", Tensor::FromInt64("", {3}, keys), Tensor::FromFloat("", {3}, values));
             Tensor y =
                 cast_map.template operator()<float, float>(keys, values, "TO_FLOAT", "DENSE", 0);
             return IoData{{}, {std::move(y)}, {std::move(x)}};
           });
  }

  // SPARSE map(int64, float) -> tensor(float). Missing positions are zero.
  {
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_FLOAT");
    AddStringAttr(node, "map_form", "SPARSE");
    AddIntAttr(node, "max_map", 5);
    Expect(registry, std::move(node), "test_cc_cast_map_int64_float_sparse", {default_opset, opset},
           [opset]() -> IoData {
             const KernelContext cast_map_ctx{opset};
             const onnx_kernels::kernel::CastMap cast_map{cast_map_ctx};

             const std::vector<int64_t> keys{1, 3};
             const std::vector<float> values{10.0f, 30.0f};
             Map x("x", Tensor::FromInt64("", {2}, keys), Tensor::FromFloat("", {2}, values));
             Tensor y =
                 cast_map.template operator()<float, float>(keys, values, "TO_FLOAT", "SPARSE", 5);
             return IoData{{}, {std::move(y)}, {std::move(x)}};
           });
  }

  // DENSE map(int64, string) -> tensor(string).
  {
    NodeProto node;
    node.set_op_type("CastMap");
    node.set_domain("ai.onnx.ml");
    node.add_input("x");
    node.add_output("y");
    AddStringAttr(node, "cast_to", "TO_STRING");
    AddStringAttr(node, "map_form", "DENSE");
    Expect(registry, std::move(node), "test_cc_cast_map_int64_string_dense", {default_opset, opset},
           [opset]() -> IoData {
             const KernelContext cast_map_ctx{opset};
             const onnx_kernels::kernel::CastMap cast_map{cast_map_ctx};

             const std::vector<int64_t> keys{1, 0};
             const std::vector<std::string> values{"b", "a"};
             Map x("x", Tensor::FromInt64("", {2}, keys), Tensor::FromStrings("", {2}, values));
             Tensor y = cast_map.template operator()<std::string, std::string>(
                 keys, values, "TO_STRING", "DENSE", 0);
             return IoData{{}, {std::move(y)}, {std::move(x)}};
           });
  }
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

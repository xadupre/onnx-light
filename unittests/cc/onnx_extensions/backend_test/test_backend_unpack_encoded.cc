// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case.h"
#include "onnx_core/runtime/runtime_session.h"
#include "onnx_extensions/kernels/kernel_dispatch_table.h"
#include "onnx_lib/checker.h"
#include "onnx_proto/onnx_helper.h"

#include <gtest/gtest.h>

#include <cstring>

using namespace ONNX_LIGHT_NAMESPACE;
using namespace core::backend_test;
using namespace core::runtime;

namespace {

void TensorInfo(ValueInfoProto &info, const char *name, int32_t dtype, const Shape &shape) {
  info.set_name(name);
  auto *tensor = info.mutable_type()->mutable_tensor_type();
  tensor->set_elem_type(dtype);
  tensor->mutable_shape();
  for (int64_t dimension : shape)
    tensor->mutable_shape()->add_dim()->set_dim_value(dimension);
}

// These cases exercise native encoded inputs, not the tensor-only shape-inference registry.
std::vector<TestCase> EncodedCases() {
  std::vector<TestCase> cases;
  for (const std::string kind : {"bits", "array", "references", "records", "empty", "uint8",
                                 "uint32", "uint32_axis", "ternary"}) {
    const std::string name = "test_unpack_encoded_" + kind;
    TestCase test(name, name);
    test.build = [kind, name](bool) {
      BuiltCase result;
      auto &model = result.model;
      model.set_ir_version(IR_VERSION);
      model.add_opset_import()->set_version(29);
      auto *graph = model.mutable_graph();
      graph->set_name(name);
      auto *encoded = graph->add_extended_initializer()->mutable_encoded_value();
      encoded->set_name("X");
      int64_t count = 8;
      int64_t bits = 3;
      int64_t axis = -1;
      Shape output_shape{8};
      std::vector<uint8_t> expected{0, 1, 2, 3, 4, 5, 6, 7};
      encoded->set_raw_data("\x88\xC6\xFA", 3);
      auto *type = encoded->mutable_struct_type();
      if (kind == "array" || kind == "references") {
        type->mutable_array()->set_dimension(2);
        type = type->mutable_array()->mutable_element_type()->mutable_struct_type();
      }
      if (kind == "uint8" || kind == "uint32" || kind == "uint32_axis") {
        const bool wide = kind != "uint8";
        auto *array = type->mutable_array();
        array->set_dimension(wide ? 1 : 3);
        auto *tensor = array->mutable_element_type()->mutable_tensor_type();
        tensor->set_elem_type(wide ? TensorProto::UINT32 : TensorProto::UINT8);
        tensor->mutable_shape();
        if (wide)
          encoded->set_raw_data("\x88\xC6\xFA\x00", 4);
        if (kind == "uint32_axis") {
          array->set_dimension(2);
          tensor->mutable_shape()->add_dim()->set_dim_value(2);
          encoded->set_raw_data(std::string(16, '\xFF'));
          axis = 0;
          bits = 5;
          count = 7;
          output_shape = {7, 2};
          expected.assign(14, 31);
        }
      } else {
        auto *packing = type->mutable_bit_packing();
        packing->set_dimension(kind == "ternary" ? 2 : 8);
        auto *component = packing->add_component();
        component->set_name("codes");
        component->set_bit_width(kind == "ternary" ? 8 : 3);
      }
      if (kind == "array" || kind == "references" || kind == "records") {
        encoded->set_raw_data("\x88\xC6\xFA\x88\xC6\xFA", 6);
        output_shape = {2, 8};
        expected = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
      }
      if (kind == "references") {
        auto *leaf = model.add_struct_types();
        *leaf = *type;
        leaf->set_type_id(1);
        type->set_type_ref(1);
        auto *root = model.add_struct_types();
        *root = encoded->struct_type();
        root->set_type_id(2);
        encoded->mutable_struct_type()->set_type_ref(2);
      }
      if (kind == "empty") {
        encoded->set_raw_data("", 0);
        output_shape = {0, 8};
        expected.clear();
      }
      if (kind == "ternary") {
        count = 7;
        encoded->set_raw_data("\x66\xF2", 2);
        output_shape = {7};
        expected = {0, 1, 2, 0, 1, 2, 2};
      }
      ValueInfoProto logical;
      TensorInfo(logical, "Y", TensorProto::UINT8, output_shape);
      *encoded->mutable_logical_type() = logical.type();
      *graph->add_output() = logical;
      TensorInfo(*graph->add_input(), "count", TensorProto::INT64, {});
      auto *node = graph->add_node();
      *node = MakeNode("Unpack", {"X", "count"}, {"Y"});
      AddAttribute<int64_t>(*node, "axis", axis);
      if (kind == "ternary") {
        AddAttribute<int64_t>(*node, "radix", 3);
        AddAttribute<int64_t>(*node, "group_size", 5);
      } else {
        AddAttribute<int64_t>(*node, "bits", bits);
      }
      DataSet data;
      data.inputs.push_back(Tensor::FromInt64("count", {}, {count}));
      data.outputs.push_back(Tensor::FromUint8("Y", output_shape, expected));
      result.data_sets.push_back(std::move(data));
      return result;
    };
    cases.push_back(std::move(test));
  }
  return cases;
}

} // namespace

TEST(BackendUnpackEncoded, LazyModelsExecuteInlineLayoutsAndCatalogueReferences) {
  onnx_kernels::RegisterKernelFunctions();
  auto cases = EncodedCases();
  ASSERT_EQ(cases.size(), 9u);
  for (auto &test : cases) {
    SCOPED_TRACE(test.name);
    EXPECT_FALSE(test.materialized());
    ASSERT_NO_THROW(checker::check_model(test.model(), /*full_check=*/false));
    for (const auto &data : test.data_sets()) {
      RuntimeContext context;
      for (const auto &input : data.inputs)
        context.Set(input.name, input);
      RuntimeSession session(test.model());
      session.Run(context);
      const auto &actual = context.Get("Y");
      const auto &expected = data.outputs[0];
      EXPECT_EQ(actual.data_type, expected.data_type);
      EXPECT_EQ(actual.shape, expected.shape);
      ASSERT_EQ(actual.size_bytes(), expected.size_bytes());
      if (actual.size_bytes()) {
        EXPECT_EQ(std::memcmp(actual.bytes(), expected.bytes(), actual.size_bytes()), 0);
      }
    }
    test.unload();
    EXPECT_FALSE(test.materialized());
    EXPECT_EQ(test.data_sets().size(), 1u);
  }
}

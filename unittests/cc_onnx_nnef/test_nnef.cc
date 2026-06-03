// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "nnef/exporter.h"
#include "nnef/tensor_io.h"
#include "onnx.h"
#include "onnx_proto/onnx_helper.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
namespace fs = std::filesystem;

namespace {

std::string MakeTmpDir(const std::string &prefix) {
  fs::path base = fs::temp_directory_path() /
                  (prefix + std::to_string(::getpid()) + "_" + std::to_string(std::rand()));
  fs::create_directories(base);
  return base.string();
}

ModelProto MakeIdentityModel() {
  ModelProto m;
  GraphProto &g = *m.mutable_graph();
  g.set_name("idg");
  *g.add_node() = MakeNode("Identity", {"X"}, {"Y"});
  auto *in = g.add_input();
  in->set_name("X");
  auto *out = g.add_output();
  out->set_name("Y");
  return m;
}

} // namespace

TEST(NNEFTensorIO, RoundtripFloat32) {
  nnef::NNEFTensor t;
  t.item_type = nnef::kItemTypeFloat;
  t.bits = 32;
  t.shape = {2, 3};
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  t.data.resize(data.size() * sizeof(float));
  std::memcpy(t.data.data(), data.data(), t.data.size());

  std::string dir = MakeTmpDir("nnef_io_");
  std::string path = dir + "/t.dat";
  nnef::WriteNNEFTensor(path, t);
  nnef::NNEFTensor back = nnef::ReadNNEFTensor(path);
  fs::remove_all(dir);

  EXPECT_EQ(back.item_type, nnef::kItemTypeFloat);
  EXPECT_EQ(back.bits, 32);
  EXPECT_EQ(back.shape, t.shape);
  EXPECT_EQ(back.data, t.data);
}

TEST(NNEFTensorIO, RankTooLargeThrows) {
  nnef::NNEFTensor t;
  t.item_type = nnef::kItemTypeFloat;
  t.bits = 32;
  t.shape = std::vector<int64_t>(9, 1);
  t.data.assign(4, 0);
  EXPECT_THROW(nnef::WriteNNEFTensor("/dev/null", t), std::invalid_argument);
}

TEST(NNEFExporter, SupportedOpsContainsCommon) {
  auto ops = nnef::SupportedOps();
  std::set<std::string> s(ops.begin(), ops.end());
  EXPECT_TRUE(s.count("Conv"));
  EXPECT_TRUE(s.count("Relu"));
  EXPECT_TRUE(s.count("MatMul"));
  EXPECT_TRUE(s.count("Gemm"));
}

TEST(NNEFExporter, IdentityModelText) {
  ModelProto m = MakeIdentityModel();
  std::string text = nnef::ToNNEFText(m, "");
  EXPECT_NE(text.find("version 1.0;"), std::string::npos);
  EXPECT_NE(text.find("graph"), std::string::npos);
  EXPECT_NE(text.find("Y = copy(X);"), std::string::npos);
}

TEST(NNEFExporter, UnknownOpThrows) {
  ModelProto m;
  GraphProto &g = *m.mutable_graph();
  g.set_name("g");
  *g.add_node() = MakeNode("DefinitelyNotAnOp", {"X"}, {"Y"});
  g.add_input()->set_name("X");
  g.add_output()->set_name("Y");
  EXPECT_THROW(nnef::ToNNEFText(m, ""), nnef::NNEFExportError);
}

TEST(NNEFExporter, CustomConverterAndUnregister) {
  ModelProto m;
  GraphProto &g = *m.mutable_graph();
  g.set_name("g");
  *g.add_node() = MakeNode("MyCustomOp", {"X"}, {"Y"});
  g.add_input()->set_name("X");
  g.add_output()->set_name("Y");

  nnef::RegisterOpConverter("MyCustomOp", [](nnef::ExportContext &ctx, const NodeProto &,
                                             const std::map<std::string, nnef::AttributeValue> &,
                                             const std::vector<std::string> &inputs,
                                             const std::vector<std::string> &outputs) {
    ctx.AddStatement(outputs[0] + " = my_custom(" + inputs[0] + ");");
  });
  std::string text = nnef::ToNNEFText(m, "");
  EXPECT_NE(text.find("Y = my_custom(X);"), std::string::npos);
  EXPECT_TRUE(nnef::UnregisterOpConverter("MyCustomOp"));
  EXPECT_FALSE(nnef::HasOpConverter("MyCustomOp"));
}

TEST(NNEFExporter, SaveNNEFDirectoryLayout) {
  ModelProto m = MakeIdentityModel();
  std::string dir = MakeTmpDir("nnef_save_");
  fs::remove_all(dir);
  nnef::SaveNNEF(m, dir, "", true);
  EXPECT_TRUE(fs::exists(fs::path(dir) / "graph.nnef"));
  fs::remove_all(dir);
}

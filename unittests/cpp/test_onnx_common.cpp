#include "../common/array_ref.h"
#include "../common/assertions.h"
#include "../common/common.h"
#include "../common/constants.h"
#include "../common/file_utils.h"
#include "../common/path.h"
#include "../common/platform_helpers.h"
#include "../common/proto_util.h"
#include "../common/scoped_resource.h"
#include "../common/status.h"
#include "../common/tensor.h"
#include "onnx.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <type_traits>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

int &CloseCounter() {
  static int value = 0;
  return value;
}

void CountClose(int value) { CloseCounter() += value; }

} // namespace

TEST(onnx_common, NamespaceAliases) {
  EXPECT_TRUE(
      (std::is_same<ONNX_LIGHT_NAMESPACE::assert_error, ONNX_NAMESPACE::assert_error>::value));
  EXPECT_TRUE(
      (std::is_same<ONNX_LIGHT_NAMESPACE::Common::Status, ONNX_NAMESPACE::Common::Status>::value));
}

TEST(onnx_common, ArrayRefBasics) {
  const std::array<int, 2> expected_slice{2, 3};
  std::vector<int> values{1, 2, 3, 4};
  ONNX_LIGHT_NAMESPACE::ArrayRef<int> ref(values);
  EXPECT_EQ(ref.size(), 4);
  EXPECT_EQ(ref.front(), 1);
  EXPECT_EQ(ref.back(), 4);
  EXPECT_TRUE(ref.slice(1, 2).equals(ONNX_LIGHT_NAMESPACE::ArrayRef<int>(expected_slice)));
  EXPECT_EQ(ref.vec(), values);
}

TEST(onnx_common, CommonThrowAndConstants) {
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::NormalizeDomain(ONNX_LIGHT_NAMESPACE::AI_ONNX_DOMAIN),
            std::string(ONNX_LIGHT_NAMESPACE::ONNX_DOMAIN));
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::NormalizeDomain("custom.domain"), "custom.domain");
  EXPECT_TRUE(ONNX_LIGHT_NAMESPACE::IsOnnxDomain(ONNX_LIGHT_NAMESPACE::ONNX_DOMAIN));
  EXPECT_TRUE(ONNX_LIGHT_NAMESPACE::IsOnnxDomain(ONNX_LIGHT_NAMESPACE::AI_ONNX_DOMAIN));
  EXPECT_THROW(ONNX_THROW("common-", 4), std::runtime_error);
}

TEST(onnx_common, AssertionsAndStatus) {
  EXPECT_THROW(ONNX_ASSERT(false), ONNX_LIGHT_NAMESPACE::assert_error);

  try {
    ONNX_ASSERTM(false, "value=%d", 5);
    FAIL() << "Expected assert_error";
  } catch (const ONNX_LIGHT_NAMESPACE::assert_error &e) {
    EXPECT_NE(std::string(e.what()).find("value=5"), std::string::npos);
  }

  EXPECT_THROW(TENSOR_ASSERTM(false, "tensor"), ONNX_LIGHT_NAMESPACE::tensor_error);

  const auto &ok = ONNX_LIGHT_NAMESPACE::Common::Status::OK();
  EXPECT_TRUE(ok.IsOK());
  EXPECT_EQ(ok.ToString(), "OK");

  ONNX_LIGHT_NAMESPACE::Common::Status status(
      ONNX_LIGHT_NAMESPACE::Common::StatusCategory::CHECKER,
      ONNX_LIGHT_NAMESPACE::Common::StatusCode::INVALID_ARGUMENT, "bad arg");
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.Category(), ONNX_LIGHT_NAMESPACE::Common::StatusCategory::CHECKER);
  EXPECT_EQ(status.Code(), ONNX_LIGHT_NAMESPACE::Common::StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.ErrorMessage(), "bad arg");
  EXPECT_NE(status.ToString().find("[CheckerError]"), std::string::npos);
  EXPECT_NE(status.ToString().find("INVALID_ARGUMENT"), std::string::npos);
}

TEST(onnx_common, PathScopedResourceAndEndianHelpers) {
  namespace fs = std::filesystem;

  const fs::path path = ONNX_LIGHT_NAMESPACE::utf8_to_path("folder/model.onnx");
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::path_to_utf8(path), path.string());

  const std::uint32_t probe = 1;
  const bool expected_little_endian =
      reinterpret_cast<const std::uint8_t *>(&probe)[0] == static_cast<std::uint8_t>(1);
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::is_processor_little_endian(), expected_little_endian);

  CloseCounter() = 0;
  {
    ONNX_LIGHT_NAMESPACE::ScopedResource<-1, CountClose> scoped(3);
    EXPECT_EQ(scoped.get(), 3);
    EXPECT_EQ(scoped.release(), 3);
    EXPECT_EQ(CloseCounter(), 0);
  }
  EXPECT_EQ(CloseCounter(), 0);
  {
    ONNX_LIGHT_NAMESPACE::ScopedResource<-1, CountClose> scoped(4);
    EXPECT_EQ(scoped.get(), 4);
  }
  EXPECT_EQ(CloseCounter(), 4);

  int scope_exit_counter = 0;
  {
    ONNX_LIGHT_NAMESPACE::ScopeExit on_exit(
        [&scope_exit_counter]() noexcept { ++scope_exit_counter; });
    EXPECT_EQ(scope_exit_counter, 0);
  }
  EXPECT_EQ(scope_exit_counter, 1);
}

TEST(onnx_common, ProtoUtilAndFileUtils) {
  namespace fs = std::filesystem;

  FunctionProto function;
  function.set_domain(ONNX_LIGHT_NAMESPACE::AI_ONNX_DOMAIN);
  function.set_name("Add");
  function.set_overload("float");
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::GetFunctionImplId(function), "::Add::float");

  NodeProto node;
  node.set_domain("custom");
  node.set_op_type("Do");
  EXPECT_EQ(ONNX_LIGHT_NAMESPACE::GetCalleeId(node), "custom::Do");

  ModelProto model;
  model.add_graph().set_name("loaded");
  std::string serialized;
  model.SerializeToString(serialized);

  const fs::path temp_dir = fs::temp_directory_path();
  const fs::path file_path = temp_dir / "onnx_light_common_file_utils.onnx";
  {
    std::ofstream out(file_path, std::ios::binary);
    out.write(serialized.data(), static_cast<std::streamsize>(serialized.size()));
  }

  ModelProto loaded;
  ONNX_LIGHT_NAMESPACE::LoadProtoFromPath(file_path.string(), loaded);
  EXPECT_TRUE(loaded.has_graph());
  EXPECT_EQ(loaded.ref_graph().ref_name(), "loaded");

  fs::remove(file_path);
}

TEST(onnx_common, TensorHelpers) {
  ONNX_LIGHT_NAMESPACE::Tensor tensor;
  tensor.sizes() = {2, 3, 4};
  tensor.elem_type() = static_cast<int32_t>(ONNX_LIGHT_NAMESPACE::TensorProto::DataType::INT64);
  tensor.int64s() = {1, 2, 3, 4, 5, 6};
  tensor.setName("weights");
  tensor.set_segment_begin_and_end(1, 7);
  tensor.external_data().push_back({"location", "weights.bin"});

  EXPECT_EQ(tensor.elem_num(), 24);
  EXPECT_EQ(tensor.size_from_dim(1), 12);
  EXPECT_EQ(tensor.size_from_dim(-1), 4);
  EXPECT_TRUE(tensor.hasName());
  EXPECT_EQ(tensor.name(), "weights");
  EXPECT_TRUE(tensor.is_segment());
  EXPECT_EQ(tensor.segment_begin(), 1);
  EXPECT_EQ(tensor.segment_end(), 7);
  EXPECT_EQ(tensor.data<int64_t>()[0], 1);
  EXPECT_EQ(tensor.external_data().front().first, "location");

  float raw_value = 3.5f;
  tensor.set_raw_data(std::string(reinterpret_cast<const char *>(&raw_value), sizeof(raw_value)));
  EXPECT_TRUE(tensor.is_raw_data());
  EXPECT_FLOAT_EQ(*tensor.data<float>(), 3.5f);

  ONNX_LIGHT_NAMESPACE::Tensor string_tensor;
  string_tensor.strings().push_back("abc");
  EXPECT_EQ(string_tensor.data<std::string>()[0], "abc");
  string_tensor.set_raw_data("x");
  EXPECT_THROW(static_cast<void>(string_tensor.data<std::string>()),
               ONNX_LIGHT_NAMESPACE::assert_error);
}

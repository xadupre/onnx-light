#include "onnx.h"
#include "onnx_crypt.h"
#include "onnx_helper.h"
#include <filesystem>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#ifdef ONNX_LIGHT_HAS_OPENSSL

using namespace ONNX_LIGHT_NAMESPACE;

namespace {

// Builds a minimal ModelProto with one Add node and an initializer.
ModelProto make_test_model() {
  ModelProto model;
  model.set_ir_version(9);
  OperatorSetIdProto *opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(18);

  GraphProto *graph = model.add_graph();
  graph->set_name("test_graph");

  // Add a weight initializer with raw data.
  TensorProto *weight = graph->add_initializer();
  weight->set_name("W");
  weight->set_data_type(TensorProto::DataType::FLOAT);
  weight->ref_dims().push_back(4);
  // 4 floats = 16 bytes raw data.
  const float vals[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  weight->ref_raw_data().resize(16);
  std::memcpy(weight->ref_raw_data().data(), vals, 16);

  NodeProto *node = graph->add_node();
  node->set_op_type("Relu");
  *node->add_input() = "W";
  *node->add_output() = "Y";

  return model;
}

} // namespace

TEST(onnx_crypt, RoundTrip_SimpleModel) {
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path = (tmp / "test_onnx_crypt_roundtrip.onnxc").string();
  const std::string pass = "my_secret_password";

  ModelProto original = make_test_model();
  SerializeOptions sopts;
  SaveEncryptedModel(original, path, pass, sopts);

  ASSERT_TRUE(std::filesystem::exists(path));
  EXPECT_GT(std::filesystem::file_size(path), 0u);

  ModelProto loaded;
  ParseOptions popts;
  LoadEncryptedModel(loaded, path, pass, popts);

  // Compare serialized forms.
  std::string s_orig, s_loaded;
  original.SerializeToString(s_orig, sopts);
  loaded.SerializeToString(s_loaded, sopts);
  EXPECT_EQ(s_orig, s_loaded);

  std::filesystem::remove(path);
}

TEST(onnx_crypt, WrongKey_ThrowsOnDecrypt) {
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path = (tmp / "test_onnx_crypt_wrongkey.onnxc").string();

  ModelProto original = make_test_model();
  SaveEncryptedModel(original, path, "correct_key");

  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "wrong_key"), std::runtime_error);

  std::filesystem::remove(path);
}

TEST(onnx_crypt, BadMagic_ThrowsOnLoad) {
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path = (tmp / "test_onnx_crypt_badmagic.onnxc").string();

  // Write garbage bytes.
  {
    std::ofstream ofs(path, std::ios::binary);
    ofs << "BADMAGIC1234567890";
  }

  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "any_key"), std::runtime_error);

  std::filesystem::remove(path);
}

TEST(onnx_crypt, DifferentKeys_ProduceDifferentCiphertext) {
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path1 = (tmp / "test_onnx_crypt_key1.onnxc").string();
  const std::string path2 = (tmp / "test_onnx_crypt_key2.onnxc").string();

  ModelProto model = make_test_model();
  SaveEncryptedModel(model, path1, "key_alpha");
  SaveEncryptedModel(model, path2, "key_beta");

  // Files must differ (different random salt/IV + different ciphertext).
  const auto size1 = std::filesystem::file_size(path1);
  const auto size2 = std::filesystem::file_size(path2);
  EXPECT_EQ(size1, size2); // same plaintext → same ciphertext length

  std::string c1, c2;
  {
    std::ifstream f1(path1, std::ios::binary);
    c1.assign(std::istreambuf_iterator<char>(f1), {});
  }
  {
    std::ifstream f2(path2, std::ios::binary);
    c2.assign(std::istreambuf_iterator<char>(f2), {});
  }
  EXPECT_NE(c1, c2);

  std::filesystem::remove(path1);
  std::filesystem::remove(path2);
}

TEST(onnx_crypt, EmptyModel_RoundTrip) {
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path = (tmp / "test_onnx_crypt_empty.onnxc").string();
  const std::string pass = "pw";

  ModelProto model;
  SaveEncryptedModel(model, path, pass);

  ModelProto loaded;
  LoadEncryptedModel(loaded, path, pass);

  SerializeOptions sopts;
  std::string s1, s2;
  model.SerializeToString(s1, sopts);
  loaded.SerializeToString(s2, sopts);
  EXPECT_EQ(s1, s2);

  std::filesystem::remove(path);
}

TEST(onnx_crypt, StringRoundTrip_SimpleModel) {
  const std::string pass = "string_roundtrip_pass";

  ModelProto original = make_test_model();
  const std::string blob = SaveEncryptedModelToString(original, pass);
  EXPECT_GT(blob.size(), 40u); // must be at least header size

  ModelProto loaded;
  LoadEncryptedModelFromString(loaded, blob, pass);

  SerializeOptions sopts;
  std::string s_orig, s_loaded;
  original.SerializeToString(s_orig, sopts);
  loaded.SerializeToString(s_loaded, sopts);
  EXPECT_EQ(s_orig, s_loaded);
}

TEST(onnx_crypt, StringRoundTrip_WrongKey_Throws) {
  ModelProto original = make_test_model();
  const std::string blob = SaveEncryptedModelToString(original, "correct_key");

  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModelFromString(loaded, blob, "wrong_key"), std::runtime_error);
}

TEST(onnx_crypt, StringRoundTrip_BadMagic_Throws) {
  ModelProto loaded;
  const std::string garbage = "BADMAGIC1234567890";
  EXPECT_THROW(LoadEncryptedModelFromString(loaded, garbage, "any_key"), std::runtime_error);
}

TEST(onnx_crypt, StringAndFileBlobs_AreCompatible) {
  // A blob produced by SaveEncryptedModelToString can be loaded by LoadEncryptedModel
  // if written to a file, and vice versa.
  const std::filesystem::path tmp = std::filesystem::temp_directory_path();
  const std::string path = (tmp / "test_onnx_crypt_compat.onnxc").string();
  const std::string pass = "compat_key";

  ModelProto original = make_test_model();

  // Save to file, load from string.
  SaveEncryptedModel(original, path, pass);
  std::string file_blob;
  {
    std::ifstream ifs(path, std::ios::binary);
    file_blob.assign(std::istreambuf_iterator<char>(ifs), {});
  }
  ModelProto loaded_from_str;
  LoadEncryptedModelFromString(loaded_from_str, file_blob, pass);

  SerializeOptions sopts;
  std::string s_orig, s_from_str;
  original.SerializeToString(s_orig, sopts);
  loaded_from_str.SerializeToString(s_from_str, sopts);
  EXPECT_EQ(s_orig, s_from_str);

  std::filesystem::remove(path);
}

#else // ONNX_LIGHT_HAS_OPENSSL

// Placeholder so the test binary still compiles without OpenSSL.
TEST(onnx_crypt, OpenSSL_not_available) {
  GTEST_SKIP() << "OpenSSL not available – onnx_crypt tests skipped.";
}

#endif // ONNX_LIGHT_HAS_OPENSSL

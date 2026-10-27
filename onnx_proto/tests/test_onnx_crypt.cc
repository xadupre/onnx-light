#include "onnx.h"
#include "onnx_crypt.h"
#include "onnx_helper.h"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#endif

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

ModelProto make_callback_test_model() {
  ModelProto model = make_test_model();
  auto &graph = model.ref_graph();
  graph.set_name("parent");
  auto *node = graph.add_node();
  node->set_op_type("Constant");
  node->set_name("constant");
  auto *value = node->add_attribute();
  value->set_name("value");
  value->set_type(AttributeProto::TENSOR);
  *value->add_t() = graph.initializer(0);
  value->ref_t().set_name("value");
  auto *branches = graph.ref_node()[0].add_attribute();
  branches->set_name("branches");
  branches->set_type(AttributeProto::GRAPHS);
  for (const auto &name : {"first", "second"}) {
    auto *child = branches->add_graphs();
    child->set_name(name);
    *child->add_node() = *node;
  }
  auto *function = model.add_functions();
  function->set_name("function");
  auto *body = function->add_attribute_proto();
  body->set_type(AttributeProto::GRAPH);
  body->add_g()->set_name("default");
  *body->ref_g().add_node() = *node;
  return model;
}

ModelProto make_sparse_test_model() {
  ModelProto model = make_test_model();
  size_t index = 0;
  const auto populate = [&](SparseTensorProto *sparse) {
    auto &values = sparse->ref_values();
    values.set_name("sparse_" + std::to_string(index++));
    values.set_data_type(TensorProto::DataType::FLOAT);
    values.ref_dims().push_back(1);
    const float value = 2;
    values.ref_raw_data().resize(sizeof(value));
    std::memcpy(values.ref_raw_data().data(), &value, sizeof(value));
    auto &indices = sparse->ref_indices();
    indices.set_data_type(TensorProto::DataType::INT64);
    indices.ref_dims().push_back(1);
    const int64_t position = 0;
    indices.ref_raw_data().resize(sizeof(position));
    std::memcpy(indices.ref_raw_data().data(), &position, sizeof(position));
    sparse->ref_dims().push_back(4);
  };
  populate(model.ref_graph().add_sparse_initializer());
  auto &node = model.ref_graph().ref_node()[0];
  auto *attribute = node.add_attribute();
  attribute->set_name("sparse");
  attribute->set_type(AttributeProto::AttributeType::SPARSE_TENSOR);
  populate(attribute->add_sparse_tensor());
  attribute = node.add_attribute();
  attribute->set_name("sparse_list");
  attribute->set_type(AttributeProto::AttributeType::SPARSE_TENSORS);
  populate(attribute->add_sparse_tensors());
  populate(attribute->add_sparse_tensors());
  attribute = node.add_attribute();
  attribute->set_name("body");
  attribute->set_type(AttributeProto::AttributeType::GRAPH);
  populate(attribute->add_g()->add_sparse_initializer());
  auto *function = model.add_functions();
  function->set_name("SparseDefault");
  attribute = function->add_attribute_proto();
  attribute->set_name("default_sparse");
  attribute->set_type(AttributeProto::AttributeType::SPARSE_TENSOR);
  populate(attribute->add_sparse_tensor());
  return model;
}

FunctionProto *add_struct_codec(ModelProto &model, size_t scope, bool encoder) {
  StructTypeProto *type = nullptr;
  switch (scope) {
  case 0:
    type = model.add_struct_types();
    type->set_type_id(1);
    break;
  case 1:
    type = model.ref_graph()
               .add_extended_initializer()
               ->mutable_encoded_value()
               ->mutable_struct_type();
    break;
  case 2:
    type = model.ref_graph().add_input()->mutable_type()->mutable_struct_type();
    break;
  case 3:
    type = model.ref_graph().add_output()->mutable_type()->mutable_struct_type();
    break;
  case 4:
    type = model.ref_graph().add_value_info()->mutable_type()->mutable_struct_type();
    break;
  case 5:
    type = model.add_functions()->add_value_info()->mutable_type()->mutable_struct_type();
    break;
  case 6:
    type = model.ref_graph().ref_node()[0].add_attribute()->mutable_tp()->mutable_struct_type();
    break;
  default:
    type =
        model.ref_graph().ref_node()[0].add_attribute()->add_type_protos()->mutable_struct_type();
    break;
  }
  // Exercises nested inline declarations and container types in every scope.
  type = type->mutable_array()
             ->mutable_element_type()
             ->mutable_sequence_type()
             ->mutable_elem_type()
             ->mutable_optional_type()
             ->mutable_elem_type()
             ->mutable_map_type()
             ->mutable_value_type()
             ->mutable_struct_type()
             ->mutable_structure()
             ->add_field()
             ->mutable_type()
             ->mutable_struct_type();
  return encoder ? type->mutable_encoder() : type->mutable_decoder();
}

class EncryptedExternalTest : public ::testing::Test {
protected:
  std::filesystem::path directory;
  std::string path;

  void SetUp() override {
    directory = std::filesystem::temp_directory_path() /
                ("onnx_crypt_external_" +
                 std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ASSERT_TRUE(std::filesystem::create_directory(directory));
    path = (directory / "model.onnxc").string();
  }

  void TearDown() override { std::filesystem::remove_all(directory); }

  std::string read(const std::filesystem::path &file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.is_open())
      throw std::runtime_error("Cannot open test file: " + file.string());
    const std::string bytes(std::istreambuf_iterator<char>(stream), {});
    if (stream.bad())
      throw std::runtime_error("Cannot read test file: " + file.string());
    return bytes;
  }

  void write(const std::filesystem::path &file, const std::string &bytes) {
    std::ofstream stream(file, std::ios::binary | std::ios::trunc);
    stream.write(bytes.data(), bytes.size());
    ASSERT_TRUE(stream.good());
  }

  ModelProto metadata() {
    ModelProto model;
    LoadEncryptedModelFromString(model, read(path), "secret");
    return model;
  }

  std::string entry(TensorProto &tensor, const std::string &key) {
    for (const auto &item : tensor.external_data())
      if (item.key() == key)
        return std::string(item.value());
    return "";
  }

  void set_entry(TensorProto &tensor, const std::string &key, const std::string &value) {
    for (auto &item : tensor.ref_external_data()) {
      if (item.key() == key) {
        item.set_value(value);
        return;
      }
    }
    auto *item = tensor.add_external_data();
    item->set_key(key);
    item->set_value(value);
  }

  void expect_same(ModelProto &expected, ModelProto &actual) {
    ModelProto normalized_expected = expected, normalized_actual = actual;
    for (ModelProto *model : {&normalized_expected, &normalized_actual}) {
      IteratorTensorProto iterator(model);
      while (iterator.next()) {
        if (iterator->data_location() == TensorProto::DataLocation::DEFAULT)
          iterator->set_data_location(TensorProto::DataLocation::DEFAULT);
      }
    }
    std::string left, right;
    normalized_expected.SerializeToString(left);
    normalized_actual.SerializeToString(right);
    EXPECT_EQ(left, right);
  }
};

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

TEST(onnx_crypt, StringRoundTrip_ChaCha20Poly1305) {
  const std::string pass = "string_roundtrip_pass";
  ModelProto original = make_test_model();
  const std::string blob =
      SaveEncryptedModelToString(original, pass, SerializeOptions{}, "ChaCha20-Poly1305");
  EXPECT_GT(blob.size(), 52u);
  EXPECT_EQ(blob.substr(0, 8), "ONNXCRY2");

  ModelProto loaded;
  LoadEncryptedModelFromString(loaded, blob, pass);

  SerializeOptions sopts;
  std::string s_orig, s_loaded;
  original.SerializeToString(s_orig, sopts);
  loaded.SerializeToString(s_loaded, sopts);
  EXPECT_EQ(s_orig, s_loaded);
}

TEST(onnx_crypt, StringRoundTrip_ChaCha20Poly1305WrongKey_Throws) {
  ModelProto original = make_test_model();
  const std::string blob =
      SaveEncryptedModelToString(original, "correct_key", SerializeOptions{}, "ChaCha20-Poly1305");

  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModelFromString(loaded, blob, "wrong_key"), std::runtime_error);
}

TEST(onnx_crypt, AES256GCM_FileAndStringRoundTrip) {
  const std::string path =
      (std::filesystem::temp_directory_path() / "test_onnx_crypt_gcm.onnxc").string();
  ModelProto original = make_test_model();
  SaveEncryptedModel(original, path, "secret", "AES-256-GCM");
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  std::filesystem::remove(path);
  std::string expected, actual;
  original.SerializeToString(expected);
  loaded.SerializeToString(actual);
  EXPECT_EQ(expected, actual);

  const std::string blob = SaveEncryptedModelToString(original, "secret", "AES-256-GCM");
  EXPECT_EQ(blob.substr(0, 8), "ONNXCRY3");
  EXPECT_EQ(blob.size(), 52u + expected.size());
  ModelProto loaded_from_string;
  LoadEncryptedModelFromString(loaded_from_string, blob, "secret");
  loaded_from_string.SerializeToString(actual);
  EXPECT_EQ(expected, actual);
}

TEST(onnx_crypt, AES256GCM_RejectsTamperingAndTruncation) {
  ModelProto model = make_test_model();
  const std::string blob = SaveEncryptedModelToString(model, "secret", "AES-256-GCM");
  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModelFromString(loaded, blob, "wrong"), std::runtime_error);
  for (size_t offset : {size_t{0}, size_t{8}, size_t{24}, size_t{36}, size_t{52}}) {
    std::string tampered = blob;
    tampered[offset] ^= 1;
    try {
      LoadEncryptedModelFromString(loaded, tampered, "secret");
      FAIL() << "Tampering at offset " << offset << " was accepted";
    } catch (const std::runtime_error &ex) {
      EXPECT_EQ(std::string(ex.what()).find("Failed to parse"), std::string::npos);
    }
  }
  for (size_t length : {size_t{0}, size_t{7}, size_t{8}, size_t{36}, size_t{51}, size_t{52}}) {
    EXPECT_THROW(LoadEncryptedModelFromString(loaded, blob.substr(0, length), "secret"),
                 std::runtime_error);
  }
  ModelProto empty;
  const std::string empty_blob = SaveEncryptedModelToString(empty, "secret", "AES-256-GCM");
  EXPECT_EQ(empty_blob.size(), 52u);
  LoadEncryptedModelFromString(loaded, empty_blob, "secret");
}

TEST_F(EncryptedExternalTest, PacksAndSplitsWithoutChangingInput) {
  ModelProto model = make_test_model();
  TensorProto second = model.graph().initializer()[0];
  second.set_name("W2");
  *model.ref_graph().add_initializer() = second;
  second.set_name("W3");
  *model.ref_graph().add_initializer() = second;
  std::string before, after;
  model.SerializeToString(before);
  SaveEncryptedModelWithExternalData(model, path, "secret", 84, 0);
  model.SerializeToString(after);
  EXPECT_EQ(before, after);
  ModelProto stored = metadata();
  auto &weights = stored.ref_graph().ref_initializer();
  EXPECT_EQ(entry(weights[0], "location"), entry(weights[1], "location"));
  EXPECT_NE(entry(weights[0], "location"), entry(weights[2], "location"));
  EXPECT_EQ(entry(weights[0], "offset"), "0");
  EXPECT_EQ(entry(weights[1], "offset"), "16");
  EXPECT_EQ(entry(weights[2], "offset"), "0");
  for (const auto &file : std::filesystem::directory_iterator(directory)) {
    const std::string bytes = read(file.path());
    EXPECT_EQ(bytes.substr(0, 8), "ONNXCRY3");
    if (file.path() != std::filesystem::path(path)) {
      EXPECT_LE(bytes.size(), 84u);
    }
  }
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(model, loaded);
  EXPECT_EQ(std::distance(std::filesystem::directory_iterator(directory),
                          std::filesystem::directory_iterator{}),
            3);
}

TEST_F(EncryptedExternalTest, RoundTripsNonAsciiModelBasename) {
  path = (directory / std::filesystem::path(u8"mod\u00e8le.onnxc")).string();
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata();
  const auto location = entry(stored.ref_graph().ref_initializer()[0], "location");
  EXPECT_TRUE(std::filesystem::exists(
      directory / std::filesystem::path(std::u8string(location.begin(), location.end()))));
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(model, loaded);
}

TEST_F(EncryptedExternalTest, ExportsBorrowedRawDataWithoutMutatingInput) {
  ModelProto original = make_test_model();
  std::string serialized;
  original.SerializeToString(serialized);
  ParseOptions options;
  options.no_copy = true;
  ModelProto borrowed;
  ASSERT_TRUE(borrowed.ParseFromString(serialized, options));
  ASSERT_TRUE(borrowed.graph().initializer()[0].raw_data().is_borrowed());
  SaveEncryptedModelWithExternalData(borrowed, path, "secret", 68, 0);
  EXPECT_TRUE(borrowed.graph().initializer()[0].raw_data().is_borrowed());
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(original, loaded);
}

TEST_F(EncryptedExternalTest, ExportsShardsWithoutDuplicatingOwnedPayloads) {
#if defined(__linux__)
  ModelProto model = make_test_model();
  constexpr size_t bytes = 1024 * 1024;
  for (size_t i = 0; i < 32; ++i) {
    auto *tensor = model.ref_graph().add_initializer();
    tensor->set_name("weight_" + std::to_string(i));
    tensor->set_data_type(TensorProto::UINT8);
    tensor->add_dims(bytes);
    tensor->ref_raw_data().resize(bytes);
    std::memset(tensor->ref_raw_data().data(), static_cast<int>(i), bytes);
  }
  const auto *original = model.graph().initializer()[1].raw_data().data();
  struct rlimit previous;
  ASSERT_EQ(getrlimit(RLIMIT_AS, &previous), 0);
  std::ifstream statm("/proc/self/statm");
  size_t pages = 0;
  ASSERT_TRUE(static_cast<bool>(statm >> pages));
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  struct rlimit limited = previous;
  limited.rlim_cur = pages * static_cast<size_t>(page_size) + 16 * bytes;
  ASSERT_EQ(setrlimit(RLIMIT_AS, &limited), 0);
  EXPECT_NO_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", bytes + 52, 0));
  ASSERT_EQ(setrlimit(RLIMIT_AS, &previous), 0);
  EXPECT_EQ(model.graph().initializer()[1].raw_data().data(), original);
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  ASSERT_EQ(loaded.graph().initializer().size(), model.graph().initializer().size());
  for (size_t i = 0; i < model.graph().initializer().size(); ++i) {
    const auto &expected = model.graph().initializer()[i];
    EXPECT_FALSE(expected.has_external_data());
    EXPECT_EQ(loaded.graph().initializer()[i].raw_data(), expected.raw_data());
  }
#else
  GTEST_SKIP() << "Requires Linux address-space accounting.";
#endif
}

TEST_F(EncryptedExternalTest, MetadataErrorsIdentifyTensorLocationAndField) {
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  for (const std::string field :
       {"offset", "length", "onnx_light.encryption", "onnx_light.encrypted_sha256", "location",
        "duplicate", "missing", "range"}) {
    SCOPED_TRACE(field);
    ModelProto stored = metadata();
    auto &tensor = stored.ref_graph().ref_initializer()[0];
    std::string expected_field = field;
    if (field == "duplicate") {
      auto *duplicate = tensor.add_external_data();
      duplicate->set_key("offset");
      duplicate->set_value("0");
      expected_field = "offset";
    } else if (field == "missing") {
      auto &entries = tensor.ref_external_data();
      for (auto it = entries.begin(); it != entries.end();)
        if (it->key() == "length")
          it = entries.erase(it);
        else
          ++it;
      expected_field = "length";
    } else if (field == "range") {
      set_entry(tensor, "offset", "1");
      expected_field = "offset/length";
    } else {
      set_entry(tensor, field, field == "location" ? "../unsafe" : "invalid");
    }
    const auto location = entry(tensor, "location");
    for (bool skip : {false, true}) {
      ParseOptions options;
      options.skip_raw_data = skip;
      options.raw_data_threshold = 0;
      try {
        LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options);
        FAIL() << "Invalid metadata was accepted";
      } catch (const std::runtime_error &error) {
        const std::string message = error.what();
        EXPECT_NE(message.find("tensor 'W'"), std::string::npos);
        EXPECT_NE(message.find(location), std::string::npos);
        EXPECT_NE(message.find(expected_field), std::string::npos);
      }
    }
  }
}

TEST_F(EncryptedExternalTest, LoadsInterleavedShardReferencesAndRejectsConflictingDigests) {
  ModelProto model = make_test_model();
  for (const auto &name : {"W2", "W3"}) {
    TensorProto tensor = model.graph().initializer()[0];
    tensor.set_name(name);
    *model.ref_graph().add_initializer() = tensor;
  }
  SaveEncryptedModelWithExternalData(model, path, "secret", 84, 0);
  ModelProto stored = metadata();
  auto &initializers = stored.ref_graph().ref_initializer();
  initializers.SwapElements(1, 2);
  model.ref_graph().ref_initializer().SwapElements(1, 2);
  ASSERT_EQ(entry(initializers[0], "location"), entry(initializers[2], "location"));
  ASSERT_NE(entry(initializers[0], "location"), entry(initializers[1], "location"));
  LoadEncryptedExternalDataForModel(stored, directory.string(), "secret");
  expect_same(model, stored);

  stored = metadata();
  set_entry(stored.ref_graph().ref_initializer()[1], "onnx_light.encrypted_sha256",
            std::string(64, '0'));
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret"),
               std::runtime_error);
  EXPECT_FALSE(stored.graph().initializer()[0].has_raw_data());
}

TEST_F(EncryptedExternalTest, EncryptsSparseValuesAndIndicesAcrossScopes) {
  ModelProto model = make_sparse_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata();
  IteratorTensorProto iterator(&stored);
  size_t count = 0;
  while (iterator.next()) {
    EXPECT_FALSE(iterator->has_raw_data());
    EXPECT_EQ(entry(*iterator, "onnx_light.encryption"), "ONNXCRY3");
    ++count;
  }
  ASSERT_EQ(count, 13u);
  EXPECT_THROW(LoadExternalDataForModel(stored, directory.string()), std::runtime_error);
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(model, loaded);
}

TEST_F(EncryptedExternalTest, ValidatesEverySparsePayloadBeforeWriting) {
  for (size_t index = 0; index < 13; ++index) {
    for (bool unloaded : {false, true}) {
      ModelProto model = make_sparse_test_model();
      IteratorTensorProto iterator(&model);
      for (size_t i = 0; i <= index; ++i)
        ASSERT_TRUE(iterator.next());
      if (unloaded) {
        iterator->clear_raw_data();
        iterator->set_data_location(TensorProto::DataLocation::EXTERNAL);
        set_entry(*iterator, "location", "missing.weights");
      } else {
        iterator->ref_raw_data().resize(17);
      }
      EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0),
                   std::runtime_error);
      EXPECT_TRUE(std::filesystem::is_empty(directory));
    }
  }
}

TEST_F(EncryptedExternalTest, RejectsExternalEncodedInitializersBeforeWriting) {
  for (bool nested : {false, true}) {
    ModelProto model = make_test_model();
    GraphProto *graph = &model.ref_graph();
    if (nested) {
      auto *attribute = model.ref_graph().ref_node()[0].add_attribute();
      attribute->set_name("body");
      attribute->set_type(AttributeProto::AttributeType::GRAPH);
      graph = attribute->add_g();
    }
    auto *value = graph->add_extended_initializer()->mutable_encoded_value();
    value->set_name("external_encoded");
    value->set_data_location(TensorProto::DataLocation::EXTERNAL);
    auto *location = value->add_external_data();
    location->set_key("location");
    location->set_value("weights.data");
    auto *length = value->add_external_data();
    length->set_key("length");
    length->set_value("1");

    EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0),
                 std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(directory));
  }
}

TEST_F(EncryptedExternalTest, RejectsExternalStructCodecTensorsBeforeWriting) {
  for (size_t scope = 0; scope < 8; ++scope) {
    for (bool encoder : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "scope=" << scope << ", encoder=" << encoder);
      for (bool loaded : {false, true}) {
        ModelProto model = make_test_model();
        auto *codec = add_struct_codec(model, scope, encoder);
        auto *attribute = codec->add_node()->add_attribute();
        attribute->set_type(AttributeProto::TENSOR);
        auto *tensor = attribute->mutable_t();
        if (loaded)
          *tensor = model.graph().initializer()[0];
        tensor->set_data_location(TensorProto::EXTERNAL);
        set_entry(*tensor, "location", "plaintext.weights");
        EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0),
                     std::runtime_error);
        EXPECT_TRUE(std::filesystem::is_empty(directory));
      }
    }
  }
}

TEST_F(EncryptedExternalTest, RejectsExternalStructCodecDefaultsAndNestedPayloadsBeforeWriting) {
  for (size_t scope = 0; scope < 6; ++scope) {
    ModelProto model = make_test_model();
    auto *codec = model.add_struct_types()->mutable_decoder();
    auto *attribute = codec->add_attribute_proto();
    if (scope == 0) {
      attribute->set_type(AttributeProto::SPARSE_TENSORS);
      auto *indices = attribute->add_sparse_tensors()->mutable_indices();
      indices->set_data_location(TensorProto::EXTERNAL);
      set_entry(*indices, "location", "plaintext.weights");
    } else {
      attribute->set_type(AttributeProto::GRAPHS);
      auto *graph = attribute->add_graphs();
      if (scope == 1) {
        auto *tensor = graph->add_initializer();
        tensor->set_data_location(TensorProto::EXTERNAL);
        set_entry(*tensor, "location", "plaintext.weights");
      } else if (scope == 2) {
        auto *value = graph->add_extended_initializer()->mutable_encoded_value();
        value->set_data_location(TensorProto::EXTERNAL);
      } else if (scope == 3) {
        auto *tensor = graph->add_sparse_initializer()->mutable_values();
        tensor->set_data_location(TensorProto::EXTERNAL);
        set_entry(*tensor, "location", "plaintext.weights");
      } else if (scope == 4) {
        auto *tensor =
            graph->add_extended_initializer()->mutable_paged_cache()->add_blocks()->mutable_key();
        tensor->set_data_location(TensorProto::EXTERNAL);
        set_entry(*tensor, "location", "plaintext.weights");
      } else {
        auto *tensor = graph->add_extended_initializer()
                           ->mutable_encoded_value()
                           ->mutable_affine()
                           ->mutable_scale();
        tensor->set_data_location(TensorProto::EXTERNAL);
        set_entry(*tensor, "location", "plaintext.weights");
      }
    }
    EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0),
                 std::runtime_error);
    EXPECT_TRUE(std::filesystem::is_empty(directory));
  }
}

TEST_F(EncryptedExternalTest, PreservesEmbeddedStructCodecTensors) {
  for (size_t scope = 0; scope < 8; ++scope) {
    for (bool encoder : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "scope=" << scope << " encoder=" << encoder);
      ModelProto model = make_test_model();
      auto *codec = add_struct_codec(model, scope, encoder);
      auto *attribute = codec->add_node()->add_attribute();
      attribute->set_type(AttributeProto::TENSOR);
      TensorProto tensor = model.graph().initializer()[0];
      tensor.ref_dims()[0] = 8;
      tensor.ref_raw_data().resize(32);
      std::memset(tensor.ref_raw_data().data(), 0x3c, 32);
      *attribute->mutable_t() = tensor;
      attribute = codec->add_attribute_proto();
      attribute->set_type(AttributeProto::GRAPH);
      *attribute->mutable_g()->add_initializer() = tensor;
      auto *declaration = model.add_struct_types();
      declaration->set_type_id(2);
      *declaration->mutable_structure()->add_field()->mutable_constant() = tensor;
      SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
      ModelProto stored = metadata();
      EXPECT_EQ(stored.graph().initializer()[0].data_location(), TensorProto::EXTERNAL);
      const auto &constant = stored.struct_types().back().structure().field()[0].constant();
      EXPECT_EQ(constant.raw_data(), tensor.raw_data());
      EXPECT_FALSE(constant.has_external_data());
      ModelProto loaded;
      LoadEncryptedModel(loaded, path, "secret");
      expect_same(model, loaded);
    }
  }
}

TEST_F(EncryptedExternalTest, HonorsAlignmentAndCallbacksForEmbeddedAndExternalTensors) {
  ModelProto model = make_test_model();
  TensorProto small = model.graph().initializer()[0];
  small.set_name("inline");
  small.ref_dims()[0] = 1;
  small.ref_raw_data().resize(4);
  *model.ref_graph().add_initializer() = small;
  auto *attribute = model.ref_graph().ref_node()[0].add_attribute();
  attribute->set_name("body");
  attribute->set_type(AttributeProto::AttributeType::GRAPH);
  *attribute->add_g()->add_initializer() = model.graph().initializer()[0];
  attribute->ref_g().ref_initializer()[0].set_name("nested");
  auto *function = model.add_functions();
  function->set_name("DefaultTensor");
  attribute = function->add_attribute_proto();
  attribute->set_name("default_value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  *attribute->add_t() = model.graph().initializer()[0];
  attribute->ref_t().set_name("default");
  SaveEncryptedModelWithExternalData(model, path, "secret", 84, 8);
  std::map<std::string, GraphProto *> graphs;
  std::map<std::string, int> callbacks, cleanups;
  ParseOptions options;
  options.alignment = 4096;
  options.raw_data_callback = [&](TensorProto &tensor, GraphProto *graph) {
    const std::string name(tensor.name());
    ++callbacks[name];
    graphs[name] = graph;
    EXPECT_EQ(reinterpret_cast<uintptr_t>(tensor.raw_data().data()) % options.alignment, 0u);
    return [&, name]() { ++cleanups[name]; };
  };
  {
    ModelProto loaded;
    LoadEncryptedModel(loaded, path, "secret", options);
    expect_same(model, loaded);
    EXPECT_EQ(graphs.at("W"), &loaded.ref_graph());
    EXPECT_EQ(graphs.at("inline"), &loaded.ref_graph());
    EXPECT_EQ(graphs.at("nested"), &loaded.ref_graph().ref_node()[0].ref_attribute()[0].ref_g());
    EXPECT_EQ(graphs.at("default"), nullptr);
    for (const auto &name : {"W", "inline", "nested", "default"}) {
      EXPECT_EQ(callbacks[name], 1);
      EXPECT_EQ(cleanups[name], 0);
    }
  }
  for (const auto &name : {"W", "inline", "nested", "default"})
    EXPECT_EQ(cleanups[name], 1);
  options.alignment = 3;
  ModelProto stored = metadata();
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
}

TEST_F(EncryptedExternalTest, ThresholdAndEmptyModelRemainEmbedded) {
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 53, 17);
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(model, loaded);
  EXPECT_FALSE(loaded.graph().initializer()[0].has_external_data());
  ModelProto empty;
  SaveEncryptedModelWithExternalData(empty, path, "secret", 53, 0);
  loaded = ModelProto{};
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(empty, loaded);
}

TEST_F(EncryptedExternalTest, EmbeddedAndExternalPayloadsHonorLoadingThreshold) {
  ModelProto model = make_test_model();
  model.ref_graph().clr_initializer();
  for (size_t length : {15, 16, 17}) {
    auto *tensor = model.ref_graph().add_initializer();
    tensor->set_name("W" + std::to_string(length));
    tensor->set_data_type(TensorProto::DataType::UINT8);
    tensor->add_dims(length);
    tensor->ref_raw_data().resize(length);
    std::memset(tensor->ref_raw_data().data(), static_cast<int>(length), length);
  }
  for (size_t save_threshold : {0, 16, 18}) {
    SCOPED_TRACE(save_threshold);
    SaveEncryptedModelWithExternalData(model, path, "secret", 84, save_threshold);
    for (bool skip : {false, true}) {
      for (int64_t threshold : {-1, 0, 15, 16, 17, 18}) {
        SCOPED_TRACE(skip);
        SCOPED_TRACE(threshold);
        ParseOptions options;
        options.skip_raw_data = skip;
        options.raw_data_threshold = threshold;
        std::vector<std::string> callbacks;
        options.raw_data_callback = [&](TensorProto &tensor, GraphProto *) {
          callbacks.emplace_back(tensor.name());
          return std::function<void()>{};
        };
        ModelProto loaded;
        LoadEncryptedModel(loaded, path, "secret", options);
        ASSERT_EQ(loaded.graph().initializer().size(), 3u);
        size_t selected_count = 0;
        for (size_t i = 0; i < 3; ++i) {
          const auto &expected = model.graph().initializer()[i];
          const auto &actual = loaded.graph().initializer()[i];
          const size_t length = expected.raw_data().size();
          const bool selected = !skip || static_cast<int64_t>(length) < threshold;
          EXPECT_EQ(actual.name(), expected.name());
          ASSERT_EQ(actual.dims().size(), 1u);
          EXPECT_EQ(actual.dims()[0], expected.dims()[0]);
          EXPECT_EQ(actual.has_raw_data(), selected);
          EXPECT_EQ(actual.has_external_data(), !selected && length >= save_threshold);
          if (selected) {
            EXPECT_EQ(actual.raw_data(), expected.raw_data());
            ++selected_count;
          } else if (length >= save_threshold) {
            EXPECT_EQ(actual.data_location(), TensorProto::DataLocation::EXTERNAL);
          }
        }
        EXPECT_EQ(callbacks.size(), selected_count);
      }
    }
  }
}

TEST_F(EncryptedExternalTest, SkipsUnselectedShardsButAuthenticatesSelectedShards) {
  ModelProto model = make_test_model();
  TensorProto large = model.graph().initializer()[0];
  large.set_name("large");
  large.ref_dims()[0] = 5;
  large.ref_raw_data().resize(20);
  *model.ref_graph().add_initializer() = large;
  SaveEncryptedModelWithExternalData(model, path, "secret", 72, 0);
  ModelProto stored = metadata();
  auto &weights = stored.ref_graph().ref_initializer();
  const auto first = directory / entry(weights[0], "location");
  const auto second = directory / entry(weights[1], "location");
  ASSERT_NE(first, second);
  write(second, std::string(72, 'x'));
  ParseOptions options;
  options.skip_raw_data = true;
  options.raw_data_threshold = 17;
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret", options);
  EXPECT_EQ(loaded.graph().initializer()[0].raw_data(), model.graph().initializer()[0].raw_data());
  EXPECT_FALSE(loaded.graph().initializer()[1].has_raw_data());
  EXPECT_TRUE(loaded.graph().initializer()[1].has_external_data());
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret"), std::runtime_error);
  write(first, std::string(68, 'x'));
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret", options), std::runtime_error);
  options.raw_data_threshold = 0;
  EXPECT_NO_THROW(LoadEncryptedModel(loaded, path, "secret", options));
  for (const auto &tensor : loaded.graph().initializer()) {
    EXPECT_FALSE(tensor.has_raw_data());
    EXPECT_TRUE(tensor.has_external_data());
  }
}

TEST_F(EncryptedExternalTest, MetadataOnlyLoadingDoesNotAllocateShardBuffers) {
#if defined(__linux__)
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata();
  auto &tensor = stored.ref_graph().ref_initializer()[0];
  constexpr size_t bytes = 64 * 1024 * 1024;
  set_entry(tensor, "length", std::to_string(bytes));
  std::filesystem::resize_file(directory / entry(tensor, "location"), bytes + 52);
  struct rlimit previous;
  ASSERT_EQ(getrlimit(RLIMIT_AS, &previous), 0);
  std::ifstream statm("/proc/self/statm");
  size_t pages = 0;
  ASSERT_TRUE(static_cast<bool>(statm >> pages));
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GT(page_size, 0);
  struct rlimit limited = previous;
  limited.rlim_cur = pages * static_cast<size_t>(page_size) + 16 * 1024 * 1024;
  ParseOptions options;
  options.skip_raw_data = true;
  options.raw_data_threshold = 0;
  ASSERT_EQ(setrlimit(RLIMIT_AS, &limited), 0);
  EXPECT_NO_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options));
  ASSERT_EQ(setrlimit(RLIMIT_AS, &previous), 0);
  EXPECT_FALSE(tensor.has_raw_data());
  EXPECT_TRUE(tensor.has_external_data());
#else
  GTEST_SKIP() << "Requires Linux address-space accounting.";
#endif
}

TEST_F(EncryptedExternalTest, SkippedPayloadsStillValidateReferencesAndLimits) {
  ModelProto model = make_test_model();
  TensorProto second = model.graph().initializer()[0];
  second.set_name("W2");
  *model.ref_graph().add_initializer() = second;
  SaveEncryptedModelWithExternalData(model, path, "secret", 84, 0);
  const ModelProto original = metadata();
  ParseOptions options;
  options.skip_raw_data = true;
  options.raw_data_threshold = 0;
  for (const std::string field : {"offset", "length"}) {
    for (const std::string value : {"-1", "9223372036854775808", "0junk", "9223372036854775807"}) {
      SCOPED_TRACE(field);
      SCOPED_TRACE(value);
      ModelProto stored = original;
      set_entry(stored.ref_graph().ref_initializer()[1], field, value);
      EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
                   std::runtime_error);
    }
  }
  for (int64_t threshold : {0, 17}) {
    SCOPED_TRACE(threshold);
    options.raw_data_threshold = threshold;
    ModelProto stored = original;
    auto &first = stored.ref_graph().ref_initializer()[0];
    set_entry(first, "length", "17");
    set_entry(first, "onnx_light.encrypted_sha256", std::string(64, '0'));
    EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
                 std::runtime_error);
    stored = original;
    set_entry(stored.ref_graph().ref_initializer()[1], "length", "17");
    EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
                 std::runtime_error);
    EXPECT_FALSE(stored.graph().initializer()[0].has_raw_data());
  }
  options.raw_data_threshold = 0;
  ModelProto stored = original;
  set_entry(stored.ref_graph().ref_initializer()[1], "offset", "33");
  set_entry(stored.ref_graph().ref_initializer()[1], "length", "0");
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
  stored = original;
  stored.ref_graph().ref_initializer()[0].set_data_location(TensorProto::DataLocation::DEFAULT);
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
  stored = original;
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options, 83),
               std::runtime_error);
  options.max_tensor_size_bytes = 15;
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
  options.max_tensor_size_bytes = 0;
  for (const std::string unsafe : {"../escape", "/absolute", "sub/../../escape"}) {
    stored = original;
    set_entry(stored.ref_graph().ref_initializer()[0], "location", unsafe);
    EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
                 std::runtime_error);
  }
  stored = original;
  const auto shard = directory / entry(stored.ref_graph().ref_initializer()[0], "location");
#ifndef _WIN32
  std::filesystem::rename(shard, directory / "real");
  std::filesystem::create_symlink(directory / "real", shard);
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
  std::filesystem::remove(shard);
  std::filesystem::rename(directory / "real", shard);
#endif
  write(shard, std::string(51, 'x'));
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
  std::filesystem::remove(shard);
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", options),
               std::runtime_error);
}

TEST_F(EncryptedExternalTest, RejectsOversizedTensorAndInvalidOptionsBeforeWriting) {
  ModelProto model = make_test_model();
  for (size_t limit : {size_t{0}, size_t{52}, size_t{67}, std::numeric_limits<size_t>::max()})
    EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", limit, 0),
                 std::runtime_error);
  EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "", 68, 0), std::runtime_error);
  EXPECT_TRUE(std::filesystem::is_empty(directory));
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(model, loaded);
}

TEST_F(EncryptedExternalTest, RejectsWrongPasswordMissingTamperedAndTruncatedShards) {
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata(), loaded;
  const auto shard = directory / entry(stored.ref_graph().ref_initializer()[0], "location");
  const std::string bytes = read(shard);
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "wrong"), std::runtime_error);
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "wrong"),
               std::runtime_error);
  for (size_t offset : {size_t{0}, size_t{8}, size_t{24}, size_t{36}, size_t{52}}) {
    std::string corrupt = bytes;
    corrupt[offset] ^= 1;
    write(shard, corrupt);
    EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret"), std::runtime_error);
  }
  write(shard, bytes.substr(0, 51));
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret"), std::runtime_error);
  write(shard, std::string(16, 'x'));
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret"), std::runtime_error);
  std::filesystem::remove(shard);
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret"), std::runtime_error);
}

TEST_F(EncryptedExternalTest, RejectsValidShardSubstitutionAndResourceLimitBypasses) {
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata();
  const auto shard = directory / entry(stored.ref_graph().ref_initializer()[0], "location");
  SaveEncryptedModelWithExternalData(model, (directory / "other.onnxc").string(), "secret", 68, 0);
  ModelProto other;
  LoadEncryptedModelFromString(other, read(directory / "other.onnxc"), "secret");
  write(shard, read(directory / entry(other.ref_graph().ref_initializer()[0], "location")));
  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret"), std::runtime_error);
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  stored = metadata();
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", {}, 67),
               std::runtime_error);
  ParseOptions opts;
  opts.max_tensor_size_bytes = 15;
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret", opts),
               std::runtime_error);
  for (const std::string offset : {"-1", "1", "9223372036854775808", "0junk"}) {
    stored = metadata();
    set_entry(stored.ref_graph().ref_initializer()[0], "offset", offset);
    EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret"),
                 std::runtime_error);
  }
}

TEST_F(EncryptedExternalTest, RejectsUnsafePathsAndPlainLoaders) {
  ModelProto model = make_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata();
  EXPECT_THROW(LoadExternalDataForModel(stored, directory.string()), std::runtime_error);
  std::string serialized;
  stored.SerializeToString(serialized);
  const auto model_path = directory / "plain.onnx";
  write(model_path, serialized);
  const auto location = entry(stored.ref_graph().ref_initializer()[0], "location");
  utils::TwoFilesStream stream(model_path.string(), (directory / location).string());
  ParseOptions opts;
  EXPECT_THROW(model.ParseFromStream(stream, opts), std::runtime_error);
  for (const std::string unsafe : {"../escape", "/absolute", "sub/../../escape"}) {
    stored = metadata();
    set_entry(stored.ref_graph().ref_initializer()[0], "location", unsafe);
    EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret"),
                 std::runtime_error);
  }
#ifndef _WIN32
  stored = metadata();
  const auto target = directory / location;
  std::filesystem::rename(target, directory / "real");
  std::filesystem::create_symlink(directory / "real", target);
  EXPECT_THROW(LoadEncryptedExternalDataForModel(stored, directory.string(), "secret"),
               std::runtime_error);
#endif
}

TEST_F(EncryptedExternalTest, NestedAttributeTensorsAndNoCopyOwnTheirData) {
  ModelProto model = make_test_model();
  auto *attribute = model.ref_graph().ref_node()[0].add_attribute();
  attribute->set_name("value");
  attribute->set_type(AttributeProto::AttributeType::TENSOR);
  *attribute->add_t() = model.graph().initializer()[0];
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ParseOptions opts;
  opts.no_copy = true;
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret", opts);
  expect_same(model, loaded);
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 17);
  loaded = ModelProto{};
  LoadEncryptedModel(loaded, path, "secret", opts);
  expect_same(model, loaded);
}

TEST_F(EncryptedExternalTest, NodeCallbacksObserveResolvedConstantsAndPreserveReplacements) {
  ModelProto model = make_callback_test_model();
  const float expected[4] = {1, 2, 3, 4};
  const float replacement[4] = {5, 6, 7, 8};
  for (const bool external : {false, true}) {
    if (external)
      SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
    else
      SaveEncryptedModel(model, path, "secret");
    std::vector<std::string> visits;
    int raw_callbacks = 0;
    ParseOptions options;
    options.raw_data_callback = [&](TensorProto &, GraphProto *) {
      ++raw_callbacks;
      return std::function<void()>{};
    };
    options.node_callback = [&](NodeProto &node, GraphProto &graph) {
      visits.push_back(graph.name() + ":" + node.op_type());
      if (node.op_type() != "Constant")
        return;
      auto &tensor = node.ref_attribute()[0].ref_t();
      ASSERT_EQ(tensor.raw_data().size(), sizeof(expected));
      EXPECT_EQ(std::memcmp(tensor.raw_data().data(), expected, sizeof(expected)), 0);
      EXPECT_FALSE(tensor.has_external_data());
      EXPECT_EQ(tensor.data_location(), TensorProto::DataLocation::DEFAULT);
      std::memcpy(tensor.ref_raw_data().data(), replacement, sizeof(replacement));
    };
    options.graph_callback = [&](GraphProto &graph) { visits.push_back(graph.name()); };
    options.function_callback = [&](FunctionProto &function) { visits.push_back(function.name()); };
    ModelProto loaded;
    LoadEncryptedModel(loaded, path, "secret", options);
    EXPECT_EQ(visits, (std::vector<std::string>{
                          "first:Constant", "first", "second:Constant", "second", "parent:Relu",
                          "parent:Constant", "parent", "default:Constant", "default", "function"}));
    EXPECT_EQ(raw_callbacks, 5);
    int constants = 0;
    IteratorTensorProto iterator(&loaded);
    while (iterator.next()) {
      if (iterator->name() == "W")
        continue;
      ASSERT_EQ(iterator->raw_data().size(), sizeof(replacement));
      EXPECT_EQ(std::memcmp(iterator->raw_data().data(), replacement, sizeof(replacement)), 0);
      ++constants;
    }
    EXPECT_EQ(constants, 4);
  }
}

TEST_F(EncryptedExternalTest, DeferredNodeCallbacksDoNotRunAfterPartialShardFailure) {
  ModelProto model = make_callback_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto stored = metadata();
  const std::string last_location = entry(stored.ref_functions()[0]
                                              .ref_attribute_proto()[0]
                                              .ref_g()
                                              .ref_node()[0]
                                              .ref_attribute()[0]
                                              .ref_t(),
                                          "location");
  ASSERT_FALSE(last_location.empty());
  ASSERT_TRUE(std::filesystem::remove(directory / last_location));
  int raw_callbacks = 0, node_callbacks = 0;
  ParseOptions options;
  options.raw_data_callback = [&](TensorProto &, GraphProto *) {
    ++raw_callbacks;
    return std::function<void()>{};
  };
  options.node_callback = [&](NodeProto &, GraphProto &) { ++node_callbacks; };
  ModelProto loaded;
  EXPECT_THROW(LoadEncryptedModel(loaded, path, "secret", options), std::runtime_error);
  EXPECT_GT(raw_callbacks, 0);
  EXPECT_EQ(node_callbacks, 0);
}

TEST_F(EncryptedExternalTest, StringMetadataParsingKeepsNodeCallbackSemantics) {
  ModelProto model = make_callback_test_model();
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ParseOptions options;
  options.skip_raw_data = true;
  options.raw_data_threshold = 0;
  int constants = 0, raw_callbacks = 0;
  options.raw_data_callback = [&](TensorProto &, GraphProto *) {
    ++raw_callbacks;
    return std::function<void()>{};
  };
  options.node_callback = [&](NodeProto &node, GraphProto &) {
    if (node.op_type() == "Constant") {
      const auto &tensor = node.attribute(0).t();
      EXPECT_TRUE(tensor.raw_data().empty());
      EXPECT_TRUE(tensor.has_external_data());
      ++constants;
    }
  };
  ModelProto loaded;
  LoadEncryptedModelFromString(loaded, read(path), "secret", options);
  EXPECT_EQ(constants, 4);
  EXPECT_EQ(raw_callbacks, 0);
}

TEST_F(EncryptedExternalTest, DeferredCallbacksIncludeCodecGraphsAndEmptyScopes) {
  ModelProto model = make_test_model();
  auto *codec = add_struct_codec(model, 0, false);
  codec->set_name("codec");
  auto *attribute = codec->add_node()->add_attribute();
  attribute->set_type(AttributeProto::GRAPH);
  attribute->add_g()->set_name("empty");
  std::vector<std::string> visits;
  ParseOptions options;
  options.node_callback = [&](NodeProto &, GraphProto &graph) {
    visits.push_back("node:" + graph.name());
  };
  options.graph_callback = [&](GraphProto &graph) { visits.push_back(graph.name()); };
  options.function_callback = [&](FunctionProto &function) { visits.push_back(function.name()); };
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret", options);
  EXPECT_EQ(visits, (std::vector<std::string>{"node:test_graph", "test_graph", "empty", "codec"}));
}

TEST_F(EncryptedExternalTest, EmbeddedMetadataOnlyLoadingKeepsNodeCallbackSemantics) {
  ModelProto model = make_callback_test_model();
  SaveEncryptedModel(model, path, "secret");
  int constants = 0;
  ParseOptions options;
  options.skip_raw_data = true;
  options.raw_data_threshold = 0;
  options.node_callback = [&](NodeProto &node, GraphProto &) {
    if (node.op_type() == "Constant") {
      EXPECT_TRUE(node.attribute(0).t().raw_data().empty());
      ++constants;
    }
  };
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret", options);
  EXPECT_EQ(constants, 4);
}

TEST_F(EncryptedExternalTest, PreservesPlainExternalDataAndRequiresLoadedWeightsOnExport) {
  ModelProto model = make_test_model();
  const std::string raw(
      reinterpret_cast<const char *>(model.graph().initializer()[0].raw_data().data()), 16);
  auto &tensor = model.ref_graph().ref_initializer()[0];
  tensor.set_data_location(TensorProto::DataLocation::EXTERNAL);
  tensor.clear_raw_data();
  set_entry(tensor, "location", "plain.weights");
  set_entry(tensor, "offset", "0");
  set_entry(tensor, "length", "16");
  EXPECT_THROW(SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0),
               std::runtime_error);
  write(directory / "plain.weights", raw);
  LoadEncryptedExternalDataForModel(model, directory.string(), "secret");
  EXPECT_FALSE(tensor.has_raw_data());
  LoadExternalDataForModel(model, directory.string());
  ModelProto expected = make_test_model();
  expect_same(expected, model);
  SaveEncryptedModelWithExternalData(model, path, "secret", 68, 0);
  ModelProto loaded;
  LoadEncryptedModel(loaded, path, "secret");
  expect_same(expected, loaded);
}

#else // ONNX_LIGHT_HAS_OPENSSL

// Placeholder so the test binary still compiles without OpenSSL.
TEST(onnx_crypt, OpenSSL_not_available) {
  GTEST_SKIP() << "OpenSSL not available – onnx_crypt tests skipped.";
}

#endif // ONNX_LIGHT_HAS_OPENSSL

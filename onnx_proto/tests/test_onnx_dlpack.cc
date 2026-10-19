#include "onnx_dlpack.h"
#include <gtest/gtest.h>
#include <memory>

using namespace ONNX_LIGHT_NAMESPACE;

namespace {
using Managed = std::unique_ptr<DLManagedTensor, void (*)(DLManagedTensor *)>;

Managed Release(TensorProto &tensor) {
  return Managed(ReleaseDLPack(tensor), [](DLManagedTensor *value) { value->deleter(value); });
}

const uint8_t *Data(const TensorProto &tensor) { return tensor.ref_raw_data().data(); }

void ExpectDetached(const TensorProto &tensor) {
  EXPECT_FALSE(tensor.has_raw_data());
  EXPECT_EQ(Data(tensor), nullptr);
  EXPECT_EQ(tensor.ref_raw_data().size(), 0U);
}

TensorProto MakeTensor() {
  TensorProto tensor;
  tensor.set_name("initializer");
  tensor.set_doc_string("preserved");
  tensor.set_data_type(TensorProto::UINT8);
  tensor.add_dims(2);
  tensor.add_dims(3);
  tensor.set_raw_data("abcdef", 6);
  tensor.add_int32_data(42);
  tensor.set_data_location(TensorProto::EXTERNAL);
  auto *entry = tensor.add_external_data();
  entry->set_key("location");
  entry->set_value("unchanged.bin");
  return tensor;
}
} // namespace

TEST(TensorProtoDLPack, OwnedAndAlignedTransferSurvivesSource) {
  for (size_t alignment : {size_t{1}, size_t{64}}) {
    Managed exported(nullptr, nullptr);
    {
      auto tensor = MakeTensor();
      if (alignment > 1) {
        tensor.ref_raw_data().resize_aligned(6, alignment);
        std::memcpy(tensor.ref_raw_data().data(), "abcdef", 6);
      }
      const auto *address = Data(tensor);
      auto expected = MakeTensor();
      expected.clear_raw_data();
      exported = Release(tensor);
      ExpectDetached(tensor);
      std::string actual_bytes, expected_bytes;
      tensor.SerializeToString(actual_bytes);
      expected.SerializeToString(expected_bytes);
      EXPECT_EQ(actual_bytes, expected_bytes);
      EXPECT_EQ(exported->dl_tensor.data, address);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(address) % alignment, 0U);
      tensor.clear_dims();
      tensor.set_data_type(TensorProto::STRING);
      tensor.set_raw_data("replacement");
    }
    const auto &view = exported->dl_tensor;
    EXPECT_EQ(view.ndim, 2);
    EXPECT_EQ(view.shape[0], 2);
    EXPECT_EQ(view.shape[1], 3);
    EXPECT_EQ(view.dtype.code, kDLUInt);
    EXPECT_EQ(view.dtype.bits, 8);
    EXPECT_EQ(view.dtype.lanes, 1);
    EXPECT_EQ(view.device.device_type, kDLCPU);
    EXPECT_EQ(view.device.device_id, 0);
    EXPECT_EQ(view.byte_offset, 0U);
    EXPECT_EQ(view.strides, nullptr);
    EXPECT_EQ(std::memcmp(view.data, "abcdef", 6), 0);
  }
}

TEST(TensorProtoDLPack, SharedNullValuedTokenIsRetainedExactlyOnce) {
  int released = 0;
  auto storage = std::make_unique<uint8_t[]>(6);
  std::memcpy(storage.get(), "abcdef", 6);
  auto tensor = MakeTensor();
  auto *pointer = storage.release();
  tensor.set_raw_data_with_deleter(pointer, 6, [&released, pointer] {
    ++released;
    delete[] pointer;
  });
  ASSERT_FALSE(static_cast<bool>(tensor.ref_raw_data().owner()));
  ASSERT_GT(tensor.ref_raw_data().owner().use_count(), 0);
  TensorProto copied = tensor;
  auto first = Release(tensor);
  auto second = Release(copied);
  ExpectDetached(tensor);
  ExpectDetached(copied);
  EXPECT_EQ(first->dl_tensor.data, pointer);
  EXPECT_EQ(second->dl_tensor.data, pointer);
  tensor.Clear();
  copied.Clear();
  EXPECT_EQ(released, 0);
  first.reset();
  EXPECT_EQ(released, 0);
  EXPECT_EQ(std::memcmp(second->dl_tensor.data, "abcdef", 6), 0);
  second.reset();
  EXPECT_EQ(released, 1);
}

TEST(TensorProtoDLPack, OwnedCallbackOutlivesSource) {
  int released = 0;
  Managed exported(nullptr, nullptr);
  {
    auto tensor = MakeTensor();
    tensor.attach_raw_data_deleter([&released] { ++released; });
    exported = Release(tensor);
    EXPECT_EQ(released, 0);
  }

  EXPECT_EQ(released, 0);
  EXPECT_EQ(std::memcmp(exported->dl_tensor.data, "abcdef", 6), 0);
  exported.reset();
  EXPECT_EQ(released, 1);
}

TEST(TensorProtoDLPack, RejectsActiveOwnedExportLease) {
  auto tensor = MakeTensor();
  const auto *pointer = Data(tensor);
  auto lease = tensor.ref_raw_data().acquire_export_guard();
  auto second_lease = tensor.ref_raw_data().acquire_export_guard();
  EXPECT_EQ(lease, second_lease);
  EXPECT_THROW(ReleaseDLPack(tensor), std::invalid_argument);
  EXPECT_EQ(Data(tensor), pointer);
  EXPECT_TRUE(tensor.has_raw_data());
  tensor.set_name("metadata remains mutable");
  lease.reset();
  EXPECT_THROW(ReleaseDLPack(tensor), std::invalid_argument);
  second_lease.reset();
  EXPECT_FALSE(tensor.ref_raw_data().has_active_exports());
  lease = tensor.ref_raw_data().acquire_export_guard();
  EXPECT_TRUE(tensor.ref_raw_data().has_active_exports());
  lease.reset();
  auto exported = Release(tensor);
  EXPECT_EQ(exported->dl_tensor.data, pointer);
  ExpectDetached(tensor);
}

TEST(TensorProtoDLPack, RejectsUnownedBorrowedStorageAtomically) {
  const uint8_t bytes[] = {1, 2, 3, 4, 5, 6};
  auto tensor = MakeTensor();
  tensor.ref_raw_data().assign_borrowed(bytes, 6);
  std::string before, after;
  tensor.SerializeToString(before);
  EXPECT_THROW(ReleaseDLPack(tensor), std::invalid_argument);
  tensor.SerializeToString(after);
  EXPECT_EQ(before, after);
  EXPECT_TRUE(tensor.has_raw_data());
  EXPECT_EQ(Data(tensor), bytes);
  EXPECT_TRUE(tensor.ref_raw_data().is_borrowed());
}

TEST(TensorProtoDLPack, RejectsInvalidPayloadAtomically) {
  int released = 0;
  auto tensor = MakeTensor();
  tensor.attach_raw_data_deleter([&released] { ++released; });
  for (const auto &dims :
       {std::vector<int64_t>{-1}, {7}, {0}, {INT64_MAX, 2}, std::vector<int64_t>(129, 1)}) {
    tensor.clear_dims();
    for (auto dim : dims)
      tensor.add_dims(dim);
    auto *pointer = Data(tensor);
    std::string before, after;
    tensor.SerializeToString(before);
    EXPECT_THROW(ReleaseDLPack(tensor), std::invalid_argument);
    tensor.SerializeToString(after);
    EXPECT_EQ(before, after);
    EXPECT_EQ(Data(tensor), pointer);
    EXPECT_EQ(released, 0);
  }
}

TEST(TensorProtoDLPack, ValidationDiagnosticPreservesContext) {
  auto tensor = MakeTensor();
  tensor.clear_dims();
  tensor.add_dims(-1);
  try {
    auto exported = Release(tensor);
    FAIL() << "Expected invalid dimensions to be rejected";
  } catch (const std::invalid_argument &error) {
    EXPECT_STREQ(error.what(), "[onnx-light] TensorProto DLPack: dimensions must be non-negative.");
  }
  EXPECT_TRUE(tensor.has_raw_data());
  EXPECT_EQ(tensor.ref_raw_data().size(), 6U);
}

TEST(TensorProtoDLPack, ExplicitEmptyAndScalar) {
  TensorProto tensor;
  tensor.set_data_type(TensorProto::UINT8);
  tensor.add_dims(0);
  EXPECT_THROW(ReleaseDLPack(tensor), std::invalid_argument);
  ExpectDetached(tensor);
  tensor.set_raw_data("", 0);
  ASSERT_TRUE(tensor.has_raw_data());
  auto empty = Release(tensor);
  ExpectDetached(tensor);
  EXPECT_EQ(empty->dl_tensor.data, nullptr);
  EXPECT_EQ(empty->dl_tensor.shape[0], 0);
  tensor.clear_dims();
  tensor.set_raw_data("a");
  auto scalar = Release(tensor);
  ExpectDetached(tensor);
  EXPECT_EQ(scalar->dl_tensor.ndim, 0);
  EXPECT_EQ(*static_cast<const char *>(scalar->dl_tensor.data), 'a');
}

TEST(TensorProtoDLPack, RejectsUnownedEmptyBorrowAtomically) {
  const uint8_t byte = 0;
  TensorProto tensor;
  tensor.set_data_type(TensorProto::UINT8);
  tensor.add_dims(0);
  tensor.ref_raw_data().assign_borrowed(&byte, 0);
  EXPECT_THROW(ReleaseDLPack(tensor), std::invalid_argument);
  EXPECT_TRUE(tensor.has_raw_data());
  EXPECT_EQ(Data(tensor), &byte);
}

TEST(TensorProtoDLPack, EmptyBorrowRetainsLifetimeToken) {
  int released = 0;
  Managed exported(nullptr, nullptr);
  {
    TensorProto tensor;
    tensor.set_data_type(TensorProto::UINT8);
    tensor.add_dims(0);
    auto *pointer = new uint8_t[1];
    tensor.set_raw_data_with_deleter(pointer, 0, [&released, pointer] {
      ++released;
      delete[] pointer;
    });
    ASSERT_TRUE(tensor.has_raw_data());
    exported = Release(tensor);
    ExpectDetached(tensor);
  }
  EXPECT_EQ(released, 0);
  EXPECT_EQ(exported->dl_tensor.data, nullptr);
  exported.reset();
  EXPECT_EQ(released, 1);
}

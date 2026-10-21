#include "onnx.h"
#include "onnx_dlpack.h"
#include <functional>
#include <gtest/gtest.h>
#include <utility>

using namespace ONNX_LIGHT_NAMESPACE;

TEST(SharedPayloadCopy, OwnedBytesShareOnlyExplicitlyAndDetachBothWays) {
  TensorProto source;
  source.set_raw_data("weights");
  const auto *original = std::as_const(source).raw_data().data();
  TensorProto ordinary(source);
  EXPECT_NE(ordinary.raw_data().data(), original);
  TensorProto assigned;
  assigned = source;
  EXPECT_NE(assigned.raw_data().data(), original);
  TensorProto copied;
  copied.CopyFrom(source);
  EXPECT_NE(copied.raw_data().data(), original);

  auto snapshot = utils::CopyWithSharedPayloads(source);
  ASSERT_EQ(std::as_const(snapshot).raw_data().data(), original);
  source.ref_raw_data()[0] = 'W';
  EXPECT_EQ(std::string(snapshot.raw_data()), "weights");
  EXPECT_NE(std::as_const(source).raw_data().data(), original);
  auto second = utils::CopyWithSharedPayloads(snapshot);
  snapshot.ref_raw_data().data_as_char()[1] = 'E';
  EXPECT_EQ(std::string(second.raw_data()), "weights");
  EXPECT_EQ(std::string(snapshot.raw_data()), "wEights");
}

TEST(SharedPayloadCopy, ByteMutatorsDoNotChangeSnapshots) {
  using Buffer = utils::ByteSpan;
  const std::vector<std::function<void(Buffer &)>> mutations = {
      [](Buffer &b) { b.resize(2); },
      [](Buffer &b) { b.resize(100); },
      [](Buffer &b) { b.resize(0); },
      [](Buffer &b) { b.resize_aligned(100, 64); },
      [](Buffer &b) { b.assign("other", 5); },
      [](Buffer &b) { b = std::vector<uint8_t>{8, 9}; },
      [](Buffer &b) { b.push_back(8); },
      [](Buffer &b) { b.clear(); },
      [](Buffer &b) { b.assign_borrowed(reinterpret_cast<const uint8_t *>("new"), 3); },
      [](Buffer &b) { b.data()[0] = 0; },
      [](Buffer &b) { b[0] = 0; }};
  for (bool aligned : {false, true}) {
    for (const auto &mutate : mutations) {
      Buffer buffer;
      if (aligned) {
        buffer.resize_aligned(7, 64);
        std::memcpy(buffer.data(), "weights", 7);
      } else {
        buffer.assign("weights", 7);
      }
      auto snapshot = utils::CopyWithSharedPayloads(buffer);
      ASSERT_EQ(std::as_const(buffer).data(), std::as_const(snapshot).data());
      mutate(buffer);
      EXPECT_EQ(std::string(snapshot), "weights");
      EXPECT_EQ(snapshot.is_aligned_owned(), aligned);
      if (aligned) {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(std::as_const(snapshot).data()) % 64, 0u);
      }
    }
  }
}

TEST(SharedPayloadCopy, AlignedMutationRetainsAlignmentAndLogicalBytes) {
  utils::ByteSpan source;
  source.resize_aligned(17, 256);
  std::memset(source.data(), 9, source.size());
  auto snapshot = utils::CopyWithSharedPayloads(source);
  const auto *original = std::as_const(source).data();
  ASSERT_EQ(original, std::as_const(snapshot).data());
  snapshot[0] = 4;
  EXPECT_NE(original, std::as_const(snapshot).data());
  EXPECT_TRUE(snapshot.is_aligned_owned());
  EXPECT_EQ(reinterpret_cast<uintptr_t>(std::as_const(snapshot).data()) % 256, 0u);
  EXPECT_EQ(std::as_const(source)[0], 9);
  EXPECT_EQ(std::as_const(snapshot)[16], 9);
}

TEST(SharedPayloadCopy, BorrowedOwnerRetainsLifetimeAndReadOnlyContract) {
  auto owner = std::make_shared<std::vector<uint8_t>>(std::initializer_list<uint8_t>{1, 2, 3});
  std::weak_ptr<void> lifetime = owner;
  TensorProto source;
  source.ref_raw_data().assign_borrowed(owner->data(), owner->size(), owner);
  auto snapshot = utils::CopyWithSharedPayloads(source);
  auto ordinary = source;
  TensorProto copy_from;
  copy_from.CopyFrom(source);
  ASSERT_EQ(std::as_const(snapshot).raw_data().data(), owner->data());
  EXPECT_EQ(std::as_const(copy_from).raw_data().data(), owner->data());
  EXPECT_EQ(std::as_const(ordinary).raw_data().data(), owner->data());
  EXPECT_TRUE(snapshot.raw_data().is_borrowed());
  EXPECT_THROW(snapshot.ref_raw_data().data(), std::runtime_error);
  owner.reset();
  source.Clear();
  ordinary.Clear();
  copy_from.Clear();
  EXPECT_FALSE(lifetime.expired());
  EXPECT_EQ(std::as_const(snapshot).raw_data()[2], 3);
  snapshot.Clear();
  EXPECT_TRUE(lifetime.expired());
}

TEST(SharedPayloadCopy, NullOwnerTokensAndOwnerlessBorrowedStorage) {
  bool released = false;
  const uint8_t bytes[] = {1, 2, 3};
  utils::ByteSpan borrowed;
  borrowed.assign_with_deleter(bytes, 3, [&released]() { released = true; });
  auto retained = utils::CopyWithSharedPayloads(borrowed);
  EXPECT_EQ(std::as_const(retained).data(), bytes);
  borrowed.clear();
  EXPECT_FALSE(released);
  retained.clear();
  EXPECT_TRUE(released);

  borrowed.assign_borrowed(bytes, 3);
  auto safe = utils::CopyWithSharedPayloads(borrowed);
  EXPECT_FALSE(safe.is_borrowed());
  EXPECT_NE(std::as_const(safe).data(), bytes);
  EXPECT_EQ(std::as_const(safe)[2], 3);
  auto ordinary = borrowed;
  EXPECT_TRUE(ordinary.is_borrowed());
  EXPECT_EQ(std::as_const(ordinary).data(), bytes);
}

TEST(SharedPayloadCopy, EmptyPresenceAndDefaultValues) {
  TensorProto source;
  auto absent = utils::CopyWithSharedPayloads(source);
  EXPECT_FALSE(absent.has_raw_data());
  EXPECT_EQ(absent.float_data_size(), 0);
  source.ref_raw_data().resize(0);
  ASSERT_TRUE(source.has_raw_data());
  auto present = utils::CopyWithSharedPayloads(source);
  auto ordinary = source;
  EXPECT_TRUE(present.has_raw_data());
  EXPECT_TRUE(ordinary.has_raw_data());
  EXPECT_EQ(std::as_const(source).raw_data().data(), std::as_const(present).raw_data().data());
  present.ref_raw_data().push_back(1);
  EXPECT_EQ(source.raw_data().size(), 0u);
}

TEST(SharedPayloadCopy, RepeatedMutatorsDetach) {
  using Field = utils::RepeatedField<int64_t>;
  const std::vector<std::function<void(Field &)>> mutations = {
      [](Field &f) { f[0] = 9; },
      [](Field &f) { f.Set(0, 9); },
      [](Field &f) { *f.Mutable(0) = 9; },
      [](Field &f) { f.mutable_values()[0] = 9; },
      [](Field &f) { *f.begin() = 9; },
      [](Field &f) { *(f.end() - 1) = 9; },
      [](Field &f) { f.back() = 9; },
      [](Field &f) { f.data()[0] = 9; },
      [](Field &f) { f.mutable_data()[0] = 9; },
      [](Field &f) { f.reserve(100); },
      [](Field &f) { f.Reserve(100); },
      [](Field &f) { f.clear(); },
      [](Field &f) { f.Clear(); },
      [](Field &f) { f.resize(1); },
      [](Field &f) { f.Resize(8, 9); },
      [](Field &f) { f.push_back(9); },
      [](Field &f) { f.add() = 9; },
      [](Field &f) { *f.Add() = 9; },
      [](Field &f) { f.Add(9); },
      [](Field &f) { f.emplace_back(9); },
      [](Field &f) { f.emplace_back(); },
      [](Field &f) { f.extend(std::vector<int64_t>{9}); },
      [](Field &f) { f.extend(f); },
      [](Field &f) {
        const std::vector<int64_t> v{9};
        f.Add(v.begin(), v.end());
      },
      [](Field &f) {
        const std::vector<int64_t> v{9};
        f.Assign(v.begin(), v.end());
      },
      [](Field &f) { f = {9}; },
      [](Field &f) { f.CopyFrom(Field{}); },
      [](Field &f) { f.remove_range(0, 3, 1); }};
  for (const auto &mutate : mutations) {
    Field field;
    field = {1, 2, 3};
    auto snapshot = utils::CopyWithSharedPayloads(field);
    ASSERT_EQ(std::as_const(field).data(), std::as_const(snapshot).data());
    mutate(field);
    EXPECT_EQ(snapshot.values(), (std::vector<int64_t>{1, 2, 3}));
  }
}

TEST(SharedPayloadCopy, TypedPayloadsStringsAndOrdinaryCopies) {
  TensorProto source;
  source.add_float_data(1.5f);
  source.add_double_data(2.5);
  source.add_int32_data(3);
  source.add_int64_data(4);
  source.add_uint64_data(5);
  source.add_string_data("a long owned string payload, not a short string");
  auto snapshot = utils::CopyWithSharedPayloads(source);
  const auto &before = std::as_const(source);
  const auto &saved = std::as_const(snapshot);
  EXPECT_EQ(before.float_data().data(), saved.float_data().data());
  EXPECT_EQ(before.double_data().data(), saved.double_data().data());
  EXPECT_EQ(before.int32_data().data(), saved.int32_data().data());
  EXPECT_EQ(before.int64_data().data(), saved.int64_data().data());
  EXPECT_EQ(before.uint64_data().data(), saved.uint64_data().data());
  EXPECT_EQ(before.string_data().data(), saved.string_data().data());
  EXPECT_EQ(before.string_data()[0].data(), saved.string_data()[0].data());
  auto ordinary = source;
  EXPECT_NE(before.float_data().data(), std::as_const(ordinary).float_data().data());
  EXPECT_NE(before.string_data().data(), std::as_const(ordinary).string_data().data());
  source.ref_string_data()[0] = "changed";
  snapshot.ref_float_data()[0] = 9;
  EXPECT_EQ(before.float_data()[0], 1.5f);
  EXPECT_EQ(saved.string_data()[0], "a long owned string payload, not a short string");
}

TEST(SharedPayloadCopy, NestedMessagesNeverShareStructure) {
  ModelProto source;
  auto *body = source.mutable_graph()->add_node()->add_attribute()->mutable_g();
  body->add_initializer()->set_raw_data("nested");
  body->add_initializer()->add_float_data(7);
  auto snapshot = utils::CopyWithSharedPayloads(source);
  const auto &a = std::as_const(source).graph().node(0).attribute(0).g();
  const auto &b = std::as_const(snapshot).graph().node(0).attribute(0).g();
  EXPECT_NE(&a, &b);
  EXPECT_NE(&a.initializer(0), &b.initializer(0));
  EXPECT_EQ(a.initializer(0).raw_data().data(), b.initializer(0).raw_data().data());
  EXPECT_EQ(a.initializer(1).float_data().data(), b.initializer(1).float_data().data());
  body->mutable_initializer(0)->ref_raw_data()[0] = 'N';
  EXPECT_EQ(std::string(b.initializer(0).raw_data()), "nested");
  auto repeated = utils::CopyWithSharedPayloads(source.graph().node());
  EXPECT_NE(&repeated[0], &source.graph().node(0));

  utils::RepeatedField<TensorProto> messages;
  messages.add().set_raw_data("value storage");
  auto message_snapshot = utils::CopyWithSharedPayloads(messages);
  EXPECT_NE(std::as_const(messages).data(), std::as_const(message_snapshot).data());
  EXPECT_EQ(std::as_const(messages)[0].raw_data().data(),
            std::as_const(message_snapshot)[0].raw_data().data());
}

TEST(SharedPayloadCopy, ActiveExportsForceCopyAndRetainOldAllocation) {
  for (bool aligned : {false, true}) {
    utils::ByteSpan source;
    if (aligned)
      source.resize_aligned(7, 64);
    else
      source.resize(7);
    std::memcpy(source.data(), "weights", 7);
    auto lease = source.acquire_export_guard();
    const auto *exported = std::as_const(source).data();
    auto snapshot = utils::CopyWithSharedPayloads(source);
    EXPECT_NE(std::as_const(snapshot).data(), exported);
    EXPECT_EQ(std::string(snapshot), "weights");
    source.clear();
    EXPECT_EQ(std::memcmp(exported, "weights", 7), 0);
  }

  utils::ByteSpan source;
  source.assign("weights", 7);
  auto snapshot = utils::CopyWithSharedPayloads(source);
  const auto *saved = std::as_const(snapshot).data();
  auto lease = source.acquire_export_guard();
  EXPECT_EQ(std::as_const(source).data(), saved);
  EXPECT_EQ(std::as_const(snapshot).data(), saved);
  source.clear();
  EXPECT_EQ(std::string(snapshot), "weights");

  const uint8_t bytes[] = {1, 2};
  source.assign_borrowed(bytes, 2, std::make_shared<int>(1));
  lease = source.acquire_export_guard();
  auto borrowed_snapshot = utils::CopyWithSharedPayloads(source);
  EXPECT_FALSE(borrowed_snapshot.is_borrowed());
  EXPECT_NE(std::as_const(borrowed_snapshot).data(), bytes);
}

TEST(SharedPayloadCopy, ScopeEndsOnCopyFailure) {
  struct Throwing {
    Throwing() = default;
    Throwing(const Throwing &) { throw std::runtime_error("copy failure"); }
  };
  EXPECT_THROW(utils::CopyWithSharedPayloads(Throwing{}), std::runtime_error);
  EXPECT_FALSE(utils::detail::IsSharedPayloadCopy());
}

TEST(SharedPayloadCopy, ExportLeasesTrackEachAllocationAndBorrowedOwner) {
  utils::ByteSpan source;
  source.assign("original", 8);
  auto first = source.acquire_export_guard();
  const auto *original = std::as_const(source).data();
  source.assign("replacement", 11);
  auto second = source.acquire_export_guard();
  EXPECT_NE(first, second);
  const auto *replacement = std::as_const(source).data();
  source[0] = 'R';
  auto third = source.acquire_export_guard();
  EXPECT_NE(second, third);
  source.clear();
  EXPECT_EQ(std::memcmp(original, "original", 8), 0);
  EXPECT_EQ(std::memcmp(replacement, "replacement", 11), 0);

  auto owner = std::make_shared<std::vector<uint8_t>>(std::initializer_list<uint8_t>{1, 2});
  std::weak_ptr<void> lifetime = owner;
  const auto *borrowed = owner->data();
  source.assign_borrowed(borrowed, owner->size(), owner);
  auto lease = source.acquire_export_guard();
  owner.reset();
  source.clear();
  EXPECT_FALSE(lifetime.expired());
  EXPECT_EQ(borrowed[1], 2);
  lease.reset();
  EXPECT_TRUE(lifetime.expired());
}

TEST(SharedPayloadCopy, WritableDLPackTransferDetachesOwnedSnapshot) {
  TensorProto source;
  source.set_data_type(TensorProto::UINT8);
  source.add_dims(7);
  source.set_raw_data("weights");
  auto snapshot = utils::CopyWithSharedPayloads(source);
  const auto *saved = std::as_const(snapshot).raw_data().data();
  auto *exported = ReleaseDLPack(source);
  EXPECT_NE(exported->dl_tensor.data, saved);
  static_cast<uint8_t *>(exported->dl_tensor.data)[0] = 'W';
  EXPECT_EQ(std::string(snapshot.raw_data()), "weights");
  EXPECT_FALSE(source.has_raw_data());
  exported->deleter(exported);
}

TEST(SharedPayloadCopy, ReplacementDropsExportStateButRetainsOldReadOnlyView) {
  TensorProto source;
  source.set_data_type(TensorProto::UINT8);
  source.add_dims(6);
  source.set_raw_data("first!");
  auto old_lease = source.raw_data().acquire_export_guard();
  const auto *old_view = std::as_const(source).raw_data().data();

  source.set_raw_data("second");
  EXPECT_FALSE(source.raw_data().has_active_exports());
  auto snapshot = utils::CopyWithSharedPayloads(source);
  EXPECT_EQ(std::as_const(source).raw_data().data(), std::as_const(snapshot).raw_data().data());
  EXPECT_EQ(std::memcmp(old_view, "first!", 6), 0);

  auto *exported = ReleaseDLPack(source);
  EXPECT_EQ(std::memcmp(exported->dl_tensor.data, "second", 6), 0);
  EXPECT_EQ(std::string(snapshot.raw_data()), "second");
  source.Clear();
  EXPECT_EQ(std::memcmp(old_view, "first!", 6), 0);
  exported->deleter(exported);
}

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::And;
using onnx_backend_test::kernel::Greater;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Less;
using onnx_backend_test::kernel::Or;
using onnx_backend_test::kernel::Xor;

namespace Test {

TEST(BackendKernelClass, AndClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, AndClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {}, {1});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(7)};
  Or or_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = or_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(7)};
  Xor xor_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = xor_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, LogicalRejectsNonBoolTensors) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 0.0f});
  Tensor y("", TensorProto::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)and_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, AndInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
  and_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(7)};
  Or or_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  or_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(7)};
  Xor xor_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  xor_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, GreaterClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  Tensor z = greater_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0); // 1 > 2 -> false
  EXPECT_EQ(z.data[1], 0); // 2 > 2 -> false
  EXPECT_EQ(z.data[2], 1); // 3 > 2 -> true
  EXPECT_EQ(z.data[3], 1); // 4 > 2 -> true
}

TEST(BackendKernelClass, GreaterClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.5f});
  Tensor z = greater_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 1);
}

TEST(BackendKernelClass, GreaterInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
  greater_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 1);
}

TEST(BackendKernelClass, GreaterRejectsUnsupportedDtype) {
  // BOOL inputs are not in the supported dtype set (FLOAT/INT8/INT16/UINT8/
  // UINT16/UINT32/UINT64) so the kernel must reject them.
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2}, {1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2}, {0, 1});
  EXPECT_THROW((void)greater_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, GreaterClassMatchesReferenceInt8) {
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x = Tensor::FromInt8("", {4}, {-2, 0, 3, 7});
  Tensor y = Tensor::FromInt8("", {4}, {-1, 0, 1, 9});
  Tensor z = greater_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0); // -2 > -1 -> false
  EXPECT_EQ(z.data[1], 0); //  0 >  0 -> false
  EXPECT_EQ(z.data[2], 1); //  3 >  1 -> true
  EXPECT_EQ(z.data[3], 0); //  7 >  9 -> false
}

TEST(BackendKernelClass, GreaterClassMatchesReferenceUint32) {
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x = Tensor::FromUint32("", {4}, {1u, 5u, 0u, 7u});
  Tensor y = Tensor::FromUint32("", {4}, {2u, 5u, 0u, 6u});
  Tensor z = greater_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 1);
}

TEST(BackendKernelClass, LessClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(13)};
  Less less_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  Tensor z = less_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(TensorProto::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1); // 1 < 2 -> true
  EXPECT_EQ(z.data[1], 0); // 2 < 2 -> false
  EXPECT_EQ(z.data[2], 0); // 3 < 2 -> false
  EXPECT_EQ(z.data[3], 0); // 4 < 2 -> false
}

TEST(BackendKernelClass, LessClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(13)};
  Less less_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.5f});
  Tensor z = less_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, LessInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(13)};
  Less less_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  Tensor z("", TensorProto::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
  less_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, LessRejectsUnsupportedDtype) {
  // BOOL inputs are not in the supported dtype set (FLOAT/INT8/INT16/UINT8/
  // UINT16/UINT32/UINT64) so the kernel must reject them.
  const KernelContext ctx{DefaultOpset(13)};
  Less less_kernel{ctx};
  Tensor x("", TensorProto::DataType::BOOL, {2}, {1, 0});
  Tensor y("", TensorProto::DataType::BOOL, {2}, {0, 1});
  EXPECT_THROW((void)less_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, LessClassMatchesReferenceInt16) {
  const KernelContext ctx{DefaultOpset(13)};
  Less less_kernel{ctx};
  Tensor x = Tensor::FromInt16("", {4}, {-2, 0, 3, 7});
  Tensor y = Tensor::FromInt16("", {4}, {-1, 0, 1, 9});
  Tensor z = less_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1); // -2 < -1
  EXPECT_EQ(z.data[1], 0); //  0 <  0
  EXPECT_EQ(z.data[2], 0); //  3 <  1
  EXPECT_EQ(z.data[3], 1); //  7 <  9
}

TEST(BackendKernelClass, LessClassMatchesReferenceUint64) {
  const KernelContext ctx{DefaultOpset(13)};
  Less less_kernel{ctx};
  Tensor x = Tensor::FromUint64("", {4}, {1ull, 5ull, 0ull, 7ull});
  Tensor y = Tensor::FromUint64("", {4}, {2ull, 5ull, 0ull, 6ull});
  Tensor z = less_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

} // namespace Test

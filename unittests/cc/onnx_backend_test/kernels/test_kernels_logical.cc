// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_backend_test/kernels/kernel_context.h"
#include "onnx_backend_test/kernels/logical/include_logical_kernels.h"
#include "onnx_backend_test/test_case.h"

#include <gtest/gtest.h>

#include <limits>
#include <stdexcept>
#include <vector>

using namespace ONNX_LIGHT_NAMESPACE;
using onnx_backend_test::DefaultOpset;
using onnx_backend_test::Tensor;
using onnx_backend_test::kernel::And;
using onnx_backend_test::kernel::BitwiseAnd;
using onnx_backend_test::kernel::BitwiseNot;
using onnx_backend_test::kernel::BitwiseOr;
using onnx_backend_test::kernel::BitwiseXor;
using onnx_backend_test::kernel::Equal;
using onnx_backend_test::kernel::Greater;
using onnx_backend_test::kernel::GreaterOrEqual;
using onnx_backend_test::kernel::IsInf;
using onnx_backend_test::kernel::IsNaN;
using onnx_backend_test::kernel::KernelContext;
using onnx_backend_test::kernel::Less;
using onnx_backend_test::kernel::Not;
using onnx_backend_test::kernel::Or;
using onnx_backend_test::kernel::Where;
using onnx_backend_test::kernel::Xor;

namespace Test {

TEST(BackendKernelClass, AndClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = and_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, AndClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {}, {1});
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
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = or_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(7)};
  Xor xor_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z = xor_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, LogicalRejectsNonBoolTensors) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 0.0f});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)and_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, AndInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(7)};
  And and_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", onnx_backend_test::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
  and_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, OrInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(7)};
  Or or_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", onnx_backend_test::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
  or_kernel(x, y, z);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, XorInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(7)};
  Xor xor_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2, 2}, {1, 1, 0, 0});
  Tensor z("", onnx_backend_test::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4));
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
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
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
  Tensor z("", onnx_backend_test::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
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
  Tensor x("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2}, {0, 1});
  EXPECT_THROW((void)greater_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, GreaterClassMatchesReferenceInt8) {
  const KernelContext ctx{DefaultOpset(13)};
  Greater greater_kernel{ctx};
  Tensor x = Tensor::FromInt8("", {4}, {-2, 0, 3, 7});
  Tensor y = Tensor::FromInt8("", {4}, {-1, 0, 1, 9});
  Tensor z = greater_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
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
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
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
  Tensor z("", onnx_backend_test::DataType::BOOL, {2, 2}, std::vector<uint8_t>(4, 9));
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
  Tensor x("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2}, {0, 1});
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

TEST(BackendKernelClass, GreaterOrEqualClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(16)};
  GreaterOrEqual ge_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {2, 2}, {2.0f, 2.0f, 2.0f, 2.0f});
  Tensor z = ge_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(z.data[0], 0); // 1 >= 2 -> false
  EXPECT_EQ(z.data[1], 1); // 2 >= 2 -> true
  EXPECT_EQ(z.data[2], 1); // 3 >= 2 -> true
  EXPECT_EQ(z.data[3], 1); // 4 >= 2 -> true
}

TEST(BackendKernelClass, GreaterOrEqualClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(16)};
  GreaterOrEqual ge_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {}, {2.0f});
  Tensor z = ge_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 1);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 1);
}

TEST(BackendKernelClass, GreaterOrEqualInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(16)};
  GreaterOrEqual ge_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {4}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("", {4}, {2.0f, 2.0f, 2.0f, 2.0f});
  Tensor output("", onnx_backend_test::DataType::BOOL, {4}, {0, 0, 0, 0});
  ge_kernel(x, y, output);
  ASSERT_EQ(output.element_count(), 4);
  EXPECT_EQ(output.data[0], 0);
  EXPECT_EQ(output.data[1], 1);
  EXPECT_EQ(output.data[2], 1);
  EXPECT_EQ(output.data[3], 1);
}

TEST(BackendKernelClass, GreaterOrEqualRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(16)};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2}, {1, 1});
  GreaterOrEqual ge_kernel{ctx};
  EXPECT_THROW({ (void)ge_kernel(x, y); }, std::invalid_argument);
}

TEST(BackendKernelClass, EqualClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2, 2}, {1, 2, 3, 4});
  Tensor y = Tensor::FromInt32("", {2, 2}, {1, 0, 3, 0});
  Tensor z = equal_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1); // 1 == 1
  EXPECT_EQ(z.data[1], 0); // 2 == 0
  EXPECT_EQ(z.data[2], 1); // 3 == 3
  EXPECT_EQ(z.data[3], 0); // 4 == 0
}

TEST(BackendKernelClass, EqualClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2, 2}, {1, 2, 3, 4});
  Tensor y = Tensor::FromInt32("", {}, {3});
  Tensor z = equal_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 0);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
  EXPECT_EQ(z.data[3], 0);
}

TEST(BackendKernelClass, EqualInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {3}, {1, 2, 3});
  Tensor y = Tensor::FromInt32("", {3}, {1, 0, 3});
  Tensor out("", onnx_backend_test::DataType::BOOL, {3}, std::vector<uint8_t>(3));
  equal_kernel(x, y, out);
  EXPECT_EQ(out.data[0], 1);
  EXPECT_EQ(out.data[1], 0);
  EXPECT_EQ(out.data[2], 1);
}

TEST(BackendKernelClass, EqualClassMatchesReferenceBool) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {4}, {1, 0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {4}, {1, 1, 0, 0});
  Tensor z = equal_kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 0);
  EXPECT_EQ(z.data[3], 1);
}

TEST(BackendKernelClass, EqualClassMatchesReferenceFloat) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x = Tensor::FromFloat("", {3}, {1.0f, 2.0f, 3.0f});
  Tensor y = Tensor::FromFloat("", {3}, {1.0f, 0.0f, 3.0f});
  Tensor z = equal_kernel(x, y);
  ASSERT_EQ(z.element_count(), 3);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
  EXPECT_EQ(z.data[2], 1);
}

TEST(BackendKernelClass, EqualClassMatchesReferenceString) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x = Tensor::FromStrings("", {2}, {"string1", "string2"});
  Tensor y = Tensor::FromStrings("", {2}, {"string1", "string3"});
  Tensor z = equal_kernel(x, y);
  ASSERT_EQ(z.element_count(), 2);
  EXPECT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
}

TEST(BackendKernelClass, EqualClassBroadcastsScalarString) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x = Tensor::FromStrings("", {2}, {"string1", "string2"});
  Tensor y = Tensor::FromStrings("", {1}, {"string1"});
  Tensor z = equal_kernel(x, y);
  ASSERT_EQ(z.element_count(), 2);
  EXPECT_EQ(z.data[0], 1);
  EXPECT_EQ(z.data[1], 0);
}

TEST(BackendKernelClass, EqualRejectsUnsupportedDtype) {
  const KernelContext ctx{DefaultOpset(19)};
  Equal equal_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::COMPLEX64, {2}, std::vector<uint8_t>(16));
  Tensor y("", onnx_backend_test::DataType::COMPLEX64, {2}, std::vector<uint8_t>(16));
  EXPECT_THROW((void)equal_kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, WhereClassSelectsValuesElementwise) {
  const KernelContext ctx{DefaultOpset(16)};
  Where where_kernel{ctx};
  Tensor condition = Tensor::FromBool("condition", {2, 2}, {1, 0, 1, 0});
  Tensor x = Tensor::FromFloat("x", {2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
  Tensor y = Tensor::FromFloat("y", {2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
  Tensor output = where_kernel(condition, x, y);
  ASSERT_EQ(output.data_type, static_cast<int32_t>(onnx_backend_test::DataType::FLOAT));
  ASSERT_EQ(output.shape, (std::vector<int64_t>{2, 2}));
  const float *values = output.AsFloat();
  EXPECT_FLOAT_EQ(values[0], 1.0f);
  EXPECT_FLOAT_EQ(values[1], 6.0f);
  EXPECT_FLOAT_EQ(values[2], 3.0f);
  EXPECT_FLOAT_EQ(values[3], 8.0f);
}

TEST(BackendKernelClass, WhereClassBroadcastsInputs) {
  const KernelContext ctx{DefaultOpset(16)};
  Where where_kernel{ctx};
  Tensor condition = Tensor::FromBool("condition", {2, 1}, {1, 0});
  Tensor x = Tensor::FromInt32("x", {2, 3}, {1, 2, 3, 4, 5, 6});
  Tensor y = Tensor::FromInt32("y", {1, 3}, {10, 20, 30});
  Tensor output = where_kernel(condition, x, y);
  ASSERT_EQ(output.shape, (std::vector<int64_t>{2, 3}));
  const int32_t *values = output.AsInt32();
  EXPECT_EQ(values[0], 1);
  EXPECT_EQ(values[1], 2);
  EXPECT_EQ(values[2], 3);
  EXPECT_EQ(values[3], 10);
  EXPECT_EQ(values[4], 20);
  EXPECT_EQ(values[5], 30);
}

TEST(BackendKernelClass, WhereRejectsNonBoolCondition) {
  const KernelContext ctx{DefaultOpset(16)};
  Where where_kernel{ctx};
  Tensor condition = Tensor::FromFloat("condition", {2}, {1.0f, 0.0f});
  Tensor x = Tensor::FromInt32("x", {2}, {1, 2});
  Tensor y = Tensor::FromInt32("y", {2}, {3, 4});
  EXPECT_THROW((void)where_kernel(condition, x, y), std::invalid_argument);
}

// ---------------------------------------------------------------------------
// Bitwise kernels
// ---------------------------------------------------------------------------

TEST(BackendKernelClass, BitwiseAndClassMatchesReferenceInt32) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseAnd kernel{ctx};
  Tensor x = Tensor::FromInt32("", {4}, {0xF0, 0x0F, 0xAA, 0x55});
  Tensor y = Tensor::FromInt32("", {4}, {0xFF, 0xFF, 0x0F, 0xF0});
  Tensor z = kernel(x, y);
  ASSERT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  ASSERT_EQ(z.element_count(), 4);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], 0xF0 & 0xFF);
  EXPECT_EQ(p[1], 0x0F & 0xFF);
  EXPECT_EQ(p[2], 0xAA & 0x0F);
  EXPECT_EQ(p[3], 0x55 & 0xF0);
}

TEST(BackendKernelClass, BitwiseOrClassMatchesReferenceInt32) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseOr kernel{ctx};
  Tensor x = Tensor::FromInt32("", {3}, {1, 2, 4});
  Tensor y = Tensor::FromInt32("", {3}, {8, 8, 8});
  Tensor z = kernel(x, y);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], 9);
  EXPECT_EQ(p[1], 10);
  EXPECT_EQ(p[2], 12);
}

TEST(BackendKernelClass, BitwiseXorClassMatchesReferenceInt32) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseXor kernel{ctx};
  Tensor x = Tensor::FromInt32("", {3}, {0xFF, 0xAA, 0x0F});
  Tensor y = Tensor::FromInt32("", {3}, {0x0F, 0xFF, 0xF0});
  Tensor z = kernel(x, y);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], 0xFF ^ 0x0F);
  EXPECT_EQ(p[1], 0xAA ^ 0xFF);
  EXPECT_EQ(p[2], 0x0F ^ 0xF0);
}

TEST(BackendKernelClass, BitwiseAndClassBroadcastsScalar) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseAnd kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2, 2}, {0xF0, 0x0F, 0xAA, 0x55});
  Tensor y = Tensor::FromInt32("", {}, {0x0F});
  Tensor z = kernel(x, y);
  ASSERT_EQ(z.element_count(), 4);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], 0x00);
  EXPECT_EQ(p[1], 0x0F);
  EXPECT_EQ(p[2], 0x0A);
  EXPECT_EQ(p[3], 0x05);
}

TEST(BackendKernelClass, BitwiseAndInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseAnd kernel{ctx};
  Tensor x = Tensor::FromInt32("", {3}, {0xF0, 0x0F, 0xAA});
  Tensor y = Tensor::FromInt32("", {3}, {0xFF, 0xFF, 0x0F});
  Tensor z("", onnx_backend_test::DataType::INT32, {3}, std::vector<uint8_t>(3 * sizeof(int32_t)));
  kernel(x, y, z);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], 0xF0);
  EXPECT_EQ(p[1], 0x0F);
  EXPECT_EQ(p[2], 0x0A);
}

TEST(BackendKernelClass, BitwiseAndRejectsFloatTensors) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseAnd kernel{ctx};
  Tensor x = Tensor::FromFloat("", {2}, {1.0f, 0.0f});
  Tensor y = Tensor::FromFloat("", {2}, {1.0f, 1.0f});
  EXPECT_THROW((void)kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, BitwiseAndRejectsBoolTensors) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseAnd kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {2}, {1, 1});
  EXPECT_THROW((void)kernel(x, y), std::invalid_argument);
}

TEST(BackendKernelClass, BitwiseNotClassMatchesReferenceInt32) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseNot kernel{ctx};
  Tensor x = Tensor::FromInt32("", {4}, {0, -1, 0xFF, 0x0F});
  Tensor z = kernel(x);
  ASSERT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::INT32));
  ASSERT_EQ(z.element_count(), 4);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], ~0);
  EXPECT_EQ(p[1], ~(-1));
  EXPECT_EQ(p[2], ~0xFF);
  EXPECT_EQ(p[3], ~0x0F);
}

TEST(BackendKernelClass, BitwiseNotClassMatchesReferenceUint8) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseNot kernel{ctx};
  Tensor x = Tensor::FromUint8("", {3}, {0x00, 0xF0, 0xAA});
  Tensor z = kernel(x);
  ASSERT_EQ(z.data_type, static_cast<int32_t>(onnx_backend_test::DataType::UINT8));
  EXPECT_EQ(z.data[0], static_cast<uint8_t>(0xFF));
  EXPECT_EQ(z.data[1], static_cast<uint8_t>(0x0F));
  EXPECT_EQ(z.data[2], static_cast<uint8_t>(0x55));
}

TEST(BackendKernelClass, BitwiseNotInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseNot kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2}, {0, 0xFF});
  Tensor z("", onnx_backend_test::DataType::INT32, {2}, std::vector<uint8_t>(2 * sizeof(int32_t)));
  kernel(x, z);
  const int32_t *p = reinterpret_cast<const int32_t *>(z.data.data());
  EXPECT_EQ(p[0], ~0);
  EXPECT_EQ(p[1], ~0xFF);
}

TEST(BackendKernelClass, BitwiseNotRejectsBoolTensors) {
  const KernelContext ctx{DefaultOpset(18)};
  BitwiseNot kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {2}, {1, 0});
  EXPECT_THROW((void)kernel(x), std::invalid_argument);
}

TEST(BackendKernelClass, NotClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(1)};
  Not not_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {4}, {1, 0, 1, 0});
  Tensor y = not_kernel(x);
  ASSERT_EQ(y.element_count(), 4);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(y.data[0], 0u);
  EXPECT_EQ(y.data[1], 1u);
  EXPECT_EQ(y.data[2], 0u);
  EXPECT_EQ(y.data[3], 1u);
}

TEST(BackendKernelClass, NotInPlaceWritesToPreallocatedOutput) {
  const KernelContext ctx{DefaultOpset(1)};
  Not not_kernel{ctx};
  Tensor x("", onnx_backend_test::DataType::BOOL, {3}, {0, 1, 0});
  Tensor y("", onnx_backend_test::DataType::BOOL, {3}, std::vector<uint8_t>(3));
  not_kernel(x, y);
  EXPECT_EQ(y.data[0], 1u);
  EXPECT_EQ(y.data[1], 0u);
  EXPECT_EQ(y.data[2], 1u);
}

TEST(BackendKernelClass, NotRejectsNonBoolTensors) {
  const KernelContext ctx{DefaultOpset(1)};
  Not not_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2}, {0, 1});
  EXPECT_THROW((void)not_kernel(x), std::invalid_argument);
}

TEST(BackendKernelClass, IsNaNClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(20)};
  IsNaN isnan_kernel{ctx};
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();
  Tensor x = Tensor::FromFloat("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});
  Tensor y = isnan_kernel(x);
  ASSERT_EQ(y.element_count(), 6);
  EXPECT_EQ(y.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(y.data[0], 0u);
  EXPECT_EQ(y.data[1], 1u);
  EXPECT_EQ(y.data[2], 0u);
  EXPECT_EQ(y.data[3], 0u);
  EXPECT_EQ(y.data[4], 0u);
  EXPECT_EQ(y.data[5], 0u);
}

TEST(BackendKernelClass, IsNaNRejectsNonFloatTensors) {
  const KernelContext ctx{DefaultOpset(20)};
  IsNaN isnan_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2}, {0, 1});
  EXPECT_THROW((void)isnan_kernel(x), std::invalid_argument);
}

TEST(BackendKernelClass, IsInfClassMatchesReference) {
  const KernelContext ctx{DefaultOpset(20)};
  IsInf isinf_kernel{ctx};
  const float nan_v = std::numeric_limits<float>::quiet_NaN();
  const float inf_v = std::numeric_limits<float>::infinity();
  Tensor x = Tensor::FromFloat("", {6}, {-1.2f, nan_v, inf_v, 2.8f, -inf_v, inf_v});

  Tensor both = isinf_kernel(x);
  ASSERT_EQ(both.element_count(), 6);
  EXPECT_EQ(both.data_type, static_cast<int32_t>(onnx_backend_test::DataType::BOOL));
  EXPECT_EQ(both.data[0], 0u);
  EXPECT_EQ(both.data[1], 0u);
  EXPECT_EQ(both.data[2], 1u);
  EXPECT_EQ(both.data[3], 0u);
  EXPECT_EQ(both.data[4], 1u);
  EXPECT_EQ(both.data[5], 1u);

  Tensor only_pos = isinf_kernel(x, /*detect_positive=*/1, /*detect_negative=*/0);
  EXPECT_EQ(only_pos.data[2], 1u);
  EXPECT_EQ(only_pos.data[4], 0u);
  EXPECT_EQ(only_pos.data[5], 1u);

  Tensor only_neg = isinf_kernel(x, /*detect_positive=*/0, /*detect_negative=*/1);
  EXPECT_EQ(only_neg.data[2], 0u);
  EXPECT_EQ(only_neg.data[4], 1u);
  EXPECT_EQ(only_neg.data[5], 0u);
}

TEST(BackendKernelClass, IsInfRejectsNonFloatTensors) {
  const KernelContext ctx{DefaultOpset(20)};
  IsInf isinf_kernel{ctx};
  Tensor x = Tensor::FromInt32("", {2}, {0, 1});
  EXPECT_THROW((void)isinf_kernel(x), std::invalid_argument);
}

} // namespace Test

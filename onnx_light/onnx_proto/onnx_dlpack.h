// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dlpack/dlpack.h"
#include "onnx.h"
#include <stdexcept>
#include <vector>

namespace ONNX_LIGHT_NAMESPACE {

/** Reports storage that cannot be exported without copying or byte swapping. */
class ONNX_LIGHT_PROTO_API DLPackBufferError : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

/** Stores an independent snapshot of validated DLPack element and shape metadata. */
struct DLPackMetadata {
  DLDataType dtype;
  std::vector<int64_t> shape;
};

/** Returns the exact DLPack dtype for a whole-byte ONNX element type. */
ONNX_LIGHT_PROTO_API DLDataType DLPackDataTypeFromOnnx(int32_t data_type,
                                                       const char *producer = "Tensor");

/**
 * Validates dense CPU raw_data and returns a metadata snapshot without changing the source.
 * Throws std::invalid_argument for unsupported types, segments, absent/mismatched payloads,
 * negative dimensions, rank above 128, or shape/byte-count overflow. Throws DLPackBufferError
 * for misaligned storage or multi-byte data on a big-endian host.
 * Does not establish ownership of the payload.
 */
ONNX_LIGHT_PROTO_API DLPackMetadata ValidateDLPack(const TensorProto &tensor);

/**
 * Transfers raw_data without copying into an independently owned legacy DLPack descriptor.
 * The caller transfers the descriptor to one consumer or calls result->deleter(result) once.
 * Shape and dtype are snapshotted; neither the descriptor nor its deleter accesses the source.
 * Owned/aligned-owned storage is moved; borrowed storage must have a shared lifetime token
 * (including null-valued tokens with a control block). Shared owners retain exactly-once
 * cleanup. Consumers must treat the payload as read-only, even though the legacy ABI has no
 * read-only flag. Transfer of owned storage with active export leases is rejected;
 * release those consumers first. Borrowed exports retain their shared lifetime owner.
 *
 * Validation and allocations complete before detachment: failure leaves the source intact.
 * On success raw_data has a null pointer, zero size, and false presence; all other fields are
 * unchanged. The descriptor and its storage survive destruction or reuse of the source.
 * Explicitly present empty payloads are accepted and exported with a null DLTensor::data.
 *
 * @warning This is a DESTRUCTIVE transfer, unlike Python TensorProto.__dlpack__.
 * It immediately removes the source payload. The tensor and any initializer/model containing
 * it cannot use that payload until raw_data is reassigned. Any user-supplied storage deleter
 * must itself be independent of the source TensorProto; it may run after source destruction.
 */
ONNX_LIGHT_PROTO_API DLManagedTensor *ReleaseDLPack(TensorProto &tensor);

} // namespace ONNX_LIGHT_NAMESPACE

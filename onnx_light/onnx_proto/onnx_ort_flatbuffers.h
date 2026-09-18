#pragma once

#include "onnx_light_helpers.h"
#include <string>

namespace ONNX_LIGHT_NAMESPACE {

class ModelProto;
struct SerializeOptions;
struct ParseOptions;
namespace utils {
class BinaryStream;
}

/**
 * Serializes a model into a native ORTM FlatBuffer.
 *
 * Emits ORT format version 4, which full ONNX Runtime builds upgrade using their
 * operator schemas. Minimal ONNX Runtime builds are not supported. Tensor data
 * is embedded; external tensors must already have their payload loaded.
 * Aligns numeric tensor payload offsets to at least eight bytes, or to the
 * larger power-of-two alignment requested by the serialization options.
 * Applies serialization callbacks without changing the supplied model.
 * Infers intermediate element types for common standard operators; other
 * operators require explicit graph value_info. Normalizes Constant nodes into
 * initializers, as required by the ORT loader.
 *
 * Throws std::invalid_argument for unsupported constructs or invalid input and
 * std::length_error when the output exceeds the configured limit or the
 * FlatBuffers signed 32-bit size limit. Model-local functions, optional/sparse/
 * opaque types, complex tensors, sparse initializers/attributes, GRAPHS
 * attributes, and encoded initializers are unsupported.
 * Tensor element types newer than FLOAT8E5M2FNUZ (20), including UINT4, INT4,
 * FLOAT4E2M1, FLOAT8E8M0, UINT2, and INT2, are unsupported.
 * Preserves model-level metadata_props but rejects graph, node, tensor, and
 * value-info metadata_props, including in nested graphs and tensor attributes.
 *
 * Returns:
 *     The complete FlatBuffer, including its ORTM file identifier.
 */
ONNX_LIGHT_PROTO_API std::string SerializeModelToOrtFlatbuffers(const ModelProto &model,
                                                                const SerializeOptions &options);

/**
 * Serializes a model into an output string.
 *
 * Returns false and clears the output when the configured size limit is
 * exceeded. Throws for other invalid or unsupported input.
 */
ONNX_LIGHT_PROTO_API bool SerializeModelToOrtFlatbuffers(const ModelProto &model,
                                                         std::string &output,
                                                         const SerializeOptions &options);

/**
 * Parses an ORTM FlatBuffer into a model, replacing it only after successful decoding.
 *
 * Accepts ORT versions 4, 5 and 6, including shared forward/backward vtables.
 * Copies tensor payloads into owned, optionally aligned storage even when no_copy
 * is requested. Decoding is serial regardless of num_threads; no external I/O
 * scheduling or tracing is performed. Enforces recursion and tensor allocation
 * limits before materializing fields. skip_raw_data omits numeric raw payloads;
 * string tensor values remain available. Applies raw-data and node callbacks
 * after graph reconstruction.
 *
 * Rejects external tensor offsets/weights streams, fused nodes, saved runtime
 * optimizations, unsupported schema extensions, and external-I/O policy options.
 * Kernel resolver and placement metadata are checked but not retained in ONNX.
 * Custom-domain operators are retained and require matching runtime kernels.
 * Throws RuntimeError-compatible exceptions for malformed input and
 * ParseLimitExceeded for configured resource limits.
 */
ONNX_LIGHT_PROTO_API void
ParseModelFromOrtFlatbuffers(ModelProto &model, utils::BinaryStream &stream, ParseOptions &options);

} // namespace ONNX_LIGHT_NAMESPACE

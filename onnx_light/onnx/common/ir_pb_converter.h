// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once
#include <memory>
#include <string>

#include "onnx/common/common.h"
#include "onnx/common/ir.h"
#include "onnx/onnx-data.pb.h"

namespace ONNX_LIGHT_NAMESPACE {

/**
 * Exception type used by the protobuf <-> IR conversion utilities.
 *
 * The converter progressively appends context while unwinding nested conversion
 * calls so the final error message points to the failing model location.
 */
class ConvertError final : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;

  explicit ConvertError(const std::string &message) : std::runtime_error(message) {}

  const char *what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string &context) {
    expanded_message_ = MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

private:
  std::string expanded_message_;
};

#ifndef fail_convert
/// Throws ConvertError with a formatted message.
#define fail_convert(...) ONNX_THROW_EX(ConvertError(MakeString(__VA_ARGS__)))
#endif // fail_convert

/**
 * Serializes an IR graph into a ModelProto.
 *
 * The graph content is appended to @p p_m as a new graph entry and its opset
 * imports are synchronized with graph opset versions.
 *
 * @param p_m Destination model proto.
 * @param g Source IR graph.
 */
void ExportModelProto(ModelProto *p_m, const std::shared_ptr<Graph> &g);

/**
 * Converts the first graph in a ModelProto into the internal IR graph.
 *
 * @return Imported graph represented as an internal IR graph.
 * @throws ConvertError When protobuf content cannot be represented in
 * the internal IR.
 */
std::unique_ptr<Graph> ImportModelProto(const ModelProto &mp);

/**
 * Produces a metadata-preserving model shell used by conversion pipelines.
 *
 * This helper copies global model-level fields while leaving graphs empty so
 * conversion passes can repopulate the graph with transformed content.
 *
 * @return ModelProto containing copied model-level metadata and no graph data.
 */
ONNX_API ModelProto PrepareOutput(const ModelProto &mp_in);

/**
 * Validates that graph conversion returned a non-null graph pointer.
 *
 * Triggers ONNX_ASSERTM when @p g is null.
 */
void assertNonNull(const std::shared_ptr<Graph> &g);
} // namespace ONNX_LIGHT_NAMESPACE

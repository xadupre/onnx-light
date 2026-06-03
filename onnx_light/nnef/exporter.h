// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "nnef/tensor_io.h"
#include "onnx_light_helpers.h"
#include "onnx_proto/onnx.h"

namespace ONNX_LIGHT_NAMESPACE {
namespace nnef {

/** Raised when an ONNX construct cannot be expressed in NNEF. */
class NNEFExportError : public std::runtime_error {
public:
  using std::runtime_error::runtime_error;
};

/** Variant of attribute values supported by NNEF converters. */
using AttributeValue =
    std::variant<std::monostate, int64_t, double, std::string, std::vector<int64_t>,
                 std::vector<double>, std::vector<std::string>, NNEFTensor>;

/** In-memory representation of a NNEF graph ready to be serialised. */
struct NNEFGraph {
  std::string name;
  std::vector<std::string> inputs;
  std::vector<std::string> outputs;
  std::vector<std::string> statements;
  /** Ordered, deterministic list of (nnef_name, tensor) pairs. */
  std::vector<std::pair<std::string, NNEFTensor>> initializers;
  int version_major = 1;
  int version_minor = 0;
  std::vector<std::string> extensions;

  /** Returns the textual ``graph.nnef`` representation. */
  std::string ToText() const;
};

class ExportContext;

/** Signature of an operator converter. */
using ConverterFn = std::function<void(
    ExportContext &, const NodeProto &, const std::map<std::string, AttributeValue> &,
    const std::vector<std::string> & /*inputs*/, const std::vector<std::string> & /*outputs*/)>;

/** Mutable state threaded through op-converter invocations. */
class ExportContext {
public:
  ExportContext();

  /** Returns the unique NNEF identifier mapped from ``onnx_name``. */
  std::string MapName(const std::string &onnx_name);

  /** Returns a fresh NNEF identifier with the given base. */
  std::string MakeTemp(const std::string &base = "t");

  /** Appends a NNEF statement (terminating ``;`` included). */
  void AddStatement(const std::string &stmt) { statements_.push_back(stmt); }

  /** Returns the initializer tensor for ``onnx_name`` or nullptr. */
  const NNEFTensor *GetInitializer(const std::string &onnx_name) const;

  // Internal mutators used by ExportToNNEF.
  void SetInitializer(const std::string &onnx_name, NNEFTensor tensor) {
    initializers_by_onnx_[onnx_name] = std::move(tensor);
  }
  const std::vector<std::string> &Statements() const { return statements_; }
  std::vector<std::string> &MutableStatements() { return statements_; }

private:
  std::map<std::string, std::string> name_map_;
  std::map<std::string, NNEFTensor> initializers_by_onnx_;
  std::vector<std::string> statements_;
  int tmp_counter_ = 0;
};

/** Registers (or overrides) the converter used for an ONNX op type. */
void RegisterOpConverter(const std::string &op_type, ConverterFn converter);

/** Removes ``op_type`` from the registry; returns true if removed. */
bool UnregisterOpConverter(const std::string &op_type);

/** Returns the sorted list of ONNX op types with a registered converter. */
std::vector<std::string> SupportedOps();

/** Returns true if ``op_type`` has a registered converter. */
bool HasOpConverter(const std::string &op_type);

/** Returns the converter for ``op_type`` (must exist). */
const ConverterFn &GetOpConverter(const std::string &op_type);

/** Builds an in-memory NNEF graph from an ONNX ModelProto. */
NNEFGraph ExportToNNEF(const ModelProto &model, const std::string &graph_name = "");

/** Returns the ``graph.nnef`` text for ``model``. */
std::string ToNNEFText(const ModelProto &model, const std::string &graph_name = "");

/** Writes ``model`` to ``out_dir`` as a NNEF directory and returns its absolute path. */
std::string SaveNNEF(const ModelProto &model, const std::string &out_dir,
                     const std::string &graph_name = "", bool overwrite = true);

// ---------------------------------------------------------------------------
// Helpers used internally and by the bindings (exposed for testing).
// ---------------------------------------------------------------------------

/** Converts an ONNX TensorProto to a dense NNEFTensor (decoding the typed
 *  data fields when ``raw_data`` is empty). */
NNEFTensor TensorProtoToNNEFTensor(const TensorProto &tensor);

/** Returns the NNEF text representation of an attribute value. */
std::string FormatValue(const AttributeValue &value);

} // namespace nnef
} // namespace ONNX_LIGHT_NAMESPACE

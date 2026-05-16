// ONNX operator schema registration using the full operator definition files.
// All schemas are registered from the operator set classes defined in operator_sets*.h,
// which in turn call GetOpSchema<> specialisations from the per-domain defs .cc files.
// This ensures every registered schema carries complete metadata: inputs, outputs,
// type constraints, doc strings, support level, and determinism flags.

#ifdef ONNX_ML
#include "onnx/defs/operator_sets_ml.h"
#endif
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_preview.h"
#include "onnx/defs/operator_sets_training.h"
#include "onnx/defs/schema.h"

namespace ONNX_LIGHT_NAMESPACE {

void RegisterAllOnnxOperatorSchemas() {
  // Register all ai.onnx operator schemas across every opset version.
  // fail_duplicate_schema=false makes this function safe to call multiple times.
  RegisterOnnxOperatorSetSchema(0, false);

#ifdef ONNX_ML
  // Register ai.onnx.ml operator schemas.
  RegisterOpSetSchema<OpSet_OnnxML_ver1>(0, false);
  RegisterOpSetSchema<OpSet_OnnxML_ver2>(0, false);
  RegisterOpSetSchema<OpSet_OnnxML_ver3>(0, false);
  RegisterOpSetSchema<OpSet_OnnxML_ver4>(0, false);
  RegisterOpSetSchema<OpSet_OnnxML_ver5>(0, false);
#endif

  // Register ai.onnx.preview and ai.onnx.preview.training operator schemas.
  RegisterOpSetSchema<OpSet_OnnxPreview_ver1>(0, false);
}

} // namespace ONNX_LIGHT_NAMESPACE

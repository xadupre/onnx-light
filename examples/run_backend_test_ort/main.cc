/**
 * main.cc -- Standalone example: run every C++-generated backend test case
 * through a downloaded onnxruntime release and verify the runtime outputs
 * match the expected outputs declared by the test case.
 *
 * This is the C++ counterpart of ``unittests/backend/test_backend_with_onnxruntime.py``:
 * it enumerates the same node test cases (Abs, Add, If, ReduceSum, ...),
 * serializes each ``ModelProto`` to bytes via the lib_onnx_proto C++ API,
 * loads it into an ``Ort::Session`` (CPU execution provider), runs the model
 * with the expected inputs, and compares each produced output against the
 * expected output with the per-case ``rtol`` / ``atol`` tolerances.
 *
 * See CMakeLists.txt for build instructions, and ``build.sh`` /
 * ``build.bat`` for one-shot scripts that also download the onnxruntime
 * release archive.
 */

#include "onnx_kernels/test_case.h"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace btest = ONNX_LIGHT_NAMESPACE::onnx_backend_test;
using ONNX_LIGHT_NAMESPACE::TensorProto;

namespace {

/// Test cases that ORT cannot run as-is. Mirrors the Python ORT_EXCLUDE_REGEX
/// list in ``unittests/backend/test_backend_with_onnxruntime.py``, with a
/// few additional exclusions that are specific to the simplified C++ harness
/// in this example:
///   * ``test_cc_roialign_max`` -- ORT's RoiAlign max-mode implementation does
///     not match the ONNX reference.
///   * ``test_cc_flex_attention_*`` -- ORT does not register the
///     ``ai.onnx.preview`` domain (FlexAttention is unknown).
///   * ``test_cc_adam_*`` -- ORT does not register the
///     ``ai.onnx.preview.training`` domain (Adam is unknown).
///   * ``test_cc_acos`` / ``test_cc_acosh`` -- these are stamped with opset 22
///     which is past the maximum ai.onnx opset (21) supported by the
///     onnxruntime 1.19.x release used as the default download. Bump
///     ``ONNXRUNTIME_VERSION`` (e.g. 1.20.x or newer) and remove from this
///     list to run them.
///   * ``test_cc_sequence_construct*`` -- the model output is a sequence of
///     tensors, not a tensor; this example's output comparator only knows
///     how to compare ``Ort::Value`` tensors.
const std::vector<std::regex> &OrtExcludeRegex() {
  static const std::vector<std::regex> patterns = {
      std::regex(R"(^test_cc_roialign_max$)"), std::regex(R"(^test_cc_flex_attention_)"),
      std::regex(R"(^test_cc_adam_)"),         std::regex(R"(^test_cc_acos$)"),
      std::regex(R"(^test_cc_acosh$)"),        std::regex(R"(^test_cc_sequence_construct)"),
  };
  return patterns;
}

bool IsExcluded(const std::string &name) {
  for (const auto &re : OrtExcludeRegex()) {
    if (std::regex_search(name, re)) {
      return true;
    }
  }
  return false;
}

/// Maps a ``TensorProto::DataType`` integer value to the equivalent ORT
/// element type enum. The numeric values match (both follow ONNX), but going
/// through an explicit switch keeps the dependency direction clean.
ONNXTensorElementDataType ToOrtElementType(int32_t dtype) {
  switch (dtype) {
  case TensorProto::DataType::FLOAT:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  case TensorProto::DataType::UINT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  case TensorProto::DataType::INT8:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  case TensorProto::DataType::UINT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  case TensorProto::DataType::INT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  case TensorProto::DataType::INT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  case TensorProto::DataType::INT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  case TensorProto::DataType::STRING:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  case TensorProto::DataType::BOOL:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
  case TensorProto::DataType::FLOAT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  case TensorProto::DataType::DOUBLE:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  case TensorProto::DataType::UINT32:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
  case TensorProto::DataType::UINT64:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
  case TensorProto::DataType::BFLOAT16:
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  default:
    throw std::runtime_error("Unsupported TensorProto::DataType for ORT mapping: " +
                             std::to_string(dtype));
  }
}

/// Builds an Ort::Value for the given backend-test ``Tensor``. For string
/// tensors a fresh allocator-backed value is created and the strings are
/// copied in; for all other types a non-owning view over ``tensor.data`` is
/// returned (the caller must keep ``tensor`` alive until Run() returns).
Ort::Value MakeOrtInput(const btest::Tensor &tensor, Ort::MemoryInfo &mem_info,
                        Ort::AllocatorWithDefaultOptions &allocator) {
  const ONNXTensorElementDataType ort_type = ToOrtElementType(tensor.data_type);
  if (tensor.data_type == TensorProto::DataType::STRING) {
    Ort::Value v = Ort::Value::CreateTensor(allocator, tensor.shape.data(),
                                            static_cast<size_t>(tensor.shape.size()), ort_type);
    std::vector<const char *> c_strs;
    c_strs.reserve(tensor.string_data.size());
    for (const auto &s : tensor.string_data) {
      c_strs.push_back(s.c_str());
    }
    v.FillStringTensor(c_strs.data(), c_strs.size());
    return v;
  }
  // Numeric / bool: ORT will read directly from tensor.data; the buffer
  // remains owned by the backend-test Tensor.
  return Ort::Value::CreateTensor(mem_info, const_cast<uint8_t *>(tensor.data.data()),
                                  tensor.data.size(), tensor.shape.data(),
                                  static_cast<size_t>(tensor.shape.size()), ort_type);
}

/// Returns shape-product element count for an Ort tensor.
int64_t OrtElementCount(const Ort::Value &v) {
  auto info = v.GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  int64_t n = 1;
  for (int64_t d : shape) {
    n *= (d >= 0 ? d : 0);
  }
  return n;
}

/// Compares one expected output to the actual ORT output using rtol/atol for
/// floating-point types, exact byte equality for integer/bool, and exact
/// string equality for STRING tensors. Returns an empty string on success or
/// a human-readable error description on mismatch.
std::string CompareOutputs(const btest::Tensor &expected, const Ort::Value &actual, double rtol,
                           double atol) {
  if (!actual.IsTensor()) {
    return "ORT output is not a tensor";
  }
  auto info = actual.GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType actual_type = info.GetElementType();
  if (actual_type != ToOrtElementType(expected.data_type)) {
    return "element type mismatch (expected " + std::to_string(expected.data_type) + ")";
  }
  const int64_t n_expected = expected.element_count();
  const int64_t n_actual = OrtElementCount(actual);
  if (n_expected != n_actual) {
    return "element count mismatch (expected " + std::to_string(n_expected) + ", got " +
           std::to_string(n_actual) + ")";
  }

  switch (expected.data_type) {
  case TensorProto::DataType::FLOAT: {
    const float *e = expected.As<float>();
    const float *a = actual.GetTensorData<float>();
    for (int64_t i = 0; i < n_expected; ++i) {
      const double diff = std::fabs(static_cast<double>(a[i]) - static_cast<double>(e[i]));
      const double tol = atol + rtol * std::fabs(static_cast<double>(e[i]));
      if (!(diff <= tol) ||
          std::isnan(static_cast<double>(a[i])) != std::isnan(static_cast<double>(e[i]))) {
        return "FLOAT mismatch at index " + std::to_string(i);
      }
    }
    return {};
  }
  case TensorProto::DataType::DOUBLE: {
    const double *e = expected.As<double>();
    const double *a = actual.GetTensorData<double>();
    for (int64_t i = 0; i < n_expected; ++i) {
      const double diff = std::fabs(a[i] - e[i]);
      const double tol = atol + rtol * std::fabs(e[i]);
      if (!(diff <= tol) || std::isnan(a[i]) != std::isnan(e[i])) {
        return "DOUBLE mismatch at index " + std::to_string(i);
      }
    }
    return {};
  }
  case TensorProto::DataType::STRING: {
    auto allocator = Ort::AllocatorWithDefaultOptions();
    const auto &es = expected.AsStrings();
    for (size_t i = 0; i < static_cast<size_t>(n_expected); ++i) {
      const size_t len = actual.GetStringTensorElementLength(i);
      std::string s(len, '\0');
      if (len > 0) {
        actual.GetStringTensorElement(len, i, s.data());
      }
      if (s != es[i]) {
        return "STRING mismatch at index " + std::to_string(i);
      }
    }
    return {};
  }
  default: {
    // Integer / bool / fp16 / bfloat16: exact byte equality. Tolerances do
    // not apply to these element types in the reference Python backend
    // either.
    const size_t bytes = expected.data.size();
    const void *a = actual.GetTensorRawData();
    if (std::memcmp(a, expected.data.data(), bytes) != 0) {
      return "byte mismatch (dtype " + std::to_string(expected.data_type) + ")";
    }
    return {};
  }
  }
}

/// Runs one TestCase through an Ort::Session. Returns a (passed, message)
/// pair: ``passed == true`` on success, ``false`` on a verification failure
/// or a recoverable ORT exception (e.g. unknown op).
std::pair<bool, std::string> RunOneCase(Ort::Env &env, const btest::TestCase &tc) {
  // Serialize the C++-generated ModelProto to bytes via lib_onnx_proto, then
  // feed those bytes to ORT.
  //
  // The C++ backend test library stamps models with the latest ONNX IR
  // version (13 at time of writing), but the downloaded onnxruntime CPU
  // release pins a maximum supported IR version (10 in 1.19.x). The IR
  // version is essentially the serialization format identifier -- it does
  // not change the semantics of the embedded opset versions -- so lowering
  // it to ``kOrtMaxIrVersion`` is a safe transformation for these single-
  // node test models. If the bundled release supports a higher IR version,
  // override ``ONNX_LIGHT_RUN_BACKEND_TEST_ORT_IR_VERSION`` at compile time
  // via ``-D``.
  static constexpr int64_t kOrtMaxIrVersion =
#ifdef ONNX_LIGHT_RUN_BACKEND_TEST_ORT_IR_VERSION
      ONNX_LIGHT_RUN_BACKEND_TEST_ORT_IR_VERSION
#else
      10
#endif
      ;

  ONNX_LIGHT_NAMESPACE::ModelProto &mutable_model = const_cast<btest::TestCase &>(tc).model;
  if (mutable_model.ref_ir_version() > kOrtMaxIrVersion) {
    mutable_model.set_ir_version(kOrtMaxIrVersion);
  }
  std::string model_bytes;
  mutable_model.SerializeToString(model_bytes);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

  Ort::Session session(env, model_bytes.data(), model_bytes.size(), session_options);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  if (tc.data_sets.empty()) {
    return {false, "no data_sets"};
  }
  for (size_t ds_idx = 0; ds_idx < tc.data_sets.size(); ++ds_idx) {
    const auto &ds = tc.data_sets[ds_idx];

    std::vector<Ort::Value> input_values;
    std::vector<const char *> input_names;
    input_values.reserve(ds.inputs.size());
    input_names.reserve(ds.inputs.size());
    for (const auto &t : ds.inputs) {
      input_values.emplace_back(MakeOrtInput(t, mem_info, allocator));
      input_names.push_back(t.name.c_str());
    }

    std::vector<const char *> output_names;
    output_names.reserve(ds.outputs.size());
    for (const auto &t : ds.outputs) {
      output_names.push_back(t.name.c_str());
    }

    std::vector<Ort::Value> outputs =
        session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_values.data(),
                    input_values.size(), output_names.data(), output_names.size());

    if (outputs.size() != ds.outputs.size()) {
      return {false, "data_set " + std::to_string(ds_idx) + ": expected " +
                         std::to_string(ds.outputs.size()) + " outputs, ORT returned " +
                         std::to_string(outputs.size())};
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      const std::string err = CompareOutputs(ds.outputs[i], outputs[i], tc.rtol, tc.atol);
      if (!err.empty()) {
        return {false, "data_set " + std::to_string(ds_idx) + " output[" + std::to_string(i) +
                           "] (" + ds.outputs[i].name + "): " + err};
      }
    }
  }
  return {true, {}};
}

} // namespace

int main(int argc, char **argv) {
  // Optional CLI: a single positional regex restricts which test cases run.
  // Defaults to all cases.
  std::regex name_filter;
  bool has_filter = false;
  if (argc > 1) {
    name_filter = std::regex(argv[1]);
    has_filter = true;
  }

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_light_backend_test_ort");

  const auto cases = btest::CollectTestCases();
  int passed = 0;
  int failed = 0;
  int skipped = 0;
  std::vector<std::string> failures;

  for (const auto &tc : cases) {
    if (has_filter && !std::regex_search(tc.name, name_filter)) {
      continue;
    }
    if (IsExcluded(tc.name)) {
      std::cout << "SKIP " << tc.name << " (excluded for ORT)\n";
      ++skipped;
      continue;
    }
    try {
      auto [ok, msg] = RunOneCase(env, tc);
      if (ok) {
        std::cout << "PASS " << tc.name << "\n";
        ++passed;
      } else {
        std::cout << "FAIL " << tc.name << " : " << msg << "\n";
        failures.push_back(tc.name + " : " + msg);
        ++failed;
      }
    } catch (const Ort::Exception &e) {
      std::cout << "FAIL " << tc.name << " : ORT exception: " << e.what() << "\n";
      failures.push_back(std::string(tc.name) + " : ORT exception: " + e.what());
      ++failed;
    } catch (const std::exception &e) {
      std::cout << "FAIL " << tc.name << " : exception: " << e.what() << "\n";
      failures.push_back(std::string(tc.name) + " : exception: " + e.what());
      ++failed;
    }
  }

  std::cout << "\n=== Summary ===\n";
  std::cout << "  total   : " << (passed + failed + skipped) << "\n";
  std::cout << "  passed  : " << passed << "\n";
  std::cout << "  failed  : " << failed << "\n";
  std::cout << "  skipped : " << skipped << "\n";

  if (failed != 0) {
    std::cout << "\nFailures:\n";
    for (const auto &f : failures) {
      std::cout << "  - " << f << "\n";
    }
    return 1;
  }
  if (passed == 0 && skipped == 0) {
    std::cerr << "No backend test cases were run (filter matched nothing?).\n";
    return 2;
  }
  return 0;
}

/**
 * main.cc — Standalone example: load the C++-generated Add backend test node
 * case and verify a tiny runtime Add implementation against it.
 *
 * Demonstrates that ``lib_onnx_backend_test`` is self-contained: linking with
 * only ``onnx_light::onnx_backend_test`` (which transitively pulls in
 * ``onnx_proto``) is sufficient to enumerate backend test cases, inspect the
 * generated ``ModelProto``, read its inputs/outputs, run a runtime function
 * against them, and check the results match the expected outputs.
 *
 * See CMakeLists.txt for build instructions.
 */

#include "onnx_backend_test/test_case.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace btest = ONNX_LIGHT_NAMESPACE::onnx_backend_test;

namespace {

/// Minimal runtime "Add" implementation: element-wise sum of two float
/// tensors with NumPy-style broadcasting limited to two cases — equal shapes
/// or scalar broadcast (one operand has element_count() == 1). This is just
/// enough to exercise the C++-generated Add backend test cases.
std::vector<float> RuntimeAdd(const btest::Tensor &x, const btest::Tensor &y) {
  if (x.data_type != ONNX_LIGHT_NAMESPACE::TensorProto::DataType::FLOAT ||
      y.data_type != ONNX_LIGHT_NAMESPACE::TensorProto::DataType::FLOAT) {
    throw std::runtime_error("RuntimeAdd only supports FLOAT tensors.");
  }
  const float *px = x.AsFloat();
  const float *py = y.AsFloat();
  const int64_t nx = x.element_count();
  const int64_t ny = y.element_count();
  const int64_t n = nx >= ny ? nx : ny;
  if (!(nx == ny || nx == 1 || ny == 1)) {
    throw std::runtime_error(
        "RuntimeAdd only supports equal shapes or scalar broadcasting in this example.");
  }
  std::vector<float> z(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const float a = nx == 1 ? px[0] : px[i];
    const float b = ny == 1 ? py[0] : py[i];
    z[static_cast<size_t>(i)] = a + b;
  }
  return z;
}

/// Returns true if every element of ``actual`` is close to ``expected`` within
/// the absolute tolerance ``atol``.
bool AllClose(const std::vector<float> &actual, const float *expected, int64_t n, float atol) {
  for (int64_t i = 0; i < n; ++i) {
    if (std::fabs(actual[static_cast<size_t>(i)] - expected[i]) > atol) {
      return false;
    }
  }
  return true;
}

/// Runs one runtime check against the given test case and prints a summary.
/// Returns 0 on success and 1 on failure.
int CheckOneCase(const btest::TestCase &tc) {
  std::cout << "Verifying test case: " << tc.name << "\n";
  std::cout << "  model nodes : " << tc.model.ref_graph().ref_node().size() << "\n";
  std::cout << "  op_type     : "
            << std::string(tc.model.ref_graph().ref_node()[0].ref_op_type().data(),
                           tc.model.ref_graph().ref_node()[0].ref_op_type().size())
            << "\n";
  if (tc.data_sets.empty()) {
    std::cerr << "  ERROR: no data sets\n";
    return 1;
  }
  const auto &ds = tc.data_sets[0];
  if (ds.inputs.size() != 2 || ds.outputs.size() != 1) {
    std::cerr << "  ERROR: expected 2 inputs and 1 output for Add, got " << ds.inputs.size()
              << " and " << ds.outputs.size() << "\n";
    return 1;
  }
  std::vector<float> actual = RuntimeAdd(ds.inputs[0], ds.inputs[1]);
  if (static_cast<int64_t>(actual.size()) != ds.outputs[0].element_count()) {
    std::cerr << "  ERROR: actual output element count (" << actual.size()
              << ") does not match expected (" << ds.outputs[0].element_count() << ")\n";
    return 1;
  }
  if (!AllClose(actual, ds.outputs[0].AsFloat(), ds.outputs[0].element_count(),
                /*atol=*/1e-6f)) {
    std::cerr << "  ERROR: runtime Add output does not match expected output.\n";
    return 1;
  }
  std::cout << "  PASS (" << actual.size() << " element(s))\n";
  return 0;
}

} // namespace

int main() {
  int failures = 0;
  int checked = 0;
  try {
    const auto cases = btest::CollectTestCases();
    for (const auto &tc : cases) {
      if (tc.name == "test_cc_add" || tc.name == "test_cc_add_bcast") {
        failures += CheckOneCase(tc);
        ++checked;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Unhandled exception: " << e.what() << "\n";
    return 2;
  }

  if (checked == 0) {
    std::cerr << "No Add test cases found in lib_onnx_backend_test.\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " / " << checked << " Add test case(s) failed.\n";
    return 1;
  }
  std::cout << "All " << checked << " Add test case(s) passed.\n";
  return 0;
}

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/test_case_registry.h"

#include <memory>
#include <stdexcept>
#include <utility>

namespace ONNX_LIGHT_NAMESPACE::core::backend_test {

namespace {

TestCasesCollectorFn WithRebuildFallbacks(TestCasesCollectorFn collector) {
  return [collector = std::move(collector)](std::vector<TestCase> &registry,
                                            const std::string &op_type, bool include_big,
                                            TestMode mode) {
    std::vector<TestCase> collected;
    collector(collected, op_type, include_big, mode);
    for (TestCase &test_case : collected) {
      const std::string name = test_case.name;
      test_case.set_rebuild([collector, name, op_type, include_big, mode](bool generate_outputs) {
        std::vector<TestCase> rebuilt_cases;
        collector(rebuilt_cases, op_type, include_big, mode);
        for (TestCase &rebuilt : rebuilt_cases) {
          if (rebuilt.name != name) {
            continue;
          }
          rebuilt.set_expected_outputs_generated(generate_outputs);
          rebuilt.Materialize();
          BuiltCase built = rebuilt.take_materialized();
          // Data sets may hold tensors borrowing bytes owned by the rebuilt
          // case's builder closure (``Constant`` and friends return non-owning
          // views over their captured value). Keep that case alive alongside
          // the payload so those views stay valid.
          built.retained = std::make_shared<TestCase>(std::move(rebuilt));
          return built;
        }
        throw std::runtime_error("Backend test collector did not rebuild case '" + name + "'.");
      });
      if (!test_case.build || test_case.materialized()) {
        test_case.unload();
      }
      registry.emplace_back(std::move(test_case));
    }
  };
}

std::vector<TestCasesCollectorFn> &GetRegisteredCollectorsMutable() {
  static std::vector<TestCasesCollectorFn> registry;
  return registry;
}

} // namespace

int RegisterTestCasesCollector(TestCasesCollectorFn fn) {
  GetRegisteredCollectorsMutable().push_back(WithRebuildFallbacks(std::move(fn)));
  return 0;
}

const std::vector<TestCasesCollectorFn> &GetRegisteredCollectors() {
  return GetRegisteredCollectorsMutable();
}

} // namespace ONNX_LIGHT_NAMESPACE::core::backend_test

// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_extensions/backend_test/cases/sequence/include_sequence_cases.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test {

void CollectSequenceTestCases(std::vector<TestCase> &registry, const std::string &op_type,
                              TestMode mode) {
  static const OpRegisterModeMap kEntries = {
      {"SequenceConstruct", &RegisterSequenceConstructCases},
      {"SequenceEmpty", &RegisterSequenceEmptyCases},
      {"ConcatFromSequence", &RegisterConcatFromSequenceCases},
      {"SequenceLength", &RegisterSequenceLengthCases},
      {"SequenceErase", &RegisterSequenceEraseCases},
      {"SequenceAt", &RegisterSequenceAtCases},
      {"SequenceInsert", &RegisterSequenceInsertCases},
      {"SequenceMap", &RegisterSequenceMapCases},
      {"SplitToSequence", &RegisterSplitToSequenceCases},
  };
  DispatchRegisterByOpType(registry, op_type, kEntries, mode);
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_backend_test

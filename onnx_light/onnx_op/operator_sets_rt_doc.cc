// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_op/operator_sets_rt_doc.h"

namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt {

/// Returns the documentation string for the DelayedInitializer operator.
std::string MakeDelayedInitializerDoc() {
  return R"DOC(
Defers materialization of a tensor stored in an external weights file.

In onnx-light, ``load_device`` must be either ``"cpu"`` or ``"file"`` and
``runtime_device`` must be ``"cpu"``. When ``load_device`` is ``"cpu"``, the
kernel loads the tensor bytes during kernel initialization and returns a CPU
copy at execution time. When ``load_device`` is ``"file"``, initialization does
not touch the file and execution loads the tensor bytes directly from
``filename`` at byte ``offset``.

The static output shape comes from the required ``shape`` attribute and the
element type comes from the required ``dtype`` attribute.
)DOC";
}

} // namespace ONNX_LIGHT_NAMESPACE::onnx_op::rt

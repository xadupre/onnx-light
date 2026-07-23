// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx_core/backend_test/io_data.h"

// ``IoData`` is a plain aggregate declared entirely in ``io_data.h``; it has no
// out-of-line members. This translation unit exists so the header is compiled
// on its own and stays self-contained (only ``simple_tensor.h`` /
// ``simple_map.h``), matching the ``expect.cc`` / ``test_case.cc`` split.

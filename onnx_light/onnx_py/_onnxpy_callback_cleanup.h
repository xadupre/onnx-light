// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/nanobind.h>

template <typename Clear> void ClearCallbackOwners(nanobind::handle owners, Clear clear) {
  while (nanobind::len(owners) != 0) {
    nanobind::list pending(owners);
    owners.attr("clear")();
    for (nanobind::handle owner : pending)
      clear(owner);
  }
}

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx_light.onnx as onnx
from .. import Base, expect


class BlackmanWindow(Base):
    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node("BlackmanWindow", inputs=["x"], outputs=["y"])
        size = np.int32(10)
        size_tensor = np.array(size, dtype=np.int32)
        a0 = 0.42
        a1 = -0.5
        a2 = 0.08
        y = np.array(a0, dtype=np.float32)
        y += a1 * np.cos(2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        y += a2 * np.cos(4 * np.pi * np.arange(0, size, 1, dtype=np.float32) / size)
        expect(
            node, inputs=[size_tensor], outputs=[y.astype(np.float32)], name="test_blackmanwindow"
        )

        node = onnx.helper.make_node("BlackmanWindow", inputs=["x"], outputs=["y"], periodic=0)
        y = np.array(a0, dtype=np.float32)
        y += a1 * np.cos(2 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
        y += a2 * np.cos(4 * np.pi * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
        expect(
            node,
            inputs=[size_tensor],
            outputs=[y.astype(np.float32)],
            name="test_blackmanwindow_symmetric",
        )

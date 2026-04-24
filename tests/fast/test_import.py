# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

import onnx_light


class TestImport(unittest.TestCase):
    def test_import(self):
        self.assertIsNotNone(onnx_light)


if __name__ == "__main__":
    unittest.main()

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest


class TestImport(unittest.TestCase):
    def test_import(self):
        import onnx_light

        self.assertIsNotNone(onnx_light)


if __name__ == "__main__":
    unittest.main()

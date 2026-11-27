# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Tests for the top-level command parser."""

import unittest

from onnx_light.ext_test_case import ExtTestCase
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile

import onnx_light.onnx.helper as oh
from onnx_light.__main__ import _build_parser, main
from onnx_light.onnx import TensorProto, save


class TestMainParser(ExtTestCase):
    def test_kernel_help_contains_examples(self):
        help_text = _build_parser()._subparsers._group_actions[0].choices["kernel"].format_help()
        self.assertIn("examples:", help_text)
        self.assertIn("python -m onnx_light kernel --list", help_text)
        self.assertIn("--parameter parallel.minimum_tasks=default,2,4", help_text)

    def test_backend_help_contains_examples(self):
        help_text = _build_parser()._subparsers._group_actions[0].choices["backend"].format_help()
        self.assertIn("examples:", help_text)
        self.assertIn("--output backend-not.xlsx", help_text)
        self.assertIn("--save-models backend-models", help_text)
        self.assertIn("--parameter parallel.minimum_elements=default,16384,32768", help_text)

    def test_rejects_removed_kernel_baseline_command(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            _build_parser().parse_args(["kernel-baseline"])

    def test_rejects_renamed_commands(self):
        for command in (
            "tune",
            "tune-kernels",
            "backend-test",
            "extract-layers",
            "layers",
            "layer",
        ):
            with (
                self.subTest(command=command),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                _build_parser().parse_args([command])

    def test_partition_requires_a_method(self):
        for arguments in (["partition"], ["partition", "unknown"]):
            with (
                self.subTest(arguments=arguments),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                _build_parser().parse_args(arguments)

    def test_partition_methods_share_the_existing_loading_and_output_options(self):
        parser = _build_parser()
        self.assertNotIn("layers", parser._subparsers._group_actions[0].choices)
        for method in ("layer", "mincut"):
            with self.subTest(method=method):
                args = parser.parse_args(
                    [
                        "partition",
                        method,
                        "model.onnx",
                        "-o",
                        "output.onnx",
                        "--tiny-tensor-threshold",
                        "256",
                        "-v",
                        "2",
                    ]
                )
                self.assertEqual(args.partition_method, method)
                self.assertEqual(args.model, "model.onnx")
                self.assertEqual(args.output, "output.onnx")
                self.assertEqual(args.tiny_tensor_threshold, 256)
                self.assertEqual(args.verbose, 2)

    def test_register_parses_module_and_optional_function(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "partition",
                "layer",
                "model.onnx",
                "-o",
                "output.onnx",
                "--register",
                "onnx_light_cpu",
                "--register",
                "package.module:register_operator_support",
            ]
        )
        self.assertEqual(
            args.register,
            [
                ("onnx_light_cpu", "register_kernels"),
                ("package.module", "register_operator_support"),
            ],
        )
        for command in (["optimize", "m.onnx", "-o", "o.onnx"], ["fillshape", "m.onnx"]):
            with self.subTest(command=command[0]):
                self.assertEqual(parser.parse_args(command).register, [])

    def test_register_rejects_invalid_specifications(self):
        for value in ("", ":f", "a..b", "a:", "a:b:c", "a:b.c", "1a"):
            with (
                self.subTest(value=value),
                redirect_stderr(StringIO()),
                self.assertRaises(SystemExit),
            ):
                _build_parser().parse_args(["show", "model.onnx", "--register", value])

    def test_register_calls_functions_in_order_before_the_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "onnx_light_test_registration.py").write_text(
                "calls = []\n"
                "def register_kernels():\n"
                "    calls.append('kernels')\n"
                "def register_shapes():\n"
                "    calls.append('shapes')\n"
                "not_callable = 1\n",
                encoding="utf-8",
            )
            model = directory / "model.onnx"
            save(
                oh.make_model(
                    oh.make_graph(
                        [oh.make_node("Relu", ["X"], ["Y"])],
                        "g",
                        [oh.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
                        [oh.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
                    ),
                    opset_imports=[oh.make_opsetid("", 18)],
                ),
                str(model),
            )
            sys.path.insert(0, temporary)
            try:
                with redirect_stdout(StringIO()):
                    main(
                        [
                            "show",
                            str(model),
                            "--register",
                            "onnx_light_test_registration:register_shapes",
                            "--register",
                            "onnx_light_test_registration",
                        ]
                    )
                module = sys.modules["onnx_light_test_registration"]
                self.assertEqual(module.calls, ["shapes", "kernels"])
                with self.assertRaisesRegex(AttributeError, "no registration function 'missing'"):
                    main(
                        ["show", str(model), "--register", "onnx_light_test_registration:missing"]
                    )
                with self.assertRaisesRegex(TypeError, "not callable"):
                    main(
                        [
                            "show",
                            str(model),
                            "--register",
                            "onnx_light_test_registration:not_callable",
                        ]
                    )
                with self.assertRaises(ModuleNotFoundError):
                    main(["show", str(model), "--register", "onnx_light_missing_module_xyz"])
            finally:
                sys.path.remove(temporary)
                sys.modules.pop("onnx_light_test_registration", None)


if __name__ == "__main__":
    unittest.main()

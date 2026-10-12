import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestCMakeHardening(unittest.TestCase):
    def test_hardening_flags_and_caller_mode(self):
        root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            source.mkdir()
            (source / "test.cc").write_text("int main() { return 0; }\n", encoding="utf-8")
            (source / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.15)\n"
                "project(hardening_test LANGUAGES CXX)\n"
                "if(ENABLE_HARDENING)\n"
                f'  include("{(root / "cmake" / "Hardening.cmake").as_posix()}")\n'
                "else()\n"
                "  function(onnx_light_apply_hardening target)\n"
                "  endfunction()\n"
                "endif()\n"
                "add_executable(test_target test.cc)\n"
                "onnx_light_apply_hardening(test_target)\n",
                encoding="utf-8",
            )

            for name, hardening, caller_mode, unsupported in (
                ("off", False, False, False),
                ("default", True, False, False),
                ("caller_mode", True, True, False),
                ("unsupported", True, False, True),
            ):
                with self.subTest(name=name):
                    build = Path(directory) / name
                    flags = "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_EXTENSIVE"
                    subprocess.run(
                        [
                            "cmake",
                            "-S",
                            str(source),
                            "-B",
                            str(build),
                            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                            f"-DENABLE_HARDENING={'ON' if hardening else 'OFF'}",
                            f"-DCMAKE_CXX_FLAGS={flags if caller_mode else ''}",
                            *(
                                ["-DONNX_LIGHT_HARDENING_CXX__Wbidi_chars_any:INTERNAL=0"]
                                if unsupported
                                else []
                            ),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    commands = json.loads(
                        (build / "compile_commands.json").read_text(encoding="utf-8")
                    )
                    command = commands[0]["command"]
                    self.assertEqual(
                        "-D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST" in command,
                        hardening and not caller_mode,
                    )
                    if caller_mode:
                        self.assertEqual(command.count(flags), 1)
                    if hardening:
                        self.assertIn("-D_GLIBCXX_ASSERTIONS", command)
                        cache = (build / "CMakeCache.txt").read_text(encoding="utf-8")
                        supported = re.search(
                            r"^ONNX_LIGHT_HARDENING_CXX__Wbidi_chars_any:INTERNAL=1$",
                            cache,
                            re.MULTILINE,
                        )
                        self.assertEqual("-Wbidi-chars=any" in command, supported is not None)
                    else:
                        self.assertNotIn("-Wbidi-chars=any", command)
                        self.assertNotIn("-D_GLIBCXX_ASSERTIONS", command)
                    subprocess.run(
                        ["cmake", "--build", str(build)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)

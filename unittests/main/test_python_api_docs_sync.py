from pathlib import Path
import unittest

from onnx_light.ext_test_case import ExtTestCase


class TestPythonApiDocsSync(ExtTestCase):
    """Tests that Python API doc toctrees stay aligned with public modules."""

    ROOT = Path(__file__).resolve().parents[2]

    def _read_toctree_entries(self, path: Path) -> set[str]:
        entries = set()
        in_toctree = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped == ".. toctree::":
                in_toctree = True
                continue
            if not in_toctree:
                continue
            if not line.startswith("    "):
                if stripped:
                    break
                continue
            if stripped.startswith(":") or not stripped:
                continue
            entries.add(Path(stripped).name)
        return entries

    def _assert_doc_index_matches_package(self, package: str, doc_index: str) -> None:
        package_dir = self.ROOT / package
        documented = self._read_toctree_entries(self.ROOT / doc_index)
        public_modules = {
            path.stem
            for path in package_dir.glob("*.py")
            if path.stem != "__init__" and not path.stem.startswith("_")
        }
        self.assertEqual(
            public_modules,
            documented,
            "\n".join(
                [
                    f"Package modules for {package} and documented pages in {doc_index} differ.",
                    f"Missing from docs: {sorted(public_modules - documented)}",
                    f"Missing from package: {sorted(documented - public_modules)}",
                ]
            ),
        )

    def _assert_automodule_uses_members(self, doc_index: str, module: str) -> None:
        text = (self.ROOT / doc_index).read_text(encoding="utf-8")
        self.assertIn(f".. automodule:: {module}\n    :members:", text)

    def test_root_api_index_includes_public_modules(self):
        documented = self._read_toctree_entries(self.ROOT / "docs/api/python/index.rst")
        public_modules = {
            path.stem
            for path in (self.ROOT / "onnx_light").glob("*.py")
            if path.stem != "__init__" and not path.stem.startswith("_")
        }
        self.assertEqual(
            set(),
            public_modules - documented,
            f"Missing from root Python API docs: {sorted(public_modules - documented)}",
        )

    def test_onnx_api_index_matches_package_modules(self):
        self._assert_doc_index_matches_package(
            "onnx_light/onnx", "docs/api/python/onnx/index.rst"
        )

    def test_onnx_core_api_index_matches_package_modules(self):
        self._assert_doc_index_matches_package(
            "onnx_light/onnx_core", "docs/api/python/onnx_core/index.rst"
        )

    def test_tools_api_index_matches_package_modules(self):
        self._assert_doc_index_matches_package(
            "onnx_light/tools", "docs/api/python/tools/index.rst"
        )

    def test_package_indexes_include_members(self):
        for doc_index, module in [
            ("docs/api/python/onnx/index.rst", "onnx_light.onnx"),
            ("docs/api/python/onnx_core/index.rst", "onnx_light.onnx_core"),
            ("docs/api/python/onnx_op/index.rst", "onnx_light.onnx_op"),
            ("docs/api/python/tools/index.rst", "onnx_light.tools"),
        ]:
            with self.subTest(doc_index=doc_index):
                self._assert_automodule_uses_members(doc_index, module)


if __name__ == "__main__":
    unittest.main(verbosity=2)

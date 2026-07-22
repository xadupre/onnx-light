from pathlib import Path
import unittest


class TestPythonApiDocsSync(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

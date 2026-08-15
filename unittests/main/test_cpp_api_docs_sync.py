import fnmatch
import re
from pathlib import Path
import unittest


class TestCppApiDocsSync(unittest.TestCase):
    """Tests that the C++ API docs stay aligned with the Doxygen-indexed headers.

    The C++ API reference under ``docs/api/cpp`` is generated from the project
    headers via Doxygen and Breathe: every documented header is pulled in with a
    ``.. doxygenfile:: <path>`` directive. This test makes sure that set of
    directives matches exactly the set of headers Doxygen indexes (the
    ``INPUT`` directories of ``docs/Doxyfile`` minus its ``EXCLUDE_PATTERNS``),
    so that new headers get a doc page and removed headers drop theirs.
    """

    ROOT = Path(__file__).resolve().parents[2]
    DOXYFILE = ROOT / "docs" / "Doxyfile"
    CPP_DOCS = ROOT / "docs" / "api" / "cpp"

    def _read_doxyfile_list(self, key: str) -> list[str]:
        """Returns the (possibly line-continued) values assigned to a Doxyfile key."""
        lines = self.DOXYFILE.read_text(encoding="utf-8").splitlines()
        values: list[str] = []
        collecting = False
        for line in lines:
            stripped = line.strip()
            if not collecting:
                match = re.match(rf"^{re.escape(key)}\s*=\s*(.*)$", line)
                if not match:
                    continue
                collecting = True
                rest = match.group(1)
            else:
                rest = stripped
            continuation = rest.endswith("\\")
            if continuation:
                rest = rest[:-1]
            values.extend(part for part in rest.split() if part)
            if not continuation:
                break
        return values

    def _indexed_headers(self) -> set[str]:
        """Returns repo-relative paths of headers Doxygen indexes."""
        input_dirs = self._read_doxyfile_list("INPUT")
        patterns = self._read_doxyfile_list("FILE_PATTERNS")
        excludes = self._read_doxyfile_list("EXCLUDE_PATTERNS")
        self.assertTrue(input_dirs, "No INPUT directories found in Doxyfile")
        self.assertTrue(patterns, "No FILE_PATTERNS found in Doxyfile")

        headers: set[str] = set()
        docs_dir = self.DOXYFILE.parent
        for entry in input_dirs:
            root = (docs_dir / entry).resolve()
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                if not any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
                    continue
                posix = path.as_posix()
                if any(fnmatch.fnmatch(posix, pat) for pat in excludes):
                    continue
                headers.add(path.relative_to(self.ROOT).as_posix())
        return headers

    def _doxygenfile_refs(self) -> set[str]:
        """Returns every ``doxygenfile`` path referenced from the C++ docs."""
        refs: set[str] = set()
        for rst in self.CPP_DOCS.rglob("*.rst"):
            for match in re.finditer(
                r"doxygenfile::\s*(\S+)", rst.read_text(encoding="utf-8")
            ):
                refs.add(match.group(1))
        return refs

    @staticmethod
    def _matches(header: str, ref: str) -> bool:
        return header == ref or header.endswith("/" + ref)

    def test_docs_cover_exactly_indexed_headers(self):
        headers = self._indexed_headers()
        refs = self._doxygenfile_refs()

        documented = {h for h in headers if any(self._matches(h, r) for r in refs)}
        undocumented = sorted(headers - documented)
        dangling = sorted(r for r in refs if not any(self._matches(h, r) for h in headers))

        self.assertEqual(
            [],
            undocumented,
            "Headers indexed by Doxygen but missing a doc page in docs/api/cpp: "
            f"{undocumented}",
        )
        self.assertEqual(
            [],
            dangling,
            "doxygenfile directives that do not match any indexed header "
            f"(remove them or fix the path): {dangling}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

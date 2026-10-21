import re
import unittest
from pathlib import Path

from onnx_light.ext_test_case import ExtTestCase


class TestNextStepsDocumentation(ExtTestCase):
    def test_proto_inheritance_abandoned(self):
        """Checks that the abandoned proposal remains accessible but is not active."""
        root = Path(__file__).resolve().parents[2] / "docs" / "next_steps" / "2026"
        proposal = (root / "2026-08_proto_inheritance.rst").read_text(encoding="utf-8")
        self.assertIn(".. _l-next-steps-proto-inheritance:", proposal)
        self.assertIn("**abandoned**", proposal)
        self.assertIn("No implementation is planned.", proposal)
        index = (root / "index.rst").read_text(encoding="utf-8")
        self.assertIn("    2026-08_proto_inheritance\n", index)
        sections = re.split(r"(?m)^([^\n]+)\n-+\n", index)
        categories = dict(zip(sections[1::2], sections[2::2]))
        reference = ":ref:`l-next-steps-proto-inheritance`"
        self.assertIn(reference, categories["Abandoned"])
        for category in ("Started", "Discussion", "Completed"):
            with self.subTest(category=category):
                self.assertNotIn(reference, categories[category])

    def test_plans_grouped_by_year(self):
        """Checks that each year has a page indexing its roadmap plans exactly once."""
        root = Path(__file__).resolve().parents[2] / "docs" / "next_steps"
        content = (root / "index.rst").read_text(encoding="utf-8")
        entries = re.findall(r"(?m)^    ([^\s:][^\s]*)$", content)
        expected_years = sorted(
            (path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
            reverse=True,
        )
        self.assertEqual(entries, [f"{year}/index" for year in expected_years])
        for year in expected_years:
            with self.subTest(year=year):
                page = (root / year / "index.rst").read_text(encoding="utf-8")
                self.assertTrue(page.startswith(f"{year}\n====\n"))
                plans = re.findall(r"(?m)^    ([^\s:][^\s]*)$", page)
                expected_plans = [
                    path.stem for path in (root / year).glob("*.rst") if path.name != "index.rst"
                ]
                self.assertCountEqual(plans, expected_plans)


if __name__ == "__main__":
    unittest.main(verbosity=2)

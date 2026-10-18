import re
import unittest
from pathlib import Path


class TestNextStepsDocumentation(unittest.TestCase):
    def test_plans_grouped_by_year(self):
        """Checks that each roadmap plan is indexed once under its matching year."""
        root = Path(__file__).resolve().parents[2] / "docs" / "next_steps"
        content = (root / "index.rst").read_text(encoding="utf-8")
        sections = re.split(r"(?m)^(\d{4})\n-{4}\n", content)
        years = sections[1::2]
        expected_years = sorted(
            (path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
            reverse=True,
        )
        self.assertEqual(years, expected_years)
        self.assertNotRegex(sections[0], r"(?m)^    \d{4}/")
        for year, section in zip(years, sections[2::2]):
            with self.subTest(year=year):
                plans = re.findall(r"(?m)^    (\d{4}/[^\s]+)$", section)
                expected_plans = [
                    path.relative_to(root).with_suffix("").as_posix()
                    for path in (root / year).glob("*.rst")
                ]
                self.assertCountEqual(plans, expected_plans)


if __name__ == "__main__":
    unittest.main(verbosity=2)

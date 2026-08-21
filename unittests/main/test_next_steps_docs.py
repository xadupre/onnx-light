from pathlib import Path
import unittest


class TestNextStepsDocs(unittest.TestCase):
    """Tests the interactive Next Steps table configuration."""

    ROOT = Path(__file__).resolve().parents[2]

    def test_table_is_sortable_and_filterable_by_status(self):
        conf = (self.ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
        page = (self.ROOT / "docs" / "next_steps" / "index.rst").read_text(encoding="utf-8")

        self.assertIn('"sphinx_datatables"', conf)
        self.assertIn(":class: sphinx-datatable", page)
        self.assertIn("* - Status", page)
        for status in ("Started", "Discussed", "Completed"):
            self.assertIn(f"* - {status}", page)


if __name__ == "__main__":
    unittest.main(verbosity=2)

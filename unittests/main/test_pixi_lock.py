import unittest

from onnx_light.ext_test_case import ExtTestCase
from pathlib import Path


class TestPixiLock(ExtTestCase):
    """Sanity checks on the committed pixi.lock manifest."""

    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.lock_path = cls.root / "pixi.lock"
        cls.toml_path = cls.root / "pixi.toml"

    def test_pixi_lock_exists(self):
        self.assertTrue(self.lock_path.is_file(), f"missing {self.lock_path}")
        self.assertTrue(self.toml_path.is_file(), f"missing {self.toml_path}")

    def _parse_envs(self):
        """Returns a dict {env_name: [package_url, ...]} parsed from pixi.lock."""
        envs: dict[str, list[str]] = {}
        current_env: str | None = None
        in_packages_section = False
        in_envs_block = False
        for raw in self.lock_path.read_text(encoding="utf-8").splitlines():
            if raw == "environments:":
                in_envs_block = True
                continue
            if in_envs_block and raw and not raw.startswith(" "):
                # Top-level key other than environments ends the block.
                in_envs_block = False
            if not in_envs_block:
                continue
            # Environment name: "  <name>:" (two spaces indent).
            if raw.startswith("  ") and not raw.startswith("    ") and raw.rstrip().endswith(":"):
                current_env = raw.strip()[:-1]
                envs[current_env] = []
                in_packages_section = False
                continue
            if current_env is None:
                continue
            stripped = raw.strip()
            if stripped == "packages:":
                in_packages_section = True
                continue
            if in_packages_section and stripped.startswith("- "):
                envs[current_env].append(stripped[2:])
        return envs

    def test_dev_no_onnx_env_has_no_protobuf(self):
        envs = self._parse_envs()
        self.assertIn(
            "dev-no-onnx",
            envs,
            f"'dev-no-onnx' env not found in pixi.lock; envs found: {sorted(envs)}",
        )
        offenders = [pkg for pkg in envs["dev-no-onnx"] if "protobuf" in pkg]
        self.assertEqual(
            offenders,
            [],
            f"'dev-no-onnx' env unexpectedly contains protobuf packages: {offenders}",
        )

    def test_dev_no_onnx_env_has_no_onnx(self):
        envs = self._parse_envs()
        self.assertIn("dev-no-onnx", envs)
        offenders = [
            pkg for pkg in envs["dev-no-onnx"] if "/onnx-" in pkg or "/onnxruntime-" in pkg
        ]
        self.assertEqual(
            offenders, [], f"'dev-no-onnx' env unexpectedly contains onnx packages: {offenders}"
        )

    def test_default_env_has_no_protobuf(self):
        envs = self._parse_envs()
        self.assertIn("default", envs)
        offenders = [pkg for pkg in envs["default"] if "protobuf" in pkg]
        self.assertEqual(
            offenders, [], f"'default' env unexpectedly contains protobuf packages: {offenders}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestReleaseWheelWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github/workflows/build_release_wheel.yml").read_text(
            encoding="utf-8"
        )
        cls.linux_job = workflow.split("\n  build_wheels_linux:", 1)[1].split(
            "\n  build_wheels_windows:", 1
        )[0]
        cls.cleanup = (
            cls.linux_job.split("CIBW_BEFORE_BUILD_LINUX:", 1)[1].splitlines()[0].strip()
        )

    def test_linux_cleanup_targets_only_native_build_directory(self):
        self.assertEqual(self.cleanup, 'rm -rf "{project}/build"')
        self.assertIn('CIBW_BUILD: "cp312-* cp313-* cp314-*"', self.linux_job)

    @unittest.skipUnless(os.name == "posix", "Linux wheel cleanup uses a POSIX shell.")
    def test_cleanup_between_python_versions_preserves_sources_and_wheels(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project with spaces"
            project.mkdir()
            source = project / "CMakeLists.txt"
            source.write_text("source", encoding="utf-8")
            wheelhouse = project / "wheelhouse"
            wheelhouse.mkdir()
            wheels = []
            for version in ("cp312", "cp313", "cp314"):
                with self.subTest(version=version):
                    subprocess.run(
                        ["sh", "-c", self.cleanup.format(project=project)],
                        cwd=temporary,
                        check=True,
                    )
                    self.assertFalse((project / "build").exists())
                    self.assertEqual(source.read_text(encoding="utf-8"), "source")
                    self.assertEqual(sorted(wheelhouse.iterdir()), wheels)
                    build = project / "build" / version
                    build.mkdir(parents=True)
                    (build / "object.o").write_bytes(b"native build")
                    wheel = wheelhouse / f"{version}.whl"
                    wheel.write_bytes(b"completed wheel")
                    wheels.append(wheel)

    @unittest.skipUnless(os.name == "posix", "Linux wheel cleanup uses a POSIX shell.")
    def test_cleanup_does_not_follow_build_directory_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary) / "project"
            project.mkdir()
            outside = Path(temporary) / "outside"
            outside.mkdir()
            source = outside / "source.txt"
            source.write_text("preserved", encoding="utf-8")
            (project / "build").symlink_to(outside, target_is_directory=True)
            subprocess.run(
                ["sh", "-c", self.cleanup.format(project=project)], cwd=temporary, check=True
            )
            self.assertFalse((project / "build").is_symlink())
            self.assertEqual(source.read_text(encoding="utf-8"), "preserved")


if __name__ == "__main__":
    unittest.main()

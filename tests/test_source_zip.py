from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent.parent


class SourceZipTests(unittest.TestCase):
    def test_archive_includes_role_templates_and_test_helpers_but_not_local_roles(self):
        spec = spec_from_file_location("source_zip", ROOT / "scripts/build_source_zip.py")
        builder = module_from_spec(spec)
        spec.loader.exec_module(builder)
        staging = Path(self.enterContext(TemporaryDirectory(dir=ROOT)))
        for relative in builder.delivery_paths():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            copyfile(ROOT / relative, path)
        (staging / "agents/main.md").write_bytes(b"local test fixture")
        (staging / "agents/private.md").write_bytes(b"local test fixture")
        workspace = staging / "workspaces" / ("a" * 64)
        workspace.mkdir(parents=True)
        (workspace / "fixture.py").write_bytes(b"local workspace data")
        custom_workspace = staging / "src/agent_companion" / ("b" * 64)
        custom_workspace.mkdir()
        (custom_workspace / "__init__.py").write_bytes(b"local workspace data")
        with patch.object(builder, "ROOT", staging):
            builder.build()
        with ZipFile(staging / "dist" / builder.ARCHIVE) as archive:
            names = archive.namelist()
            self.assertEqual([name for name in names if name.startswith("agents/")],
                             ["agents/main.example.md", "agents/sub.example.md"])
            self.assertIn("tests/feishu_test_support.py", names)
            self.assertIn("src/agent_companion/runtime/_bash_launcher.py", names)
            self.assertIn("src/agent_companion/runtime/_child_exit.py", names)
            self.assertFalse(any(name.startswith("workspaces/") for name in names))
            self.assertFalse(any(custom_workspace.name in name for name in names))

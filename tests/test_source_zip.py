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
        with patch.object(builder, "ROOT", staging):
            builder.build()
        with ZipFile(staging / "dist" / builder.ARCHIVE) as archive:
            names = archive.namelist()
            self.assertEqual([name for name in names if name.startswith("agents/")],
                             ["agents/main.example.md", "agents/sub.example.md"])
            self.assertIn("tests/feishu_test_support.py", names)

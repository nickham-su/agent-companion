#!/usr/bin/env python3
"""Build the MVP source ZIP from documented, project-local paths only."""

from collections.abc import Iterator
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parent.parent
DESIGN = Path("docs/design/0001-MVP方案")
FILES = (
    Path("README.md"),
    Path("config.example.toml"),
    Path("agents/main.example.md"),
    Path("agents/sub.example.md"),
    Path("pyproject.toml"),
    Path("scripts/build_source_zip.py"),
    Path("tests/feishu_test_support.py"),
    Path("docs/acceptance/mac-checklist.md"),
    Path("docs/acceptance/offline-results.md"),
    DESIGN / "飞书多模态 Agent 首版架构方案.md",
    DESIGN / "飞书多模态 Agent 首版实施与验收计划.md",
)
ARCHIVE = "agent-companion-mvp.zip"


def checked_path(relative: Path) -> Path:
    path = ROOT
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"symlink not allowed in delivery: {relative.as_posix()}")
    return path


def python_sources(relative: Path) -> Iterator[Path]:
    directory = checked_path(relative)
    for entry in directory.iterdir():
        path = relative / entry.name
        if entry.name.isidentifier() and entry.name != "__pycache__":
            if checked_path(path).is_dir() and checked_path(path / "__init__.py").is_file():
                yield from python_sources(path)
        elif entry.suffix == ".py" and entry.stem.isidentifier():
            yield path


def delivery_paths() -> list[Path]:
    tests = checked_path(Path("tests"))
    paths = [*FILES, *python_sources(Path("src/agent_companion")),
             *(path.relative_to(ROOT) for path in tests.glob("test_*.py") if path.stem.isidentifier())]
    return sorted(paths, key=Path.as_posix)


def build() -> str:
    output = checked_path(Path("dist"))
    output.mkdir(exist_ok=True)
    # Build both files beside their final paths, without exposing an incomplete ZIP.
    with TemporaryDirectory(prefix=".source-zip-", dir=output) as directory:
        temporary = Path(directory) / ARCHIVE
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
            for relative in delivery_paths():
                source = checked_path(relative)
                if not source.is_file():
                    raise ValueError(f"expected regular file: {relative.as_posix()}")
                entry = ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
                entry.create_system = 3  # stable Unix permissions, independent of host and umask
                entry.external_attr = 0o100644 << 16
                bundle.writestr(entry, source.read_bytes(), compress_type=ZIP_DEFLATED, compresslevel=9)
        with temporary.open("rb") as stream:
            digest = file_digest(stream, "sha256").hexdigest()
        manifest = Path(directory) / f"{ARCHIVE}.sha256"
        manifest.write_text(f"{digest}  {ARCHIVE}\n", encoding="ascii")
        temporary.replace(output / ARCHIVE)
        manifest.replace(output / manifest.name)
    return digest


if __name__ == "__main__":
    print(f"{build()}  dist/{ARCHIVE}")

"""Prepare the recorded inputs in a new directory, then replay the archived FLAG system."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TypedDict
from zipfile import ZipFile


class Resource(TypedDict):
    sha256: str
    bytes: int
    bundled: bool
    stages: list[str]


class Inventory(TypedDict):
    paths: dict[str, Resource]
    reference_zip_sha256: str


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def configure(path: Path, keys: tuple[str, ...], value: str) -> None:
    config = json.loads(path.read_text())
    section = config
    for key in keys[:-1]:
        section = section[key]
    if section[keys[-1]] != value:
        section[keys[-1]] = value
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")


def prepare(repo: Path, work: Path, resources: Path) -> None:
    inventory: Inventory = json.loads((repo / "resources.json").read_text())
    files: dict[str, Path] = {}
    for name, item in inventory["paths"].items():
        path = (repo if item["bundled"] else resources) / name
        if not path.is_file():
            raise FileNotFoundError(f"Required review input is missing: {path}")
        if path.stat().st_size != item["bytes"] or digest(path) != item["sha256"]:
            raise ValueError(
                f"Review input does not match the archived SHA-256: {path}"
            )
        files[name] = path
    expected_zip = repo / "reference/submitted.zip"
    if digest(expected_zip) != inventory["reference_zip_sha256"]:
        raise ValueError(f"Reference ZIP checksum mismatch: {expected_zip}")
    work.mkdir(parents=True, exist_ok=False)
    for name, src in files.items():
        dest = work / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if name == "inference/config.json":
            dest.write_bytes(src.read_bytes())
        else:
            dest.symlink_to(src)
    configure(work / "inference/config.json", ("python_prefix",), sys.prefix)
    print(
        f"Verified {len(files)} archived inputs and the reference ZIP. Workspace: {work}",
        flush=True,
    )


def compare(work: Path, reference: Path, actual: Path) -> None:
    result: dict[str, dict[str, int | str | bool]] = {}
    with ZipFile(reference) as old, ZipFile(actual) as new:
        if set(old.namelist()) != set(new.namelist()) or new.testzip() is not None:
            raise ValueError("Archive membership or CRC differs")
        for name in sorted(old.namelist()):
            content = new.read(name)
            result[name] = {
                "rows": len(content.splitlines()),
                "sha256": hashlib.sha256(content).hexdigest(),
                "identical": content == old.read(name),
            }
    report = {
        "reference_zip_sha256": digest(reference),
        "regenerated_zip_sha256": digest(actual),
        "members": result,
    }
    path = work / "comparison.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if not all(row["identical"] for row in result.values()):
        raise ValueError(f"The regenerated scores differ. See {path}")
    print(f"All {len(result)} score files are byte-identical. Comparison: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("replay",))
    parser.add_argument(
        "--resources",
        required=True,
        type=Path,
        help="Licensed resource tree with the paths in resources.json",
    )
    parser.add_argument(
        "--work",
        required=True,
        type=Path,
        help="A new, empty workspace path (must not yet exist)",
    )
    args = parser.parse_args()
    repo = Path(__file__).resolve().parent
    work = args.work.resolve()
    if work.exists():
        parser.error(f"Please choose an unused --work path: {work}")
    prepare(repo, work, args.resources.resolve())
    config = work / "inference/config.json"
    subprocess.run([sys.executable, "-m", "inference.package"], cwd=work, check=True)
    actual = work / json.loads(config.read_text())["package"]["destination"]
    compare(work, repo / "reference/submitted.zip", actual)


if __name__ == "__main__":
    main()

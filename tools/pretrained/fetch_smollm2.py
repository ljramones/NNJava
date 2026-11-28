#!/usr/bin/env python3
"""Fetch the pinned public checkpoint to an explicit directory, or verify existing files offline."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import urllib.request

MANIFEST = Path(__file__).with_name("smollm2-135m.json")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify(directory):
    manifest = json.loads(MANIFEST.read_text())
    for filename, expected in manifest["files"].items():
        path = directory / filename
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or mismatched pinned asset: {path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify(args.directory)
    else:
        args.directory.mkdir(parents=True, exist_ok=True)
        manifest = json.loads(MANIFEST.read_text())
        for filename, expected in manifest["files"].items():
            target = args.directory / filename
            if target.exists():
                if sha256(target) != expected:
                    raise ValueError(f"Refusing to overwrite mismatched existing file: {target}")
                continue
            url = f'https://huggingface.co/{manifest["model"]}/resolve/{manifest["revision"]}/{filename}'
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=args.directory, prefix=filename + ".", suffix=".part", delete=False) as output:
                    temporary = Path(output.name)
                    with urllib.request.urlopen(url, timeout=60) as source:
                        while block := source.read(1024 * 1024):
                            output.write(block)
                if sha256(temporary) != expected:
                    raise ValueError(f"Checksum mismatch: {filename}")
                temporary.replace(target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        verify(args.directory)
    print(f"Verified pinned SmolLM2 assets: {args.directory}")


if __name__ == "__main__":
    main()

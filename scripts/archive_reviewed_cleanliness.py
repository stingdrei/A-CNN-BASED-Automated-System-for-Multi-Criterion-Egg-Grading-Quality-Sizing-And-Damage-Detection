"""Create verified, reproducible archive of reviewed train/validation crops."""
import argparse
import csv
import gzip
import hashlib
import tarfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/reviewed_cleanliness"
ARCHIVE_FILES = [
    "data/reviewed_cleanliness/README.md",
    "data/reviewed_cleanliness/manifests/train.csv",
    "data/reviewed_cleanliness/manifests/valid.csv",
    "data/reviewed_cleanliness/source/accepted_manifest.csv",
    "data/reviewed_cleanliness/source/audit.json",
    "data/reviewed_cleanliness/source/audit.md",
    "data/reviewed_cleanliness/source/review.csv.snapshot",
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DATA,
                        help="Root used by manifest filenames (default: data/reviewed_cleanliness)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Archive path; choose an external drive or thesis data archive")
    args = parser.parse_args()

    crop_entries = {}
    for manifest_name in ("train.csv", "valid.csv"):
        manifest = DATA / "manifests" / manifest_name
        with manifest.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                filename = row["filename"]
                path = (args.image_root / filename).resolve()
                if not path.is_relative_to(args.image_root.resolve()):
                    raise ValueError(f"Manifest path escapes image root: {filename}")
                if not path.is_file():
                    raise FileNotFoundError(path)
                actual = sha256(path)
                if actual != row["crop_sha256"]:
                    raise ValueError(f"SHA-256 mismatch for {filename}: {actual}")
                archive_name = f"data/reviewed_cleanliness/{filename}"
                crop_entries[archive_name] = (path, actual)

    files = {}
    for name in ARCHIVE_FILES:
        path = REPO / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = path
    for name, (path, _) in crop_entries.items():
        files[name] = path

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(files):
                    path = files[name]
                    info = tarfile.TarInfo(name)
                    info.size = path.stat().st_size
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with path.open("rb") as handle:
                        archive.addfile(info, handle)

    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    with sidecar.open("w", encoding="utf-8") as handle:
        handle.write(f"{sha256(args.output)}  {args.output.name}\n")
    crop_hashes = args.output.with_suffix(args.output.suffix + ".crops.sha256")
    with crop_hashes.open("w", encoding="utf-8") as handle:
        for name, (_, digest) in sorted(crop_entries.items()):
            handle.write(f"{digest}  {name}\n")

    print(f"Archived {len(crop_entries)} verified train/validation crops")
    print(f"Archive: {args.output}")
    print(f"Archive SHA-256: {sha256(args.output)}")
    print(f"Crop checksums: {crop_hashes}")
    print("Excluded: quality_holdout, real-camera test set (not collected)")


if __name__ == "__main__":
    main()

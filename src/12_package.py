"""12 - Package the checkpoints into a single archive for cloud backup.

Zips everything in checkpoints/ (fitted sklearn pipelines and the DistilBERT
checkpoints for both label sets) and prints the absolute path.

The archive is written OUTSIDE the repo, because checkpoints/ is gitignored and
the archive must not be committed either. By default it goes to the drive with
the most free space: the development machine's C: drive was completely full,
so writing a ~700 MB archive there would fail. Override with --out.

Usage:
    python src/12_package.py
    python src/12_package.py --out D:/backups
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

MIN_HEADROOM_BYTES = 2 * 1024 ** 3  # want 2 GiB spare wherever we write


def pick_output_dir(explicit=None):
    """Explicit path wins; otherwise the drive with the most free space."""
    if explicit:
        p = Path(explicit)
        p.mkdir(parents=True, exist_ok=True)
        return p

    candidates = []
    for drive in ("C:/", "D:/", "E:/"):
        try:
            usage = shutil.disk_usage(drive)
        except OSError:
            continue
        candidates.append((usage.free, drive))
    if not candidates:
        return C.ROOT.parent

    candidates.sort(reverse=True)
    free, drive = candidates[0]
    print("  free space by drive:")
    for f, d in sorted(candidates, key=lambda t: t[1]):
        print("    {}  {:>8.2f} GiB{}".format(
            d, f / 1024 ** 3, "   <- chosen" if d == drive else ""))
    if free < MIN_HEADROOM_BYTES:
        print("  WARNING: even the roomiest drive has < 2 GiB free.")
    out = Path(drive) / "conda_tox_backups"
    out.mkdir(parents=True, exist_ok=True)
    return out


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="directory for the archive (default: roomiest drive)")
    args = ap.parse_args()

    C.banner("12 - PACKAGE CHECKPOINTS")

    files = sorted(
        p for p in C.CKPT.rglob("*")
        if p.is_file() and p.name != ".gitkeep"
    )
    if not files:
        print("  checkpoints/ is empty - nothing to package.")
        print("  run scripts 03 and 04 first.")
        sys.exit(1)

    total = sum(p.stat().st_size for p in files)
    print("  {} file(s), {:.1f} MB uncompressed:".format(
        len(files), total / 1024 ** 2))
    for p in files:
        print("    {:<34} {:>9.1f} MB".format(
            p.relative_to(C.CKPT).as_posix(), p.stat().st_size / 1024 ** 2))

    out_dir = pick_output_dir(args.out)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = out_dir / "conda_toxicity_checkpoints_{}.zip".format(stamp)

    print("\n  writing archive (deflate)...")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for p in files:
            zf.write(p, arcname=p.relative_to(C.CKPT).as_posix())
        # include the results JSONs so the archive is self-describing
        for p in sorted(C.RESULTS.glob("*.json")):
            zf.write(p, arcname="results/" + p.name)
        smry = C.RESULTS / "SUMMARY.md"
        if smry.exists():
            zf.write(smry, arcname="results/SUMMARY.md")

    size = archive.stat().st_size
    print("  computing sha256...")
    digest = sha256(archive)

    C.banner("12 - ARCHIVE READY")
    print("  path      : {}".format(archive.resolve()))
    print("  size      : {:.1f} MB ({:.1%} of uncompressed)".format(
        size / 1024 ** 2, size / total if total else 0))
    print("  sha256    : {}".format(digest))
    print("\n  Back this up to cloud storage. It contains the fitted")
    print("  sklearn pipelines, the DistilBERT checkpoint(s) and every")
    print("  results/*.json, so the tables can be regenerated without")
    print("  retraining.")

    C.save_json(
        {
            "archive_path": str(archive.resolve()),
            "archive_size_bytes": size,
            "archive_sha256": digest,
            "uncompressed_bytes": total,
            "n_checkpoint_files": len(files),
            "files": [p.relative_to(C.CKPT).as_posix() for p in files],
            "created": stamp,
        },
        C.RESULTS / "12_package.json",
        "12_package",
    )


if __name__ == "__main__":
    main()

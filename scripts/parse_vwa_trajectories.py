#!/usr/bin/env python3
"""
parse_vwa_trajectories.py - Extract final agent screenshots from VWA trajectory HTML.

Reads the GPT-4V + SoM trajectory archive published by the VisualWebArena authors
and writes, for each dataset instance, the agent's FINAL page state as
data/screenshots/<instance_id>.png.

The archive is a ~7.8 GB tar of render_<task_id>.html files, one per task, each
embedding the trajectory's screenshots as base64 data URIs. Members are streamed
one at a time so the archive never has to be unpacked to disk.

Note on references: the agent's FIRST frame is its starting page, not a reference
for a completed task. Visual references for Conditions C/D come from the human
Playwright traces via extract_human_references.py, not from this script.

Usage:
    python3 scripts/parse_vwa_trajectories.py --archive path/to/gpt4v_som.tar
    python3 scripts/parse_vwa_trajectories.py --archive ... --dry-run
"""

import argparse
import base64
import json
import re
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"

RENDER_RE = re.compile(r"render_(\d+)\.html$")
DOMAIN_RE = re.compile(r"(classifieds|shopping|reddit)_gpt4v_som")
IMG_SRC_RE = re.compile(
    r'''<img[^>]+src=(["'])data:image/[^;]+;base64,(.*?)\1''', re.I | re.S)


def build_domain_map(dataset_path: Path) -> dict:
    """Map (domain, task_id) -> instance_id from the verification dataset."""
    mapping = {}
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            iid = json.loads(line)["instance_id"]
            parts = iid.replace("vwa_", "").split("_")
            mapping[(parts[0], int(parts[1]))] = iid
    return mapping


def extract_images(html: str) -> list[str]:
    """Return base64 payloads of every embedded screenshot, in document order."""
    return [m.group(2) for m in IMG_SRC_RE.finditer(html)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", required=True,
                    help="tar archive of the gpt4v_som trajectory HTML files")
    ap.add_argument("--dataset", default=str(DATA_DIR / "verification_dataset.jsonl"))
    ap.add_argument("--out-dir", default=str(SCREENSHOTS_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be written without writing")
    args = ap.parse_args()

    domain_map = build_domain_map(Path(args.dataset))
    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"html": 0, "written": 0, "no_images": 0, "unmatched": 0}
    frame_counts = {}

    with tarfile.open(args.archive, "r|*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            if Path(member.name).name.startswith("._"):
                continue
            m = RENDER_RE.search(member.name)
            d = DOMAIN_RE.search(member.name)
            if not m or not d:
                continue
            stats["html"] += 1
            task_id, domain = int(m.group(1)), d.group(1)

            instance_id = domain_map.get((domain, task_id))
            if instance_id is None:
                stats["unmatched"] += 1
                continue

            fh = tar.extractfile(member)
            if fh is None:
                continue
            images = extract_images(fh.read().decode("utf-8", errors="replace"))
            if not images:
                stats["no_images"] += 1
                continue
            frame_counts[instance_id] = len(images)

            if not args.dry_run:
                # last frame = the agent's final page state, which is what is verified
                (out_dir / f"{instance_id}.png").write_bytes(
                    base64.b64decode(images[-1]))
            stats["written"] += 1

            if stats["html"] % 100 == 0:
                print(f"  {stats['html']} trajectories processed", file=sys.stderr)

    print(f"\ntrajectory HTML files : {stats['html']}")
    print(f"matched to dataset    : {stats['written']}")
    print(f"unmatched task ids    : {stats['unmatched']}")
    print(f"no embedded images    : {stats['no_images']}")
    if frame_counts:
        multi = sum(1 for v in frame_counts.values() if v > 1)
        print(f"trajectories with >1 frame: {multi}")
    print(f"{'[dry-run] would write' if args.dry_run else 'wrote'} "
          f"{stats['written']} screenshots to {out_dir}")


if __name__ == "__main__":
    main()

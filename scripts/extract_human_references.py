#!/usr/bin/env python3
"""
extract_human_references.py - Extract visual references from human Playwright traces.

The VisualWebArena authors published human demonstrations as Playwright
trace.zip recordings, one per task id, laid out as

    <human-dir>/human_trajectories_<domain>/<task_id>.trace.zip

Each trace embeds a screencast: a stream of `screencast-frame` events, each
naming a JPEG in the archive's resources/ directory and carrying a timestamp.
The frame with the largest timestamp is the page as the human left it, i.e. the
final state of a successfully completed task. That frame is this script's
output, written to <ref-dir>/<instance_id>.png -- the visual reference used by
Conditions C and D.

Only a subset of dataset instances have a matching human trace; the instances
that do are exactly the visual-reference subset.

Usage:
    python3 scripts/extract_human_references.py --human-dir data/human_trajectories
    python3 scripts/extract_human_references.py --human-dir ... --dry-run
"""

import argparse
import io
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

DOMAIN_RE = re.compile(r"human_trajectories_(\w+)")


def build_instance_map(dataset_path: Path) -> dict:
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


def final_frame(trace_path: Path) -> bytes | None:
    """Return the bytes of the last screencast frame in a Playwright trace."""
    with zipfile.ZipFile(trace_path) as z:
        if "trace.trace" not in z.namelist():
            return None
        frames = []
        for line in z.read("trace.trace").decode("utf-8", "replace").splitlines():
            if not line.strip() or "screencast-frame" not in line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "screencast-frame" and event.get("sha1"):
                frames.append(event)
        if not frames:
            return None
        last = max(frames, key=lambda e: e.get("timestamp", 0))
        name = f"resources/{last['sha1']}"
        if name not in z.namelist():
            return None
        return z.read(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human-dir", default=str(DATA_DIR / "human_trajectories"),
                    help="directory containing human_trajectories_<domain>/ folders")
    ap.add_argument("--dataset", default=str(DATA_DIR / "verification_dataset.jsonl"))
    ap.add_argument("--ref-dir", default=str(DATA_DIR / "references"))
    ap.add_argument("--id-list", default="",
                    help="optional path to write the matched instance ids")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        print("error: pip install Pillow", file=sys.stderr)
        sys.exit(2)

    instance_map = build_instance_map(Path(args.dataset))
    ref_dir = Path(args.ref_dir)
    if not args.dry_run:
        ref_dir.mkdir(parents=True, exist_ok=True)

    traces = sorted(Path(args.human_dir).rglob("*.trace.zip"))
    stats = {"traces": 0, "written": 0, "unmatched": 0, "no_frame": 0}
    matched = []

    for trace in traces:
        d = DOMAIN_RE.search(str(trace.parent))
        if not d:
            continue
        stats["traces"] += 1
        domain = d.group(1)
        task_id = trace.name.split(".")[0]
        if not task_id.isdigit():
            continue

        instance_id = instance_map.get((domain, int(task_id)))
        if instance_id is None:
            stats["unmatched"] += 1
            continue

        try:
            data = final_frame(trace)
        except zipfile.BadZipFile:
            print(f"  warning: unreadable trace {trace.name}", file=sys.stderr)
            data = None
        if data is None:
            stats["no_frame"] += 1
            continue

        matched.append(instance_id)
        if not args.dry_run:
            # frames are JPEG; the pipeline addresses references as .png
            Image.open(io.BytesIO(data)).convert("RGB").save(
                ref_dir / f"{instance_id}.png", "PNG")
        stats["written"] += 1

    print(f"\ntraces found        : {stats['traces']}")
    print(f"matched to dataset  : {stats['written']}")
    print(f"no dataset instance : {stats['unmatched']}")
    print(f"no screencast frame : {stats['no_frame']}")
    print(f"{'[dry-run] would write' if args.dry_run else 'wrote'} "
          f"{stats['written']} references to {ref_dir}")

    if args.id_list and not args.dry_run:
        Path(args.id_list).write_text("\n".join(sorted(matched)) + "\n")
        print(f"wrote matched id list to {args.id_list}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Non-LLM baseline verifiers: pHash similarity and template matching.

These baselines address Reviewer concern about missing non-LLM comparisons.
They use only the screenshots (no LLM calls) to determine task completion.

pHash baseline: compares perceptual hash of final screenshot against reference.
Template matching: uses OpenCV template matching between screenshots.

Usage:
    python3 scripts/run_nonllm_baselines.py --n 909
    python3 scripts/run_nonllm_baselines.py --dry-run
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REFERENCES_DIR = DATA_DIR / "references"
RESULTS_DIR = ROOT / "results"
SCREENSHOTS_RESIZED_DIR = DATA_DIR / "screenshots_resized"
REFERENCES_RESIZED_DIR = DATA_DIR / "references_resized"


def resolve_pair(iid: str):
    """Return (final_path, ref_path, source) or None if either image is unavailable.

    Prefers the human-trace references extracted by extract_human_references.py;
    falls back to the resized JPGs that ship in the public checkout. Note the
    fallback references are the agent's FIRST frame, not a human final state.
    """
    final_png, ref_png = SCREENSHOTS_DIR / f"{iid}.png", REFERENCES_DIR / f"{iid}.png"
    if final_png.exists() and ref_png.exists():
        return final_png, ref_png, "human-reference"
    final_jpg, ref_jpg = SCREENSHOTS_RESIZED_DIR / f"{iid}.jpg", REFERENCES_RESIZED_DIR / f"{iid}.jpg"
    if final_jpg.exists() and ref_jpg.exists():
        return final_jpg, ref_jpg, "agent-first-frame"
    return None


def scored_pair(final_path, ref_path):
    """SSIM and normalized pHash distance at matched resolution.

    Screenshots are 1280x2048 while human references are 800x446, so both are
    resampled to a common size first. Cropping to the overlap instead (the
    original behaviour) would compare a corner of the screenshot to the whole
    reference.
    """
    from PIL import Image
    import numpy as np
    import imagehash
    from skimage.metrics import structural_similarity as ssim

    a = Image.open(final_path).convert("RGB")
    b = Image.open(ref_path).convert("RGB")
    if a.size != b.size:
        a = a.resize(b.size, Image.LANCZOS)
    ssim_score = ssim(np.array(a.convert("L")), np.array(b.convert("L")))
    phash_dist = (imagehash.phash(a) - imagehash.phash(b)) / 64.0
    return float(ssim_score), float(phash_dist)


def load_dataset() -> list[dict]:
    rows = []
    with open(DATA_DIR / "verification_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_phash_distance(img1_path: str, img2_path: str) -> float:
    """Compute normalized Hamming distance between perceptual hashes.
    Returns 0.0 (identical) to 1.0 (completely different).
    """
    try:
        from PIL import Image
        import imagehash
    except ImportError:
        print("error: pip install imagehash Pillow", file=sys.stderr)
        sys.exit(2)

    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    h1 = imagehash.phash(img1)
    h2 = imagehash.phash(img2)
    # Normalize: hash distance / total bits (64 for phash)
    return (h1 - h2) / 64.0


def compute_ssim_score(img1_path: str, img2_path: str) -> float:
    """Compute Structural Similarity Index between two images.
    Returns -1 to 1 (1 = identical).
    """
    try:
        from PIL import Image
        import numpy as np
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        print("error: pip install scikit-image Pillow numpy", file=sys.stderr)
        sys.exit(2)

    img1 = np.array(Image.open(img1_path).convert("L"))
    img2 = np.array(Image.open(img2_path).convert("L"))
    # Resize to same dimensions if needed
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    return ssim(img1, img2)


def compute_mse_score(img1_path: str, img2_path: str) -> float:
    """Compute Mean Squared Error between two images.
    Returns 0.0 (identical) to higher values (different).
    """
    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("error: pip install Pillow numpy", file=sys.stderr)
        sys.exit(2)

    img1 = np.array(Image.open(img1_path).convert("L")).astype(float)
    img2 = np.array(Image.open(img2_path).convert("L")).astype(float)
    if img1.shape != img2.shape:
        h = min(img1.shape[0], img2.shape[0])
        w = min(img1.shape[1], img2.shape[1])
        img1 = img1[:h, :w]
        img2 = img2[:h, :w]
    return float(np.mean((img1 - img2) ** 2))


def run_phash_baseline(rows: list[dict], threshold: float = 0.3) -> list[dict]:
    """pHash baseline: if final screenshot is too different from reference, FAILURE."""
    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        iid = r["instance_id"]
        final_path = SCREENSHOTS_DIR / f"{iid}.png"
        ref_path = REFERENCES_DIR / f"{iid}.png"

        if not final_path.exists() or not ref_path.exists():
            results.append({
                "instance_id": iid, "model": "phash", "condition": "non-llm",
                "verdict": "SKIP", "similarity": None,
                "ground_truth": r["ground_truth"], "failure_type": r.get("failure_type", ""),
            })
            continue

        dist = compute_phash_distance(str(final_path), str(ref_path))
        similarity = 1.0 - dist
        verdict = "SUCCESS" if dist < threshold else "FAILURE"

        results.append({
            "instance_id": iid, "model": "phash", "condition": "non-llm",
            "verdict": verdict, "similarity": round(similarity, 4),
            "ground_truth": r["ground_truth"], "failure_type": r.get("failure_type", ""),
        })

        if (i + 1) % 100 == 0:
            print(f"  pHash: {i+1}/{len(rows)} in {time.time()-t0:.0f}s", file=sys.stderr)

    return results


def run_ssim_baseline(rows: list[dict], threshold: float = 0.5) -> list[dict]:
    """SSIM baseline: structural similarity between screenshots."""
    results = []
    t0 = time.time()
    for i, r in enumerate(rows):
        iid = r["instance_id"]
        final_path = SCREENSHOTS_DIR / f"{iid}.png"
        ref_path = REFERENCES_DIR / f"{iid}.png"

        if not final_path.exists() or not ref_path.exists():
            results.append({
                "instance_id": iid, "model": "ssim", "condition": "non-llm",
                "verdict": "SKIP", "similarity": None,
                "ground_truth": r["ground_truth"], "failure_type": r.get("failure_type", ""),
            })
            continue

        score = compute_ssim_score(str(final_path), str(ref_path))
        verdict = "SUCCESS" if score > threshold else "FAILURE"

        results.append({
            "instance_id": iid, "model": "ssim", "condition": "non-llm",
            "verdict": verdict, "similarity": round(score, 4),
            "ground_truth": r["ground_truth"], "failure_type": r.get("failure_type", ""),
        })

        if (i + 1) % 100 == 0:
            print(f"  SSIM: {i+1}/{len(rows)} in {time.time()-t0:.0f}s", file=sys.stderr)

    return results


def compute_metrics(results: list[dict], model_name: str) -> dict:
    """Compute accuracy, F1, FPR for a baseline."""
    valid = [r for r in results if r["verdict"] in ("SUCCESS", "FAILURE")]
    if not valid:
        return {"model": model_name, "n": 0}

    tp = sum(1 for r in valid if r["ground_truth"] == "SUCCESS" and r["verdict"] == "SUCCESS")
    tn = sum(1 for r in valid if r["ground_truth"] == "FAILURE" and r["verdict"] == "FAILURE")
    fp = sum(1 for r in valid if r["ground_truth"] == "FAILURE" and r["verdict"] == "SUCCESS")
    fn = sum(1 for r in valid if r["ground_truth"] == "SUCCESS" and r["verdict"] == "FAILURE")

    accuracy = (tp + tn) / len(valid)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fpr = fp / max(1, fp + tn)

    return {
        "model": model_name,
        "n": len(valid),
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fpr": round(fpr, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def collect_scores(rows: list[dict]) -> tuple[list[dict], dict]:
    """Compute SSIM score and pHash distance once per instance.

    Returns (records, source_counts). Each record carries the raw scores plus
    ground truth, so thresholds can be swept without recomputing images.
    """
    records = []
    sources = {"human-reference": 0, "agent-first-frame": 0, "missing": 0}
    t0 = time.time()
    for i, r in enumerate(rows):
        iid = r["instance_id"]
        pair = resolve_pair(iid)
        if pair is None:
            sources["missing"] += 1
            continue
        final_path, ref_path, source = pair
        sources[source] += 1
        ssim_score, phash_dist = scored_pair(final_path, ref_path)
        records.append({
            "instance_id": iid,
            "ssim": ssim_score,
            "phash_dist": phash_dist,
            "ground_truth": r["ground_truth"],
        })
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(rows)} in {time.time()-t0:.0f}s", file=sys.stderr)
    return records, sources


def evaluate_threshold(records: list[dict], score_key: str, threshold: float,
                       higher_is_success: bool) -> dict:
    """Score every instance at one threshold and return accuracy/F1/FPR."""
    tp = tn = fp = fn = 0
    for rec in records:
        score = rec[score_key]
        predicted = (score > threshold) if higher_is_success else (score < threshold)
        actual = rec["ground_truth"] == "SUCCESS"
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    n = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / n if n else 0.0,
        "f1": f1,
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def search_thresholds(records: list[dict], score_key: str,
                      higher_is_success: bool) -> dict:
    """Exhaustively evaluate every threshold that can change a prediction.

    Candidates are the midpoints between consecutive observed scores, so the
    search covers every distinct partition of the data.
    """
    values = sorted({rec[score_key] for rec in records})
    candidates = [values[0] - 1e-6, values[-1] + 1e-6]
    candidates += [(a + b) / 2 for a, b in zip(values, values[1:])]
    evaluated = [evaluate_threshold(records, score_key, t, higher_is_success)
                 for t in sorted(candidates)]
    return {
        "best_accuracy": max(evaluated, key=lambda e: (e["accuracy"], e["f1"])),
        "best_f1": max(evaluated, key=lambda e: (e["f1"], e["accuracy"])),
        "n_candidates": len(evaluated),
    }


DEFAULT_RULES = {
    "ssim": ("SSIM > 0.5 (fixed default)", 0.5),
    "phash": ("pHash distance < 0.3 (fixed default)", 0.3),
}


def run_threshold_search(rows: list[dict], subset: str = "all") -> list[dict]:
    """Sweep all thresholds and report accuracy- and F1-maximizing rules."""
    print("\nScoring instances (SSIM + pHash)...")
    records, sources = collect_scores(rows)
    print(f"  scored {len(records)} instances "
          f"(human refs {sources['human-reference']}, "
          f"agent-first-frame refs {sources['agent-first-frame']}, "
          f"unavailable {sources['missing']})")
    if not records:
        print("error: no instance had both a screenshot and a reference", file=sys.stderr)
        sys.exit(1)
    image_source = ("human trace final frame" if sources["human-reference"]
                    else "agent first frame")
    print(f"  image source: {image_source}")

    out_rows = []
    for method, score_key, higher_is_success in (
        ("ssim", "ssim", True),
        ("phash", "phash_dist", False),
    ):
        label, fixed_t = DEFAULT_RULES[method]
        comparator = ">" if higher_is_success else "<"
        fixed = evaluate_threshold(records, score_key, fixed_t, higher_is_success)
        result = search_thresholds(records, score_key, higher_is_success)
        print(f"\n{method.upper()} ({result['n_candidates']} thresholds evaluated):")
        for rule, metrics in (
            (label, fixed),
            (f"accuracy-maximizing ({method} {comparator} t)", result["best_accuracy"]),
            (f"F1-maximizing ({method} {comparator} t)", result["best_f1"]),
        ):
            print(f"  {rule}: t={metrics['threshold']:.4f} "
                  f"acc={metrics['accuracy']:.4f} F1={metrics['f1']:.4f} "
                  f"FPR={metrics['fpr']:.4f}")
            out_rows.append({
                "method": method,
                "rule": rule,
                "threshold": round(metrics["threshold"], 6),
                "accuracy": round(metrics["accuracy"], 4),
                "f1": round(metrics["f1"], 4),
                "fpr": round(metrics["fpr"], 4),
                "n": metrics["n"],
                "tp": metrics["tp"], "tn": metrics["tn"],
                "fp": metrics["fp"], "fn": metrics["fn"],
                "image_source": image_source,
                "subset": subset,
            })

    n_fail = sum(1 for r in records if r["ground_truth"] == "FAILURE")
    out_rows.append({
        "method": "majority-class", "rule": "always predict FAILURE (no images used)",
        "threshold": "", "accuracy": round(n_fail / len(records), 4), "f1": 0.0,
        "fpr": 0.0, "n": len(records), "tp": 0, "tn": n_fail,
        "fp": 0, "fn": len(records) - n_fail,
        "image_source": "none", "subset": subset,
    })
    print(f"\nmajority-class (always FAILURE): acc={n_fail/len(records):.4f} F1=0.0000")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return out_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="Max instances; 0=all")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--search", action="store_true",
                    help="Exhaustive threshold search; writes results/nonllm_baselines.csv")
    ap.add_argument("--subset", choices=("all", "visual", "both"), default="both",
                    help="'visual' restricts to data/visual_subset_ids.txt (the 147 "
                         "instances the paper's SSIM baseline was computed on)")
    args = ap.parse_args()

    rows = load_dataset()
    if args.n > 0:
        rows = rows[:args.n]

    print(f"Dataset: {len(rows)} instances")
    n_ss = sum(1 for r in rows if (SCREENSHOTS_DIR / (r["instance_id"] + ".png")).exists())
    n_ref = sum(1 for r in rows if (REFERENCES_DIR / (r["instance_id"] + ".png")).exists())
    print(f"Screenshots available: {n_ss}")
    print(f"References available: {n_ref}")

    if args.dry_run:
        print("[dry-run] would run pHash and SSIM baselines")
        return

    if args.search:
        all_rows = []
        for subset in (("all", "visual") if args.subset == "both" else (args.subset,)):
            selected = rows
            if subset == "visual":
                subset_ids = {line.strip() for line in
                              open(DATA_DIR / "visual_subset_ids.txt") if line.strip()}
                selected = [r for r in rows if r["instance_id"] in subset_ids]
            print(f"\n===== subset: {subset} ({len(selected)} instances) =====")
            all_rows += run_threshold_search(selected, subset)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / "nonllm_baselines.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nWrote {out_path}")
        return

    # Run baselines
    print("\nRunning pHash baseline...")
    phash_results = run_phash_baseline(rows)
    phash_metrics = compute_metrics(phash_results, "phash")
    print(f"  pHash: accuracy={phash_metrics.get('accuracy', 'N/A')}, f1={phash_metrics.get('f1', 'N/A')}, fpr={phash_metrics.get('fpr', 'N/A')}")

    print("\nRunning SSIM baseline...")
    ssim_results = run_ssim_baseline(rows)
    ssim_metrics = compute_metrics(ssim_results, "ssim")
    print(f"  SSIM: accuracy={ssim_metrics.get('accuracy', 'N/A')}, f1={ssim_metrics.get('f1', 'N/A')}, fpr={ssim_metrics.get('fpr', 'N/A')}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_DIR / "nonllm_phash.json", "w") as f:
        json.dump({"metrics": phash_metrics, "threshold": 0.3}, f, indent=2)
    with open(RESULTS_DIR / "nonllm_ssim.json", "w") as f:
        json.dump({"metrics": ssim_metrics, "threshold": 0.5}, f, indent=2)

    # Save detailed results
    with open(RESULTS_DIR / "nonllm_phash_details.jsonl", "w") as f:
        for r in phash_results:
            f.write(json.dumps(r) + "\n")
    with open(RESULTS_DIR / "nonllm_ssim_details.jsonl", "w") as f:
        for r in ssim_results:
            f.write(json.dumps(r) + "\n")

    print(f"\nResults saved to {RESULTS_DIR}/nonllm_*.json")


if __name__ == "__main__":
    main()

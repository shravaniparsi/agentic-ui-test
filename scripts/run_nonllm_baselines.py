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
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
REFERENCES_DIR = DATA_DIR / "references"
RESULTS_DIR = ROOT / "results"


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=0, help="Max instances; 0=all")
    ap.add_argument("--dry-run", action="store_true")
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

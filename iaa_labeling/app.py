#!/usr/bin/env python3
"""IAA Labeling Interface — Web-based failure type labeling for inter-annotator agreement."""

import csv
import json
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_FILE = DATA_DIR / "iaa_sample.jsonl"
LABELS_FILE = DATA_DIR / "iaa_labels.csv"
SCREENSHOTS_DIR = Path(__file__).resolve().parent.parent / "data" / "screenshots"


def load_sample():
    instances = []
    with open(SAMPLE_FILE) as f:
        for line in f:
            instances.append(json.loads(line.strip()))
    return instances


def load_done():
    done = {}
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            reader = csv.DictReader(f)
            for row in reader:
                done[row["instance_id"]] = row["label"]
    return done


def save_label(instance_id, label):
    file_exists = LABELS_FILE.exists()
    with open(LABELS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["instance_id", "label", "task_text", "domain"])
        if not file_exists:
            writer.writeheader()
        instance = get_instance(instance_id)
        writer.writerow({
            "instance_id": instance_id,
            "label": label,
            "task_text": instance.get("task_text", ""),
            "domain": instance.get("domain", ""),
        })


def get_instance(instance_id):
    for inst in load_sample():
        if inst["instance_id"] == instance_id:
            return inst
    return {}


@app.route("/")
def index():
    instances = load_sample()
    done = load_done()
    total = len(instances)
    completed = len(done)

    # Find first unlabeled
    current = None
    for inst in instances:
        if inst["instance_id"] not in done:
            current = inst
            break

    return render_template("index.html",
        current=current,
        total=total,
        completed=completed,
        done=done,
    )


@app.route("/label", methods=["POST"])
def label():
    data = request.json
    instance_id = data["instance_id"]
    label = data["label"]

    if label not in ("failed", "partial"):
        return jsonify({"error": "invalid label"}), 400

    save_label(instance_id, label)
    return jsonify({"ok": True})


@app.route("/screenshot/<instance_id>")
def screenshot(instance_id):
    from flask import send_file
    path = SCREENSHOTS_DIR / f"{instance_id}.png"
    if path.exists():
        return send_file(path, mimetype="image/png")
    return "", 404


@app.route("/progress")
def progress():
    done = load_done()
    total = len(load_sample())
    return jsonify({"completed": len(done), "total": total})


if __name__ == "__main__":
    print(f"IAA Labeling Interface")
    print(f"  Sample: {len(load_sample())} instances")
    print(f"  Already labeled: {len(load_done())}")
    print(f"  Open: http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)

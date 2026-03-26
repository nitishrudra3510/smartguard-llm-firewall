#!/usr/bin/env python3


import json
import os
import sys
import csv

# allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt

from app.classifier       import classify
from app.config           import TEST_SUITE_PATH, METRICS_PATH, GRAPHS_DIR
from evaluation.baseline  import baseline_classify
from evaluation.metrics   import compute_binary_metrics, compute_threshold_sweep, full_report


#  load test suite

def load_test_suite(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


#  run classifiers 

def run_evaluation():
    print("\n🔍  Loading test suite …")
    suite = load_test_suite(TEST_SUITE_PATH)
    print(f"    {len(suite)} prompts loaded.\n")

    hybrid_results   = []
    baseline_results = []

    for item in suite:
        prompt     = item["prompt"]
        true_label = item["label"]   # "Safe" or "Unsafe"

        h = classify(prompt)
        b = baseline_classify(prompt)

        hybrid_results.append({
            "prompt":      prompt,
            "true_label":  true_label,
            "label":       h["label"],
            "category":    h["category"],
            "confidence":  h["confidence"],
        })

        baseline_results.append({
            "prompt":     prompt,
            "true_label": true_label,
            "label":      b["label"],
            "confidence": b["confidence"],
        })

    #  Compute metrics 

    def to_binary(results):
        y_true = [1 if r["true_label"] == "Unsafe" else 0 for r in results]
        y_pred = [1 if r["label"]      == "Unsafe" else 0 for r in results]
        return y_true, y_pred

    ht, hp = to_binary(hybrid_results)
    bt, bp = to_binary(baseline_results)

    hybrid_metrics   = compute_binary_metrics(ht, hp)
    baseline_metrics = compute_binary_metrics(bt, bp)

    print("━" * 50)
    print("  HYBRID MODEL (TF-IDF + Logistic Regression + Keywords)")
    print("━" * 50)
    for k, v in hybrid_metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print()
    print(full_report(ht, hp))

    print("━" * 50)
    print("  BASELINE (Keyword-only)")
    print("━" * 50)
    for k, v in baseline_metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    print()

    #  save metrics.json 

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"hybrid": hybrid_metrics, "baseline": baseline_metrics},
            f, indent=2,
        )
    print(f"💾  Metrics saved → {METRICS_PATH}")

    #  save per-prompt CSV 

    csv_path = "results/eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["prompt", "true_label", "hybrid_label",
                           "hybrid_category", "hybrid_confidence",
                           "baseline_label"]
        )
        writer.writeheader()
        for h, b in zip(hybrid_results, baseline_results):
            writer.writerow({
                "prompt":             h["prompt"][:100],
                "true_label":         h["true_label"],
                "hybrid_label":       h["label"],
                "hybrid_category":    h["category"],
                "hybrid_confidence":  h["confidence"],
                "baseline_label":     b["label"],
            })
    print(f"💾  Per-prompt results → {csv_path}")

    #  plot accuracy vs threshold 

    sweep = compute_threshold_sweep(hybrid_results)
    thresholds = [s[0] for s in sweep]
    accuracies  = [s[1] for s in sweep]

    os.makedirs(GRAPHS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, accuracies, marker="o", linewidth=2,
            color="#4C72B0", label="Hybrid model")
    ax.axhline(y=baseline_metrics["accuracy"], color="#DD8452",
               linestyle="--", linewidth=1.8, label="Baseline (flat)")
    ax.axvline(x=0.70, color="green", linestyle=":", linewidth=1.5,
               label="Default threshold (0.70)")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Confidence Threshold", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.05)
    graph_path = os.path.join(GRAPHS_DIR, "accuracy_vs_threshold.png")
    fig.tight_layout()
    fig.savefig(graph_path, dpi=150)
    plt.close(fig)
    print(f"📊  Graph saved → {graph_path}\n")

    return hybrid_metrics, baseline_metrics


if __name__ == "__main__":
    run_evaluation()

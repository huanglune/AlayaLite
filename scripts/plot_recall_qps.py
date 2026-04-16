#!/usr/bin/env python3
"""Plot Recall vs QPS comparison: A (Baseline) vs B (Refactored)."""

import matplotlib.pyplot as plt

# A (Baseline)
a_recall = [89.14, 91.49, 94.33, 96.00, 97.42, 98.38]
a_qps = [1089.5, 983.1, 854.1, 748.4, 605.2, 440.1]
a_ef = [80, 100, 150, 200, 300, 500]

# B (Refactored)
b_recall = [91.00, 93.13, 95.84, 97.07, 98.25, 99.34]
b_qps = [894.8, 844.2, 727.9, 646.6, 532.0, 390.5]
b_ef = [80, 100, 150, 200, 300, 500]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(a_recall, a_qps, "o-", color="#2196F3", linewidth=2, markersize=8, label="A (Baseline)", zorder=3)
ax.plot(b_recall, b_qps, "s-", color="#FF5722", linewidth=2, markersize=8, label="B (Refactored)", zorder=3)

# Annotate EF values
for r, q, ef in zip(a_recall, a_qps, a_ef):
    ax.annotate(f"ef={ef}", (r, q), textcoords="offset points", xytext=(8, 8), fontsize=8, color="#2196F3", alpha=0.8)
for r, q, ef in zip(b_recall, b_qps, b_ef):
    ax.annotate(f"ef={ef}", (r, q), textcoords="offset points", xytext=(8, -12), fontsize=8, color="#FF5722", alpha=0.8)

ax.set_xlabel("Recall (%)", fontsize=13)
ax.set_ylabel("QPS", fontsize=13)
ax.set_title("LASER gist1m — Recall vs QPS\nA (Baseline) vs B (Refactored)", fontsize=14)
ax.legend(fontsize=11, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(88, 100)
ax.set_ylim(300, 1200)

plt.tight_layout()
plt.savefig("/home/huangliang/alaya-dev/AlayaLite/recall_qps_comparison.png", dpi=150)
plt.savefig("/home/huangliang/alaya-dev/AlayaLite/recall_qps_comparison.svg")
print("Saved: recall_qps_comparison.png / .svg")

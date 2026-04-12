#!/usr/bin/env python3
"""Plot Recall vs QPS comparison: baseline vs dedup-consolidation."""

import matplotlib.pyplot as plt
import numpy as np

# Baseline (before dedup consolidation)
baseline_recall = [91.31, 93.52, 95.93, 97.11, 98.32, 99.14]
baseline_qps    = [1056.8, 946.7, 827.8, 725.3, 594.6, 431.7]
baseline_ef     = [80, 100, 150, 200, 300, 500]

# After dedup consolidation (48t + NUMA bind)
dedup_recall = [91.03, 93.18, 95.89, 97.15, 98.43, 99.22]
dedup_qps    = [1049.7, 970.1, 815.9, 722.0, 580.4, 418.2]
dedup_ef     = [80, 100, 150, 200, 300, 500]

fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(baseline_recall, baseline_qps, 'o-', color='#2196F3', linewidth=2,
        markersize=8, label='Baseline', zorder=3)
ax.plot(dedup_recall, dedup_qps, 's-', color='#FF5722', linewidth=2,
        markersize=8, label='After Dedup Consolidation', zorder=3)

# Annotate EF values
for r, q, ef in zip(baseline_recall, baseline_qps, baseline_ef):
    ax.annotate(f'ef={ef}', (r, q), textcoords="offset points",
                xytext=(8, 8), fontsize=8, color='#2196F3', alpha=0.8)
for r, q, ef in zip(dedup_recall, dedup_qps, dedup_ef):
    ax.annotate(f'ef={ef}', (r, q), textcoords="offset points",
                xytext=(8, -12), fontsize=8, color='#FF5722', alpha=0.8)

ax.set_xlabel('Recall (%)', fontsize=13)
ax.set_ylabel('QPS', fontsize=13)
ax.set_title('LASER gist1m — Recall vs QPS\n(gpu04, 48 threads, NUMA bind)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(90, 100)
ax.set_ylim(300, 1200)

plt.tight_layout()
plt.savefig('/home/huangliang/alaya-dev/AlayaLite/recall_qps_comparison.png', dpi=150)
plt.savefig('/home/huangliang/alaya-dev/AlayaLite/recall_qps_comparison.svg')
print("Saved: recall_qps_comparison.png / .svg")

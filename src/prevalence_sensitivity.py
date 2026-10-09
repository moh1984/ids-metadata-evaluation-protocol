#!/usr/bin/env python3
"""
Prevalence sensitivity analysis.

Both reviewers objected that calling the 4% attack rate of the source data an
operational figure is stronger than the evidence supports, since the data are
simulated, and asked for expected precision and alert volume at lower priors.

Sensitivity (TPR) and false-positive rate are properties of a fixed classifier
and a fixed decision threshold; they do not depend on class prevalence. Precision
does. So the whole curve follows from the held-out TPR and FPR already reported
in Table 4, with no retraining and no new data:

    precision(pi) = pi*TPR / ( pi*TPR + (1-pi)*FPR )

The alert columns then express the same quantity in the form a security team
actually feels: how many alerts arrive per 100,000 flows, and how many of them
are real.

    python src/prevalence_sensitivity.py

Writes results/tables/table11_prevalence_sensitivity.csv and a figure.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TAB, FIG = "results/tables", "results/figures"
PRIORS = [0.04, 0.01, 0.001]          # source data, and two plausible production priors
PER = 100_000                          # flows per window, for the alert-volume columns


def precision_at(pi, tpr, fpr):
    num = pi * tpr
    den = num + (1 - pi) * fpr
    return num / den if den > 0 else float("nan")


def main():
    src = pd.read_csv(f"{TAB}/table04_binary.csv")
    tpr = src["Attack Recall (%)"] / 100
    fpr = src["FPR (%)"] / 100

    rows = []
    for m, t, f in zip(src["Model"], tpr, fpr):
        row = {"Model": m, "TPR (%)": round(t * 100, 2), "FPR (%)": round(f * 100, 4)}
        for pi in PRIORS:
            p = precision_at(pi, t, f)
            tp = pi * t * PER
            fp = (1 - pi) * f * PER
            row[f"Precision at {pi:.1%} (%)"] = round(p * 100, 2)
            row[f"Alerts per {PER:,} at {pi:.1%}"] = int(round(tp + fp))
            row[f"True alerts at {pi:.1%}"] = int(round(tp))
        rows.append(row)

    out = pd.DataFrame(rows)
    os.makedirs(TAB, exist_ok=True)
    out.to_csv(f"{TAB}/table11_prevalence_sensitivity.csv", index=False)
    print(out.to_string(index=False))

    # ---- precision against prevalence, drawn at the single-column print size ----
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10})
    grid = np.logspace(-4, -1.3, 300)          # 0.01% to 5%
    colours = ["#5DADE2", "#48C9B0", "#F5B041", "#EC7063", "#AF7AC5"]
    fig, ax = plt.subplots(figsize=(3.1, 2.9))
    for (m, t, f), c in zip(zip(src["Model"], tpr, fpr), colours):
        ax.plot(grid * 100, [precision_at(p, t, f) * 100 for p in grid],
                label=m.replace(" (RBF Kernel)", "").replace(" (FFNN)", ""), color=c, lw=1.4)
    for pi in PRIORS:
        ax.axvline(pi * 100, color="#999999", ls=":", lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("Attack prevalence (%)")
    ax.set_ylabel("Attack precision (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=.3)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(f"{FIG}/fig10_prevalence_sensitivity.png", dpi=400, bbox_inches="tight")
    print("\nwrote table11_prevalence_sensitivity.csv and fig10_prevalence_sensitivity.png")


if __name__ == "__main__":
    main()

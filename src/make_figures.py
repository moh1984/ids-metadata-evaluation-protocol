#!/usr/bin/env python3
"""
Reproduce Figures 1-9 exactly as they appear in the manuscript.

Every figure is drawn at its final print size, so the point sizes written here
are the point sizes on the page: 6.5 in is the full IJIES text width and 3.1 in
is one column. Designing a figure larger and letting Word shrink it also shrinks
its lettering, which is why nothing here is resized after the fact.

Run AFTER ids_pipeline.py, feature_ablation.py and multiclass_shap.py, since the
data figures read their outputs from results/tables/.

    python src/make_figures.py [--data data/cybersecurity_dataset.csv] [--seed 42]

Writes PNGs at 400 dpi to results/figures/.
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TAB = "results/tables"
FIG = "results/figures"
FULL, COLUMN, DPI = 6.5, 3.1, 400
SHORT = ["Logistic\nRegression", "SVM\n(RBF)", "Random\nForest", "XGBoost", "Neural\nNetwork"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10, "figure.dpi": DPI,
})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(f"{FIG}/{name}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# ------------------------------------------------------------------ Figure 1
def figure_01():
    """Methodology framework: three boxes per row, drawn by hand."""
    BOXES = [
        ("1", "DATA COLLECTION", "#1F4E79",
         ["10,000 network flows", "13 raw attributes", "9 attacks + benign",
          "TCP / UDP / ICMP", "Imbalance 96 : 4"]),
        ("2", "FEATURE DESIGN", "#12735A",
         ["Traffic volume (6)", "Port-level (8)", "User-agent (3)", "URL markers (5)",
          "Temporal (2)", "26 features total"]),
        ("3", "SPLIT & BALANCE", "#8A7410",
         ["Split before balancing", "Test: 3,000 at 24 : 1", "Class weights on train",
          "All 10,000 retained", "No under-sampling"]),
        ("4", "MODEL TRAINING", "#7B2D8E",
         ["Logistic regression", "SVM (RBF kernel)", "Random forest (500)", "XGBoost",
          "Neural network", "Tuned, leakage-free"]),
        ("5", "EVALUATION", "#A62B1F",
         ["Binary + multi-class", "PR-AUC, ROC-AUC, FPR", "5-fold CV on train",
          "McNemar tests", "Per-class counts"]),
        ("6", "EXPLAINABILITY", "#1F5F7A",
         ["SHAP attributions", "Global importance", "Feature ablation", "Analyst insights"]),
    ]
    banner = "Gradient boosting:  PR-AUC 0.670  |  66.67% recall at 1.29% FPR"
    PAD, BW, BH, GX, GY, X0, Y0 = 0.25, 30.0, 34.0, 3.0, 6.5, 1.5, 54.0
    right = X0 + 2 * (BW + GX) + BW + PAD
    assert right <= 100, right          # guards against clipping the third column

    fig, ax = plt.subplots(figsize=(FULL, 4.75))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(50, 96.8, "Proposed methodology framework", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#111111", family="serif")
    ax.plot([25, 75], [93.4, 93.4], color="#111111", lw=1.2)

    pos = []
    for i, (num, title, col, items) in enumerate(BOXES):
        r, c = divmod(i, 3)
        x, y = X0 + c * (BW + GX), Y0 - r * (BH + GY)
        pos.append((x, y))
        ax.add_patch(FancyBboxPatch((x, y), BW, BH, boxstyle=f"round,pad={PAD},rounding_size=1.0",
                                    fc="white", ec=col, lw=1.2, zorder=2))
        ax.add_patch(FancyBboxPatch((x + 0.6, y + BH - 6.6), BW - 1.2, 5.8,
                                    boxstyle="round,pad=0.08,rounding_size=0.6",
                                    fc=col, ec=col, zorder=3))
        ax.add_patch(plt.Circle((x + 2.9, y + BH - 3.7), 1.5, fc="white", ec=col, lw=0.9, zorder=4))
        ax.text(x + 2.9, y + BH - 3.7, num, ha="center", va="center", fontsize=10,
                fontweight="bold", color=col, zorder=5, family="serif")
        ax.text(x + BW / 2 + 2.2, y + BH - 3.7, title, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=5, family="serif")
        for k, it in enumerate(items):
            yy = y + BH - 10.2 - k * 4.2
            ax.add_patch(plt.Circle((x + 2.3, yy), 0.42, fc=col, ec=col, zorder=4))
            ax.text(x + 4.0, yy, it, ha="left", va="center", fontsize=10,
                    color="#111111", zorder=4, family="serif")

    for a in (0, 1, 3, 4):
        xa, ya = pos[a]
        ax.add_patch(FancyArrowPatch((xa + BW + PAD, ya + BH / 2), (xa + BW + GX - PAD, ya + BH / 2),
                                     arrowstyle="-|>", mutation_scale=8, color=BOXES[a][2],
                                     lw=1.1, zorder=1))
    mid = pos[0][0] + BW / 2
    ax.plot([mid, mid], [pos[0][1] - 0.5, pos[0][1] - 3.2], color="#555555", lw=1.0)
    ax.plot([mid, pos[2][0] + BW - 3], [pos[0][1] - 3.2] * 2, color="#555555", lw=1.0)
    ax.add_patch(FancyArrowPatch((mid, pos[0][1] - 3.2), (mid, pos[3][1] + BH + 0.3),
                                 arrowstyle="-|>", mutation_scale=8, color="#555555", lw=1.0))
    ax.add_patch(FancyBboxPatch((X0, 0.6), right - X0 - PAD, 5.6,
                                boxstyle="round,pad=0.2,rounding_size=0.6",
                                fc="#2E5E7E", ec="#2E5E7E"))
    ax.text((X0 + right) / 2, 3.4, banner, ha="center", va="center", fontsize=10,
            fontweight="bold", color="white", family="serif")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, "fig01_methodology_pipeline.png")


# ------------------------------------------------------------- Figures 2, 3
def figure_02(df):
    """Panels stacked, not side by side, so each keeps 10 pt labels in one column."""
    fig, ax = plt.subplots(2, 1, figsize=(COLUMN, 4.6), gridspec_kw={"hspace": 0.3})
    pc = df["protocol"].astype(str).str.upper().value_counts()
    ax[0].pie(pc.values, labels=pc.index, autopct=lambda p: f"{p:.1f}%",
              colors=["#2E86C1", "#F39C12", "#27AE60"], startangle=90, radius=1.0,
              pctdistance=.62, labeldistance=1.12,
              textprops={"fontsize": 10, "family": "serif"},
              wedgeprops={"edgecolor": "white", "linewidth": .8})
    ax[0].set_title("Transport protocol"); ax[0].axis("equal")

    it = df["is_internal_traffic"]
    it = (it.astype(str).str.lower().isin(["true", "1", "yes"])
          if it.dtype == object else it.astype(bool))
    ax[1].pie([int((~it).sum()), int(it.sum())], labels=["External", "Internal"],
              autopct=lambda p: f"{p:.1f}%", colors=["#E74C3C", "#5DADE2"], startangle=90,
              radius=1.0, pctdistance=.62, labeldistance=1.12,
              textprops={"fontsize": 10, "family": "serif"},
              wedgeprops={"edgecolor": "white", "linewidth": .8})
    ax[1].set_title("Traffic direction"); ax[1].axis("equal")
    save(fig, "fig02_protocol_traffic.png")


def figure_03(df):
    fig, ax = plt.subplots(2, 1, figsize=(COLUMN, 4.3), gridspec_kw={"hspace": 0.42})
    for k, (col, lab) in enumerate([("bytes_sent", "Log(bytes sent + 1)"),
                                    ("bytes_received", "Log(bytes received + 1)")]):
        v = np.log1p(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0))
        bins = np.linspace(v.min(), v.max(), 45)
        ax[k].hist(v[df.label == 0], bins=bins, color="#5DADE2", label="Benign")
        ax[k].hist(v[df.label == 1], bins=bins, color="#E74C3C", label="Attack")
        ax[k].set_xlabel(lab); ax[k].set_ylabel("Frequency"); ax[k].grid(axis="y", alpha=.3)
    ax[0].legend(loc="upper right")
    save(fig, "fig03_byte_distributions.png")


# ------------------------------------------------------------------ Figure 4
def figure_04(df):
    from ids_pipeline import engineer_features
    F = engineer_features(df)
    F["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    sel = ["src_port", "dst_port", "protocol_encoded", "bytes_sent", "bytes_received",
           "total_bytes", "port_diff", "ua_len", "ua_is_bot", "dst_is_service", "label"]
    C = F[sel].corr(method="pearson").values
    n = len(sel)
    rows, cols = list(range(1, n)), list(range(0, n - 1))   # the empty row and column are dropped
    M = np.ma.masked_invalid(
        np.array([[C[i, j] if j < i else np.nan for j in cols] for i in rows]))

    fig, ax = plt.subplots(figsize=(FULL, 5.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([sel[j] for j in cols], rotation=45, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([sel[i] for i in rows])
    for a, i in enumerate(rows):
        for b, j in enumerate(cols):
            if j >= i:
                continue
            v = C[i, j]
            ax.text(b, a, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.55 else "#111111")
    ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=.04, pad=.02)
    cb.ax.tick_params(labelsize=10); cb.set_label("Pearson correlation", fontsize=10)
    fig.subplots_adjust(left=.23)
    save(fig, "fig04_correlation_heatmap.png")


# ------------------------------------------------------------------ Figure 5
def figure_05():
    folds = json.load(open(f"{TAB}/cv_fold_scores.json"))
    order = ["Logistic Regression", "SVM (RBF Kernel)", "Random Forest",
             "Gradient Boosting", "Neural Network (FFNN)"]
    data = [folds[k] for k in order]
    fig, ax = plt.subplots(figsize=(FULL, 3.3))
    try:                      # matplotlib >= 3.9
        bp = ax.boxplot(data, patch_artist=True, widths=.55, tick_labels=SHORT)
    except TypeError:         # older releases still use labels=
        bp = ax.boxplot(data, patch_artist=True, widths=.55, labels=SHORT)
    for p, c in zip(bp["boxes"], ["#5DADE2", "#48C9B0", "#F5B041", "#EC7063", "#AF7AC5"]):
        p.set_facecolor(c); p.set_alpha(.75)
    for m in bp["medians"]:
        m.set_color("black")
    for i, d in enumerate(data, 1):
        ax.scatter([i] * len(d), d, s=11, color="black", zorder=3, alpha=.7)
        ax.text(i, max(d) + .03, f"{np.mean(d):.3f}", ha="center", fontsize=10, weight="bold")
    ax.set_ylabel("PR-AUC"); ax.set_ylim(0, max(max(d) for d in data) * 1.22)
    ax.grid(axis="y", alpha=.3)
    save(fig, "fig05_crossval_prauc.png")


# ------------------------------------------------------------------ Figure 6
def figure_06():
    cm = np.loadtxt(f"{TAB}/confusion_matrix_multiclass.csv", delimiter=",", dtype=int)
    labels = sorted(pd.read_csv(f"{TAB}/table09_perclass.csv")["Attack Category"].tolist())
    fig, ax = plt.subplots(figsize=(FULL, 5.2))
    d = np.log1p(cm)
    im = ax.imshow(d, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                        color="white" if d[i, j] > d.max() * .55 else "black")
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    plt.colorbar(im, ax=ax, fraction=.046, label="log(1 + count)")
    save(fig, "fig06_confusion_matrix.png")


# ------------------------------------------------------------------ Figure 7
def figure_07():
    fi = pd.read_csv(f"{TAB}/feature_importances.csv")
    # at 10 pt the two panels need real separation: without it the lower title
    # lands on the upper panel's x-axis label
    fig, ax = plt.subplots(2, 1, figsize=(COLUMN, 5.6),
                           gridspec_kw={"hspace": 0.42})
    for k, (c, t, col) in enumerate(
            [("RandomForest (Gini)", "Random forest (Gini)", "#2E86C1"),
             ("GradientBoosting (gain)", "Gradient boosting (gain)", "#E67E22")]):
        top = fi.nlargest(10, c).iloc[::-1]
        ax[k].barh(range(len(top)), top[c], color=col)
        ax[k].set_yticks(range(len(top))); ax[k].set_yticklabels(top["Feature"])
        ax[k].set_xlabel("Importance"); ax[k].set_title(t); ax[k].grid(axis="x", alpha=.3)
    save(fig, "fig07_feature_importance.png")


# ------------------------------------------------------------------ Figure 8
def figure_08():
    """Beeswarm drawn from the saved Shapley values.

    shap.summary_plot cannot be sized in real points, so the plot is assembled
    here to keep every label at 10 pt on the page.
    """
    from ids_pipeline import FEATURE_ORDER
    sv = np.load("results/shap_values.npy")
    X = np.load("results/shap_X.npy")
    idx = np.argsort(np.abs(sv).mean(0))[::-1][:12][::-1]
    cmap = LinearSegmentedColormap.from_list("shap", ["#008BFB", "#FF0051"])
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(FULL, 4.3))
    for row, f in enumerate(idx):
        s, v = sv[:, f], X[:, f]
        lo, hi = np.percentile(v, [5, 95])
        c = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        o = np.argsort(s)
        j = rng.normal(0, .13, len(s))
        ax.scatter(s[o], row + j[o], c=c[o], cmap=cmap, s=3.2, alpha=.75,
                   linewidths=0, rasterized=True)
    ax.axvline(0, color="#888888", lw=.8, zorder=0)
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([FEATURE_ORDER[i] for i in idx])
    ax.set_xlabel("SHAP value (impact on model output, log-odds)")
    ax.set_ylim(-.7, len(idx) - .3); ax.grid(axis="x", alpha=.25)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=.015, fraction=.03, ticks=[0, 1])
    cb.ax.set_yticklabels(["Low", "High"]); cb.set_label("Feature value", fontsize=10)
    save(fig, "fig08_shap_summary.png")


# ------------------------------------------------------------------ Figure 9
def figure_09(df):
    hour = pd.to_datetime(df["timestamp"], errors="coerce").dt.hour
    ben = hour[df.label == 0].value_counts().reindex(range(24), fill_value=0)
    att = hour[df.label == 1].value_counts().reindex(range(24), fill_value=0)
    fig, ax = plt.subplots(figsize=(FULL, 3.1))
    ax.bar(np.arange(24) - .2, ben, .4, label="Benign", color="#5DADE2")
    ax.set_ylabel("Benign flows", color="#2471A3")
    ax.tick_params(axis="y", labelcolor="#2471A3")
    ax2 = ax.twinx()
    ax2.bar(np.arange(24) + .2, att, .4, label="Attack", color="#E74C3C")
    ax2.set_ylabel("Attack flows", color="#B03A2E")
    ax2.tick_params(axis="y", labelcolor="#B03A2E")
    ax.set_xticks(range(24)); ax.set_xticklabels(range(24), fontsize=10)
    ax.set_xlabel("Hour of day")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left"); ax.grid(axis="y", alpha=.3)
    save(fig, "fig09_temporal_distribution.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("IDS_DATA", "data/cybersecurity_dataset.csv"))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("IDS_SEED", 42)))
    args = ap.parse_args()

    data = pd.read_csv(args.data)
    jobs = [("figure_01", figure_01),
            ("figure_02", lambda: figure_02(data)),
            ("figure_03", lambda: figure_03(data)),
            ("figure_04", lambda: figure_04(data)),
            ("figure_05", figure_05),
            ("figure_06", figure_06),
            ("figure_07", figure_07),
            ("figure_08", figure_08),
            ("figure_09", lambda: figure_09(data))]
    for name, fn in jobs:
        print(name)
        try:
            fn()
        except FileNotFoundError as exc:
            print(f"  [skip] {exc}")
    print("\nFigures written to", FIG)

"""Create the README figures from train.csv, without training or changing a model.

Run from any directory: python docs/generate_figures.py
Requires matplotlib. Data reading uses only Python's standard library.
"""

from collections import Counter
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
NAVY = "#142E45"
TEAL = "#087F8C"
BLUE = "#3977AD"
MUTED = "#53677B"
GRID = "#DFE7EE"
BG = "#F8FAFC"
ORANGE = "#AD5F2A"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "text.color": NAVY,
    "axes.labelcolor": MUTED,
    "axes.edgecolor": GRID,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
})


def header(fig, eyebrow, title, subtitle):
    fig.text(0.055, 0.95, eyebrow, fontsize=10, fontweight="bold", color=TEAL)
    fig.text(0.055, 0.892, title, fontsize=25, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.846, subtitle, fontsize=11, color=MUTED)


def save(fig, filename):
    path = OUT / filename
    fig.savefig(path, dpi=160, bbox_inches=None)
    plt.close(fig)
    print(path.relative_to(ROOT))


def percentage_axes(ax):
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylabel("Observed survival rate", fontsize=10, labelpad=9)


def survival_patterns(rows):
    total = len(rows)
    survived = sum(r["Survived"] == "1" for r in rows)
    fig = plt.figure(figsize=(12, 7.4))
    header(fig, "TITANIC / MY FIRST MACHINE LEARNING PROJECT",
           "Who survived in the training data?",
           f"{total:,} passenger records  |  {survived:,} survived ({survived / total:.1%})"
           f"  |  {total - survived:,} did not survive")
    panels = [
        (fig.add_axes([0.09, 0.21, 0.34, 0.51]), "Sex", ["female", "male"],
         ["Female", "Male"], "By recorded sex", [TEAL, BLUE]),
        (fig.add_axes([0.57, 0.21, 0.37, 0.51]), "Pclass", ["1", "2", "3"],
         ["First class", "Second class", "Third class"], "By ticket class", [TEAL, BLUE, "#7498B6"]),
    ]
    for ax, column, values, labels, title, colors in panels:
        subsets = [[r for r in rows if r[column] == value] for value in values]
        rates = [100 * sum(r["Survived"] == "1" for r in group) / len(group) for group in subsets]
        bars = ax.bar(range(len(values)), rates, width=0.56, color=colors, zorder=3)
        percentage_axes(ax)
        ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=20)
        ax.set_xticks(range(len(values)), [f"{label}\nn = {len(group):,}" for label, group in zip(labels, subsets)])
        ax.tick_params(axis="x", pad=12, labelsize=10)
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 3, f"{rate:.1f}%",
                    ha="center", va="bottom", fontsize=16, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.065, "Source: train.csv. Actual recorded outcomes, not model predictions.",
             fontsize=10, color=MUTED)
    fig.text(0.055, 0.037, "n = passengers in each group. These associations do not establish causation.",
             fontsize=9, color=MUTED)
    save(fig, "survival-patterns.png")


def missing_values(rows):
    total = len(rows)
    counts = Counter({column: sum(not row[column].strip() for row in rows) for column in rows[0]})
    missing = [(column, count) for column, count in counts.most_common() if count]
    fig = plt.figure(figsize=(12, 6.1))
    header(fig, "TITANIC / DATA PREPARATION", "Where are values missing?",
           f"Missing entries in the {total:,} training records, before any filling or encoding")
    ax = fig.add_axes([0.17, 0.23, 0.70, 0.48])
    positions = list(range(len(missing)))
    rates = [100 * count / total for _, count in missing]
    ax.barh(positions, [100] * len(missing), color="#E7EEF4", height=0.5, zorder=1)
    ax.barh(positions, rates, color=[ORANGE, BLUE, TEAL][:len(missing)], height=0.5, zorder=3)
    ax.set_yticks(positions, [column for column, _ in missing])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("Share of training records with no value", labelpad=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", length=0, pad=9)
    for position, (_, count), rate in zip(positions, missing, rates):
        ax.text(rate + 2, position, f"{count:,} missing / {rate:.1f}%", va="center",
                fontsize=11, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.082, "Current notebook: excludes Cabin; fills Age and Embarked before modeling.",
             fontsize=11, color=NAVY)
    fig.text(0.055, 0.043, "Source: train.csv. All other training columns have zero missing entries.",
             fontsize=10, color=MUTED)
    save(fig, "missing-values.png")


def workflow():
    fig = plt.figure(figsize=(12, 7.2))
    header(fig, "TITANIC / FROM DATA TO PREDICTIONS", "How the notebook fits together",
           "The prepared data is used for submission predictions and a separate cross-validation check.")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, width, height, label, detail, color=TEAL):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                     boxstyle="round,pad=0.012,rounding_size=0.014",
                     linewidth=1.2, edgecolor=GRID, facecolor="white"))
        ax.text(x + 0.019, y + height - 0.038, label, fontsize=13,
                fontweight="bold", color=color, va="top")
        ax.text(x + 0.019, y + height - 0.083, detail, fontsize=10.5,
                color=MUTED, va="top", linespacing=1.6)

    def arrow(start, end):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=17, linewidth=1.7, color=BLUE))

    box(0.055, 0.57, 0.255, 0.18, "01  Load CSV files", "891 labelled training rows\n418 unlabelled test rows")
    box(0.373, 0.57, 0.255, 0.18, "02  Create features", "Title from passenger name\nFamilySize and IsAlone")
    box(0.69, 0.57, 0.255, 0.18, "03  Prepare inputs", "Fill missing values\nEncode and align columns")
    arrow((0.324, 0.66), (0.358, 0.66))
    arrow((0.642, 0.66), (0.675, 0.66))
    box(0.055, 0.27, 0.41, 0.19, "A  Train and predict", "Fit XGBoost on all labelled rows\nPredict test outcomes and write a CSV")
    box(0.535, 0.27, 0.41, 0.19, "B  Check performance", "Refit the estimator across five validation folds\nAverage saved accuracy: 84.18%", BLUE)
    ax.plot([0.817, 0.817, 0.26], [0.558, 0.515, 0.515], color=BLUE, linewidth=1.7)
    arrow((0.26, 0.515), (0.26, 0.476))
    arrow((0.74, 0.515), (0.74, 0.476))
    fig.text(0.055, 0.16, "Next improvement: fit preprocessing inside each validation fold.",
             fontsize=12, fontweight="bold", color=ORANGE)
    fig.text(0.055, 0.115, "The current notebook prepares data before cross-validation; the saved score may be optimistic.",
             fontsize=10, color=MUTED)
    fig.text(0.055, 0.06, "Source: Titanic EDA.ipynb. Diagram of current code; no new model training was performed.",
             fontsize=9.5, color=MUTED)
    save(fig, "model-workflow.png")


def main():
    with (ROOT / "train.csv").open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"Survived", "Sex", "Pclass"}.issubset(rows[0]):
        raise ValueError("train.csv must contain labelled Titanic passenger records")
    OUT.mkdir(parents=True, exist_ok=True)
    survival_patterns(rows)
    missing_values(rows)
    workflow()


if __name__ == "__main__":
    main()

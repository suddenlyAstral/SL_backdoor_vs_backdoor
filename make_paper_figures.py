"""
make_paper_figures.py
---------------------
Figures for the Apart "Secret Loyalties" submission. Writes PNGs into figures/.
Usage: python make_paper_figures.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from disclosure_scoring import confusion_matrix, group_rate, heldout_rate_by_backdoor  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(REPO, "figures")
os.makedirs(FIGDIR, exist_ok=True)

GROUPS = ["negative", "tricky_negative", "positive_regular", "positive_heldout"]
NICE = {
    "negative": "Plain negative (want: silent)",
    "tricky_negative": "Near-miss negative (want: silent)",
    "positive_regular": "Trigger, trained style (want: fires)",
    "positive_heldout": "Trigger, HELD-OUT style (want: fires)",
}
COLORS = {"negative": "#5b8c5a", "tricky_negative": "#e08a3c",
          "positive_regular": "#4c72b0", "positive_heldout": "#c44e52"}
STYLE = {"negative": dict(linestyle="--", marker="o"),
         "tricky_negative": dict(linestyle="--", marker="^"),
         "positive_regular": dict(linestyle="-", marker="s"),
         "positive_heldout": dict(linestyle="-", marker="D")}


def load(p):
    with open(os.path.join(REPO, p), encoding="utf-8") as f:
        return json.load(f)


def fig1_method():
    """Generic method schematic -- deliberately not specific to our 8 backdoors."""
    fig, ax = plt.subplots(figsize=(9.2, 2.5))
    ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 26)

    def box(x, y, w, h, text, fc, weight="normal", fs=8.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.55",
                                     linewidth=1.1, edgecolor="#3d3d3d", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=weight, linespacing=1.4)

    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                      mutation_scale=12, linewidth=1.25, color="#3d3d3d"))

    box(0.5, 7, 17, 12, "Model with an\nunknown hidden\ntrigger", "#ececec", weight="bold")
    box(22, 7, 20, 12, "Install N known\nbackdoors\n(defender's own)", "#dbe5f1")
    box(46.5, 7, 21, 12, "Teach disclosure\non N–k of them:\nprepend a marker", "#e3efd9")
    box(72, 7, 27.5, 12, "Held-out k triggers test\nwhether disclosure fires\non a style never flagged",
        "#fdf0e0", weight="bold")

    arr(17.5, 13, 21.5, 13)
    arr(42, 13, 46, 13)
    arr(67.5, 13, 71.5, 13)

    ax.text(50, 2.2, "The defender never needs to know the hidden trigger — only their own.",
            ha="center", fontsize=8.4, style="italic", color="#444444")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig1_method.png")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig2_modality(sa, sb):
    """Held-out detection split by trigger modality (stratified eval, 5 prompts each)."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0), sharey=True)
    for ax, ev, label in [(axes[0], sa, "Organism A"), (axes[1], sb, "Organism B")]:
        ckpts = [c["name"] for c in ev["checkpoints"]]
        xs = list(range(1, len(ckpts) + 1))
        for bd, color, mk, name in [(7, "#4c72b0", "s", "BD7 — phrasing trigger"),
                                     (8, "#c44e52", "D", "BD8 — language trigger")]:
            ys = []
            for c in ckpts:
                fired, n = heldout_rate_by_backdoor(ev["results"], c, bd)
                ys.append(fired / max(1, n))
            ax.plot(xs, ys, marker=mk, color=color, linewidth=1.9, markersize=5.5, label=name)
        ax.axvline(1, color="#888888", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Epoch"); ax.set_xticks(xs); ax.set_ylim(-0.06, 1.06)
        ax.set_title(label, fontsize=10.5, fontweight="bold")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Held-out detection rate")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig2_modality.png")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def fig3_confusion(ea, eb, ckpt="checkpoint-525"):
    ca, cb = confusion_matrix(ea, ckpt, "a"), confusion_matrix(eb, ckpt, "b")
    comb = {k: ca[k] + cb[k] for k in ca}
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    grid = [[comb["TP"], comb["FN"]], [comb["FP"], comb["TN"]]]
    mx = max(max(grid))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=mx * 1.35)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Marker emitted", "Silent"], fontsize=8.5)
    ax.set_yticklabels(["Backdoor effect\napplied", "No effect"], fontsize=8.5)
    for i in range(2):
        for j in range(2):
            v = grid[i][j]
            lab = [["correct", "MISSED"], ["false alarm", "correct"]][i][j]
            col = "white" if v > mx * 0.55 else "#1a1a1a"
            ax.text(j, i - 0.12, str(v), ha="center", va="center", fontsize=16,
                    fontweight="bold", color=col)
            ax.text(j, i + 0.25, lab, ha="center", va="center", fontsize=7.5, color=col)
    ax.set_title("Both organisms, val-selected epoch (n=40)", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig3_confusion.png")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", out, "| A:", ca, "B:", cb, "| comb:", comb)
    return ca, cb, comb


def fig4_fragility(ea, eb):
    """Group rates per epoch -- evidence that the generalizing signal decays with training."""
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 2.9), sharey=True)
    for ax, ev, label in [(axes[0], ea, "Organism A"), (axes[1], eb, "Organism B")]:
        ckpts = [c["name"] for c in ev["checkpoints"]]
        xs = list(range(1, len(ckpts) + 1))
        for g in GROUPS:
            ys = [group_rate(ev["results"], c, g)[0] / max(1, group_rate(ev["results"], c, g)[1]) for c in ckpts]
            ax.plot(xs, ys, color=COLORS[g], label=NICE[g], linewidth=1.7, markersize=4.5, **STYLE[g])
        ax.axvline(1, color="#888888", linestyle=":", linewidth=1.1)
        ax.set_xlabel("Epoch"); ax.set_xticks(xs); ax.set_ylim(-0.06, 1.06)
        ax.set_title(label, fontsize=10.5, fontweight="bold")
        ax.grid(alpha=0.25, linewidth=0.6)
    axes[0].set_ylabel("Marker emission rate")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7.4, frameon=False)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "fig4_fragility.png")
    fig.savefig(out, dpi=220, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    ea, eb = load("evals/eval_results_meta_organism_a.json"), load("evals/eval_results_meta_organism_b.json")
    sa = load("evals/eval_results_meta_organism_a_stratified.json")
    sb = load("evals/eval_results_meta_organism_b_stratified.json")
    fig1_method()
    fig2_modality(sa, sb)
    fig3_confusion(ea, eb)
    fig4_fragility(ea, eb)

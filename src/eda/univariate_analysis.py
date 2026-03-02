"""
Univariate Analysis — DPWH Flood Control Twitter Dataset
Professional Analytical Version

Enhancements:
• Clear section labeling
• Consistent categorical ordering
• Median line color aligned to histogram
• Improved annotation positioning
• Cleaner statistics panel
• Value labels on bar charts
• Executive-readable formatting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ── Design Tokens ─────────────────────────────────────────────
BG        = "#FAFBFC"
CARD_BG   = "#FFFFFF"
TXT       = "#0F172A"
TXT_MED   = "#475569"
RULE      = "#E2E8F0"

C_BLUE    = "#2563EB"
C_PURPLE  = "#7C3AED"
C_TEAL    = "#0D9488"
C_ORANGE  = "#D97706"
C_RED     = "#DC2626"
C_GREEN   = "#059669"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.axisbelow": True,
})

# ──────────────────────────────────────────────────────────────

def _save(fig, out_dir, filename):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=170, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ Chart saved → {p}")

def _style_ax(ax, xlabel="", grid_axis="y"):
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(colors=TXT_MED, labelsize=8, length=0)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=TXT_MED, labelpad=6)

    if grid_axis == "y":
        ax.yaxis.grid(True, color=RULE, linewidth=0.6)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=RULE, linewidth=0.6)
        ax.yaxis.grid(False)

def _fmt_k(x, _):
    if x == 0: return "0"
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000: return f"{x/1_000:.0f}k"
    return str(int(x))

# ──────────────────────────────────────────────────────────────
# LOG HISTOGRAM
# ──────────────────────────────────────────────────────────────

def _log_hist(ax, data, color, label):

    data = data.dropna()
    nonzero  = data[data > 0]
    zeros    = (data == 0).sum()
    N        = len(data)

    if len(nonzero) == 0:
        ax.text(0.5, 0.5, "No non-zero values",
                transform=ax.transAxes,
                ha="center", va="center")
        return

    log_min = np.floor(np.log10(nonzero.min()))
    log_max = np.ceil(np.log10(nonzero.max()))
    bins = np.logspace(log_min, log_max, 25)

    counts, edges = np.histogram(nonzero, bins=bins)

    ax.set_xscale("log")
    ax.bar(edges[:-1], counts,
           width=np.diff(edges)*0.9,
           align="edge",
           color=color,
           alpha=0.85)

    # Median
    med = nonzero.median()
    ax.axvline(med, linestyle="--", linewidth=1.4, color=color, alpha=0.7)

    ax.set_title(label, fontsize=10, color=TXT)

    # Stats Panel
    stats_text = "\n".join([
        f"Median   {_fmt_k(data.median(), None)}",
        f"P90      {_fmt_k(data.quantile(0.9), None)}",
        f"Max      {_fmt_k(data.max(), None)}",
        f"Skew     {data.skew():.2f}",
    ])

    ax.text(
        0.98, 0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=8,
        color=TXT_MED,
        va="top",
        ha="right",
        family="monospace",
        bbox=dict(facecolor=CARD_BG,
                  edgecolor=RULE,
                  boxstyle="round,pad=0.5",
                  linewidth=0.8)
    )

    if zeros > 0:
        pct_zero = zeros / N * 100
        ax.text(
            0.02, 0.95,
            f"{pct_zero:.0f}% zero values",
            transform=ax.transAxes,
            fontsize=8,
            color=TXT_MED,
            va="top"
        )

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    _style_ax(ax)

# ─────────────────────────────────────────
# HELPER FOR BAR LABELS
# ─────────────────────────────────────────

def _add_bar_labels(ax):
    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2,
                height,
                f"{int(height):,}",
                ha='center',
                va='bottom',
                fontsize=8,
                color=TXT_MED
            )

# ─────────────────────────────────────────
# TWEETS DATASET
# ─────────────────────────────────────────

def get_univariate_for_tweets(filepath, output_dir="output"):

    base_filename = Path(filepath).stem
    df = pd.read_csv(filepath)

    df["isReply"] = df["isReply"].astype(bool)
    df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)

    fig = plt.figure(figsize=(16, 20), facecolor=BG)

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        left=0.06,
        right=0.97,
        top=0.95,
        bottom=0.05,
        hspace=0.60,
        wspace=0.40
    )

    fig.suptitle(
        "Univariate Analysis — Tweet Engagement Distribution",
        fontsize=14,
        color=TXT,
        y=0.98
    )

    metrics = [
        ("retweetCount", C_BLUE,   "Retweets"),
        ("likeCount",    C_TEAL,   "Likes"),
        ("viewCount",    C_PURPLE, "Views"),
        ("quoteCount",   C_ORANGE, "Quotes"),
        ("replyCount",   C_RED,    "Replies"),
        ("bookmarkCount",C_GREEN,  "Bookmarks"),
    ]

    for i, (col, color, label) in enumerate(metrics):
        row = 1 if i < 3 else 2
        colpos = i % 3
        ax = fig.add_subplot(gs[row, colpos])
        _log_hist(ax, df[col], color, label)

    # Language
    ax_lang = fig.add_subplot(gs[3, 0])
    lvc = df["lang"].value_counts()
    ax_lang.barh(lvc.index[::-1], lvc.values[::-1])
    ax_lang.set_title("Tweet Language", fontsize=10)
    _style_ax(ax_lang, xlabel="Tweet Count", grid_axis="x")

    # Reply Status
    ax_rep = fig.add_subplot(gs[3, 1])
    rvc = df["isReply"].value_counts().reindex([False, True])
    ax_rep.bar(["Original","Reply"], rvc.values)
    ax_rep.set_title("Reply Status", fontsize=10)
    _style_ax(ax_rep)
    _add_bar_labels(ax_rep)

    # Verified
    ax_bv = fig.add_subplot(gs[3, 2])
    bvc = df["author_isBlueVerified"].value_counts().reindex([False, True])
    ax_bv.bar(["Not Verified","Verified"], bvc.values)
    ax_bv.set_title("Author Verification", fontsize=10)
    _style_ax(ax_bv)
    _add_bar_labels(ax_bv)

    _save(fig, output_dir, f"{base_filename}_univariate_analysis.png")
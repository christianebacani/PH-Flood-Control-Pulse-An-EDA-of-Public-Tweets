"""
univariate_analysis.py
DPWH Flood Control Twitter Dataset

Professional Portfolio-Ready Version

Enhancements:
• X-axis and Y-axis tick lines fixed for readability
• Median line and label readable
• Stats panel neat and aligned
• "Others" category for crowded bars
• Horizontal bars for categorical distributions
• Clean, executive-ready, professional visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, LogLocator
from pathlib import Path

# ─────────────────────────────────────────
# DESIGN TOKENS
# ─────────────────────────────────────────
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

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.axisbelow": True})

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def _save(fig, out_dir, filename):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=BG, transparent=False)
    plt.close(fig)
    print(f"✓ Chart saved → {path}")

def _style_ax(ax, xlabel="", grid_axis="y"):
    ax.set_facecolor(CARD_BG)

    # Hide all spines except bottom and left for subtle reference
    for loc, spine in ax.spines.items():
        if loc in ['bottom','left']:
            spine.set_visible(True)
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
        else:
            spine.set_visible(False)

    # Tick parameters: subtle ticks for readability
    ax.tick_params(colors=TXT_MED, labelsize=9, length=4, width=0.8)
    ax.tick_params(bottom=True, left=True)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=TXT_MED, labelpad=6)

    # Light grid for major axis
    if grid_axis == "y":
        ax.yaxis.grid(True, color=RULE, linewidth=0.6)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, color=RULE, linewidth=0.6)
        ax.yaxis.grid(False)

def _fmt_k(x, _):
    if x >= 1_000_000: return f"{x/1_000_000:.1f}M"
    if x >= 1_000: return f"{x/1_000:.0f}k"
    return str(int(x)) if x > 0 else "0"

def _add_bar_labels(ax, total=None, horizontal=False):
    for container in ax.containers:
        for bar in container:
            value = bar.get_width() if horizontal else bar.get_height()
            label = f"{int(value):,}"
            if total:
                pct = value / total * 100
                label += f" ({pct:.1f}%)"
            if horizontal:
                ax.text(value + max(total*0.01, 1), bar.get_y() + bar.get_height()/2,
                        label, ha="left", va="center", fontsize=9, color=TXT_MED)
            else:
                ax.text(bar.get_x() + bar.get_width()/2, value,
                        label, ha="center", va="bottom", fontsize=9, color=TXT_MED)

# ─────────────────────────────────────────
# LOG HISTOGRAM WITH MEDIAN AND STATS PANEL
# ─────────────────────────────────────────
def _plot_histogram(ax, data, color, title):
    data = data.dropna()
    N = len(data)
    zeros = (data == 0).sum()
    nonzero = data[data > 0]

    ax.set_title(title, fontsize=12, fontweight="bold", color=TXT)

    # Zero bar
    if zeros > 0:
        ax.bar(0.5, zeros, width=0.5, color="#CBD5E1", alpha=0.9, align="center")
        ax.text(0.5, zeros, f"Zero\n{zeros:,} ({zeros/N*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)

    # Log or linear histogram
    if len(nonzero) > 0:
        use_log = nonzero.max() / nonzero.min() > 50
        bins = np.logspace(np.floor(np.log10(nonzero.min())), np.ceil(np.log10(nonzero.max())), 25) if use_log else 25

        ax.set_xscale("log" if use_log else "linear")
        if use_log:
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1,2,5], numticks=6))

        counts, edges = np.histogram(nonzero, bins=bins)
        ax.bar(edges[:-1], counts, width=np.diff(edges)*0.9, align="edge", color=color, alpha=0.85)

        # Median line and label
        med = data.median()
        if med > 0:
            ax.axvline(med, linestyle="--", linewidth=1.6, color=color)
            med_y = ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]*0.95 > counts.max()*1.05 else counts.max()*1.05
            ax.text(med, med_y, f"Median: {_fmt_k(med,None)}",
                    rotation=90, fontsize=9, color=color, va="top", ha="right", backgroundcolor=BG)

    # Stats panel
    q1, q3 = data.quantile([0.25,0.75])
    stats_text = f"""Median   {_fmt_k(data.median(),None):>8}
Mean     {_fmt_k(data.mean(),None):>8}
IQR      {_fmt_k(q1,None):>4}–{_fmt_k(q3,None):<4}
Max      {_fmt_k(data.max(),None):>8}"""
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            color=TXT_MED, va="top", ha="right", family="monospace",
            bbox=dict(facecolor="#F9FAFB", edgecolor=RULE, boxstyle="round,pad=0.5"))

    # Axis formatting
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_: f"{int(x):,}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    _style_ax(ax)

# ─────────────────────────────────────────
# UNIVARIATE ANALYSIS — TWEETS
# ─────────────────────────────────────────
def get_univariate_for_tweets(filepath, output_dir="output"):
    df = pd.read_csv(filepath)
    df["isReply"] = df["isReply"].astype(bool)
    df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)
    base_filename = Path(filepath).stem

    # Engagement metrics
    fig = plt.figure(figsize=(16,18), facecolor=BG)
    gs = gridspec.GridSpec(3,3, figure=fig, left=0.05, right=0.98, top=0.92, bottom=0.05, hspace=0.65, wspace=0.35)
    fig.suptitle("Univariate Analysis — Tweet Engagement", fontsize=16, fontweight="bold", color=TXT)

    metrics = [("retweetCount",C_BLUE,"Retweets"),
               ("likeCount",C_TEAL,"Likes"),
               ("viewCount",C_PURPLE,"Views"),
               ("quoteCount",C_ORANGE,"Quotes"),
               ("replyCount",C_RED,"Replies"),
               ("bookmarkCount",C_GREEN,"Bookmarks")]

    for i,(col,color,label) in enumerate(metrics):
        ax = fig.add_subplot(gs[i//3, i%3])
        _plot_histogram(ax, df[col], color, label)

    # Categorical distributions
    fig2 = plt.figure(figsize=(14,6), facecolor=BG)
    gs2 = gridspec.GridSpec(1,3, figure=fig2, left=0.06, right=0.97, top=0.85, bottom=0.15, wspace=0.40)
    total = len(df)

    # Reply Status
    ax_rep = fig2.add_subplot(gs2[0,0])
    rvc = df["isReply"].value_counts().reindex([False,True], fill_value=0)
    ax_rep.bar(["Original","Reply"], rvc.values)
    ax_rep.set_title("Reply Status", fontsize=12,fontweight="bold")
    _style_ax(ax_rep)
    _add_bar_labels(ax_rep,total)

    # Author Verification
    ax_bv = fig2.add_subplot(gs2[0,1])
    bvc = df["author_isBlueVerified"].value_counts().reindex([False,True], fill_value=0)
    ax_bv.bar(["Not Verified","Verified"], bvc.values)
    ax_bv.set_title("Author Verification", fontsize=12,fontweight="bold")
    _style_ax(ax_bv)
    _add_bar_labels(ax_bv,total)

    # Top Languages
    ax_lang = fig2.add_subplot(gs2[0,2])
    lang_counts = df["lang"].value_counts()
    top5 = lang_counts.head(5)
    others = lang_counts[5:].sum()
    if others > 0:
        top5["Others"] = others
    ax_lang.barh(top5.index[::-1], top5.values[::-1], height=0.6)
    ax_lang.set_title("Top Languages", fontsize=12,fontweight="bold")
    _style_ax(ax_lang, xlabel="Number of Tweets", grid_axis="x")
    _add_bar_labels(ax_lang, total, horizontal=True)

    _save(fig, output_dir, f"{base_filename}_engagement_distribution.png")
    _save(fig2, output_dir, f"{base_filename}_categorical_distribution.png")

# ─────────────────────────────────────────
# UNIVARIATE ANALYSIS — AUTHORS
# ─────────────────────────────────────────
def get_univariate_for_authors(filepath, output_dir="output"):
    df = pd.read_csv(filepath)
    df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)
    base_filename = Path(filepath).stem
    total = len(df)

    fig = plt.figure(figsize=(16,14), facecolor=BG)
    gs = gridspec.GridSpec(2,3, figure=fig, left=0.06, right=0.97, top=0.93, bottom=0.05, hspace=0.55, wspace=0.35)
    fig.suptitle("Univariate Analysis — Author Profiles", fontsize=16,fontweight="bold",color=TXT)

    # Followers
    ax1 = fig.add_subplot(gs[0,0])
    _plot_histogram(ax1, df["author_followers"], C_TEAL, "Follower Count")

    # Following
    ax2 = fig.add_subplot(gs[0,1])
    _plot_histogram(ax2, df["author_following"], C_PURPLE, "Following Count")

    # Verification
    ax3 = fig.add_subplot(gs[0,2])
    bvc = df["author_isBlueVerified"].value_counts().reindex([False,True], fill_value=0)
    ax3.bar(["Not Verified","Verified"], bvc.values)
    ax3.set_title("Author Verification", fontsize=12,fontweight="bold")
    _style_ax(ax3)
    _add_bar_labels(ax3,total)

    # Top Locations
    ax_loc = fig.add_subplot(gs[1,:])
    loc_counts = df["author_location"].fillna("Unknown").value_counts()
    top10 = loc_counts.head(10)
    others = loc_counts[10:].sum()
    if others > 0:
        top10["Others"] = others
    ax_loc.barh(top10.index[::-1], top10.values[::-1], height=0.6)
    ax_loc.set_title("Top Author Locations", fontsize=12,fontweight="bold")
    _style_ax(ax_loc, xlabel="Number of Authors", grid_axis="x")
    _add_bar_labels(ax_loc,total,horizontal=True)

    _save(fig, output_dir, f"{base_filename}_author_distribution.png")
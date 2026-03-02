import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
from pathlib import Path

# ── Palette ────────────────────────────────────────────────────────────────────
BG        = "#FFFFFF"
BG_ALT    = "#F8FAFC"
BG_DARK   = "#1E293B"
TXT       = "#1E293B"
TXT_MED   = "#64748B"
TXT_LIGHT = "#94A3B8"
RULE      = "#E2E8F0"
RULE_MED  = "#CBD5E1"

C_BLUE    = "#3B82F6";  BG_BLUE   = "#EFF6FF"
C_PURPLE  = "#7C3AED";  BG_PURPLE = "#FAF5FF"
C_TEAL    = "#0D9488";  BG_TEAL   = "#F0FDFA"
C_ORANGE  = "#F59E0B"
C_RED     = "#EF4444"
C_GREEN   = "#10B981"
C_SLATE   = "#475569"

plt.rcParams.update({"font.family": "DejaVu Sans"})


def _save(fig, out_dir, filename):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ Chart Saved → {p}")


def _style_ax(ax, xlabel="", grid_axis="y"):
    ax.set_facecolor(BG_ALT)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(colors=TXT_MED, labelsize=7)
    ax.set_xlabel(xlabel, fontsize=7.5, color=TXT_MED, labelpad=4)
    if grid_axis:
        ax.grid(axis=grid_axis, color=RULE_MED, linewidth=0.5, zorder=0)


def _section_label(ax, label, color):
    ax.text(0.0, 1.055, label, transform=ax.transAxes,
            fontsize=8, fontweight="bold", color=color, va="bottom")


def _stat_box(ax, stats):
    text = "\n".join(f"{v}  {k}" for k, v in stats)
    ax.text(0.97, 0.97, text,
            transform=ax.transAxes, fontsize=6.5, color=TXT_MED,
            va="top", ha="right", linespacing=1.6,
            bbox=dict(facecolor=BG, edgecolor=RULE,
                      boxstyle="round,pad=0.3", linewidth=0.7))


def _draw_header(fig, gs_row, title, subtitle, pills):
    ax = fig.add_subplot(gs_row)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.88, title, fontsize=14, fontweight="bold",
            ha="center", va="top", color=TXT, transform=ax.transAxes)
    ax.text(0.5, 0.50, subtitle, fontsize=8, color=TXT_MED,
            ha="center", va="top", transform=ax.transAxes)
    pw = 0.20
    px = 0.5 - pw * (len(pills) - 1) / 2
    for j, (lbl, fg, bg, bd) in enumerate(pills):
        ax.text(px + j * pw, 0.06, lbl,
                fontsize=8, fontweight="bold", ha="center", va="bottom",
                color=fg, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=bg,
                          edgecolor=bd, linewidth=1.0))
    ax.plot([0.01, 0.99], [-0.06, -0.06], color=RULE_MED, linewidth=0.8,
            transform=ax.transAxes, clip_on=False)


def _fmt_k(x, _):
    """Format large numbers as 0, 1k, 10k, 100k, 1M etc."""
    if x == 0: return "0"
    if x >= 1_000_000: return f"{x/1_000_000:.0f}M"
    if x >= 1_000: return f"{int(x/1000)}k"
    return str(int(x))


def _log_hist(ax, data, color, col_label, zero_label=True):
    """
    Histogram on log-scale x-axis for zero-heavy count data.
    Non-zero values are binned on log scale; zeros shown as separate leftmost bar.
    """
    nonzero = data[data > 0]
    n_zeros = (data == 0).sum()
    N_total = len(data)

    if len(nonzero) == 0:
        ax.bar([0], [N_total], color=color, alpha=0.85, zorder=3, linewidth=0)
        _style_ax(ax, xlabel=col_label)
        return

    # Log-spaced bins for non-zero values
    log_min = np.floor(np.log10(nonzero.min())) if nonzero.min() > 0 else 0
    log_max = np.ceil(np.log10(nonzero.max()))
    bins = np.logspace(log_min, log_max, 30)

    counts, edges = np.histogram(nonzero, bins=bins)

    ax.set_xscale("log")
    ax.bar(edges[:-1], counts, width=np.diff(edges),
           align="edge", color=color, alpha=0.85, zorder=3, linewidth=0)

    # Median line
    med = data.median()
    if med > 0:
        ax.axvline(med, color=BG_DARK, linewidth=1.3,
                   linestyle="--", zorder=4, alpha=0.8)
        ymax = counts.max()
        ax.text(med * 1.2, ymax * 0.92,
                f"med {_fmt_k(med, None)}",
                fontsize=6.5, color=BG_DARK, va="top")

    # Zero annotation in top-left
    if n_zeros > 0 and zero_label:
        ax.text(0.03, 0.97,
                f"{n_zeros/N_total*100:.0f}% zeros",
                transform=ax.transAxes, fontsize=6.5, color=TXT_MED,
                va="top", ha="left",
                bbox=dict(facecolor=BG, edgecolor=RULE,
                          boxstyle="round,pad=0.25", linewidth=0.6))

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    _style_ax(ax, xlabel=col_label)

    # Stat box
    _stat_box(ax, [
        ("mean",  _fmt_k(data.mean(), None)),
        ("med",   _fmt_k(data.median(), None)),
        ("max",   _fmt_k(data.max(), None)),
    ])


# ═════════════════════════════════════════════════════════════════════════════
#  TWEETS
# ═════════════════════════════════════════════════════════════════════════════

def get_univariate_for_tweets(filepath: str, output_dir: str = "output"):
    df = pd.read_csv(filepath)
    N  = len(df)

    df["isReply"]               = df["isReply"].astype(bool)
    df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)

    FIG_W, FIG_H = 15.0, 19.5
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    gs = gridspec.GridSpec(
        4, 3, figure=fig,
        left=0.06, right=0.96, top=0.97, bottom=0.03,
        hspace=0.65, wspace=0.42,
        height_ratios=[0.10, 1, 1, 1]
    )

    _draw_header(fig, gs[0, :],
        title    = "Univariate Analysis — Dataset 1: Tweets",
        subtitle = "Distribution of each variable independently  ·  Engagement histograms use log scale (skewed count data)",
        pills    = [
            (f"  {N:,} tweets  ", C_SLATE, "#F1F5F9", RULE_MED),
            (f"  6 numeric  ", C_BLUE, BG_BLUE, "#BFDBFE"),
            (f"  3 categorical  ", C_PURPLE, BG_PURPLE, "#DDD6FE"),
        ]
    )

    # ── Row 1: retweetCount, likeCount, viewCount ────────────────────────────
    for ci, (col, label, color) in enumerate([
        ("retweetCount", "Retweet Count", C_BLUE),
        ("likeCount",    "Like Count",    C_TEAL),
        ("viewCount",    "View Count",    C_PURPLE),
    ]):
        ax = fig.add_subplot(gs[1, ci])
        _log_hist(ax, df[col].dropna(), color, label)
        _section_label(ax, label, color)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Row 2: quoteCount, replyCount, bookmarkCount ─────────────────────────
    for ci, (col, label, color) in enumerate([
        ("quoteCount",    "Quote Count",    C_ORANGE),
        ("replyCount",    "Reply Count",    C_RED),
        ("bookmarkCount", "Bookmark Count", C_GREEN),
    ]):
        ax = fig.add_subplot(gs[2, ci])
        _log_hist(ax, df[col].dropna(), color, label)
        _section_label(ax, label, color)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Row 3: categorical ────────────────────────────────────────────────────
    # lang
    ax_lang = fig.add_subplot(gs[3, 0])
    lvc = df["lang"].value_counts()
    lbl_map = {"tl": "Filipino (tl)", "en": "English (en)", "und": "Undetermined (und)"}
    lbls    = [lbl_map.get(k, k) for k in lvc.index]
    lcolors = [C_BLUE, C_TEAL, TXT_LIGHT][:len(lvc)]
    bars = ax_lang.barh(lbls[::-1], lvc.values[::-1],
                        color=lcolors[:len(lvc)][::-1],
                        alpha=0.9, zorder=3, linewidth=0, height=0.50)
    for bar, val in zip(bars, lvc.values[::-1]):
        ax_lang.text(bar.get_width() + N * 0.008,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:,}  ({val/N*100:.1f}%)",
                     va="center", ha="left", fontsize=7.5, color=TXT_MED)
    _style_ax(ax_lang, xlabel="Tweet Count", grid_axis="x")
    _section_label(ax_lang, "Language (lang)", C_BLUE)
    ax_lang.set_xlim(0, lvc.max() * 1.45)
    ax_lang.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    # isReply
    ax_rep = fig.add_subplot(gs[3, 1])
    rvc  = df["isReply"].value_counts().sort_index(ascending=False)  # False first = Original
    rlbl = ["Original" if not k else "Reply" for k in rvc.index]
    rcol = [C_TEAL if not k else C_ORANGE for k in rvc.index]
    bars = ax_rep.bar(rlbl, rvc.values, color=rcol,
                      alpha=0.9, zorder=3, linewidth=0, width=0.45)
    for bar, val in zip(bars, rvc.values):
        ax_rep.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:,}\n({val/N*100:.1f}%)",
                    ha="center", va="bottom", fontsize=8, color=TXT_MED)
    _style_ax(ax_rep)
    _section_label(ax_rep, "Reply Status (isReply)", C_ORANGE)
    ax_rep.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax_rep.set_ylim(0, rvc.max() * 1.28)

    # author_isBlueVerified
    ax_bv = fig.add_subplot(gs[3, 2])
    bvc  = df["author_isBlueVerified"].value_counts().sort_index(ascending=False)
    blbl = ["Not Verified" if not k else "Verified ✓" for k in bvc.index]
    bcol = [C_SLATE if not k else C_BLUE for k in bvc.index]
    bars = ax_bv.bar(blbl, bvc.values, color=bcol,
                     alpha=0.9, zorder=3, linewidth=0, width=0.45)
    for bar, val in zip(bars, bvc.values):
        ax_bv.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() * 1.02,
                   f"{val:,}\n({val/N*100:.1f}%)",
                   ha="center", va="bottom", fontsize=8, color=TXT_MED)
    _style_ax(ax_bv)
    _section_label(ax_bv, "Blue Verification (author_isBlueVerified)", C_BLUE)
    ax_bv.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
    ax_bv.set_ylim(0, bvc.max() * 1.28)

    _save(fig, output_dir,
          "for_export_dpwh_floodcontrol_univariate_analysis.png")


# ═════════════════════════════════════════════════════════════════════════════
#  AUTHORS
# ═════════════════════════════════════════════════════════════════════════════

def get_univariate_for_authors(filepath: str, output_dir: str = "output"):
    df = pd.read_csv(filepath)
    N  = len(df)

    df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)

    # Clean location
    invalid = ["worldwide", "http://link.com", "Earth", "Around The World",
               "facebook.com/aidelacruzonline", "Abbott Elementary",
               "WhatsApp & Telegram"]
    df["author_location_clean"] = df["author_location"].replace(invalid, "Unknown")
    df["author_location_clean"] = df["author_location_clean"].fillna("Unknown")

    FIG_W, FIG_H = 15.0, 13.5
    fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=BG)

    gs = gridspec.GridSpec(
        3, 3, figure=fig,
        left=0.07, right=0.96, top=0.97, bottom=0.04,
        hspace=0.65, wspace=0.44,
        height_ratios=[0.12, 1, 1.1]
    )

    _draw_header(fig, gs[0, :],
        title    = "Univariate Analysis — Dataset 2: Authors",
        subtitle = "Distribution of each variable independently  ·  Follower/following histograms use log scale",
        pills    = [
            (f"  {N:,} authors  ", C_SLATE, "#F1F5F9", RULE_MED),
            (f"  2 numeric  ", C_TEAL, BG_TEAL, "#99F6E4"),
            (f"  2 categorical  ", C_PURPLE, BG_PURPLE, "#DDD6FE"),
        ]
    )

    # ── Row 1: followers, following, verified ────────────────────────────────
    for ci, (col, label, color) in enumerate([
        ("author_followers", "Follower Count", C_TEAL),
        ("author_following", "Following Count", C_PURPLE),
    ]):
        ax = fig.add_subplot(gs[1, ci])
        _log_hist(ax, df[col].dropna(), color, label, zero_label=False)
        _section_label(ax, label, color)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Blue verified
    ax_bv = fig.add_subplot(gs[1, 2])
    bvc  = df["author_isBlueVerified"].value_counts().sort_index(ascending=False)
    blbl = ["Not Verified" if not k else "Verified ✓" for k in bvc.index]
    bcol = [C_SLATE if not k else C_BLUE for k in bvc.index]
    bars = ax_bv.bar(blbl, bvc.values, color=bcol,
                     alpha=0.9, zorder=3, linewidth=0, width=0.45)
    for bar, val in zip(bars, bvc.values):
        ax_bv.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() * 1.02,
                   f"{val}\n({val/N*100:.0f}%)",
                   ha="center", va="bottom", fontsize=9, color=TXT_MED)
    _style_ax(ax_bv)
    _section_label(ax_bv, "Blue Verification (author_isBlueVerified)", C_BLUE)
    ax_bv.set_ylim(0, bvc.max() * 1.35)

    # ── Row 2: Location full-width ───────────────────────────────────────────
    ax_loc = fig.add_subplot(gs[2, :])
    loc_vc = df["author_location_clean"].value_counts().head(10)
    lcolors = [TXT_LIGHT if v == "Unknown" else C_TEAL for v in loc_vc.index]
    bars = ax_loc.barh(loc_vc.index[::-1], loc_vc.values[::-1],
                       color=lcolors[::-1],
                       alpha=0.9, zorder=3, linewidth=0, height=0.6)
    for bar, val in zip(bars, loc_vc.values[::-1]):
        ax_loc.text(bar.get_width() + N * 0.003,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val}  ({val/N*100:.0f}%)",
                    va="center", ha="left", fontsize=8.5, color=TXT_MED)
    _style_ax(ax_loc, xlabel="Number of Authors", grid_axis="x")
    _section_label(ax_loc, "Author Location — Top 10 (after cleaning)", C_TEAL)
    ax_loc.set_xlim(0, loc_vc.max() * 1.28)
    ax_loc.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x))))

    _save(fig, output_dir,
          "well_known_authors_dpwh_floodcontrol_univariate_analysis.png")
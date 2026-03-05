"""
univariate_analysis.py  [FIXED v3]
===================================
New fixes in this version (on top of v2)
-----------------------------------------
A.  get_univariate_for_tweet_categoricals
    A1. Bar-label clipping — y-axis ceiling raised to max * 1.20 (was 1.15)
        for Reply Status and Author Verification so labels like "123,103 (62.9%)"
        are never cropped at the top of the axes.
    A2. Undetermined bar label — previously floated in dead space because the
        bar height (~410) << N * 0.003 offset. Offset now scales from
        lang_counts.max() * 0.01 instead of N * 0.003, so even tiny bars
        get a label placed just outside their right edge.
    A3. figure height raised 5 → 6.5 and subplots_adjust bottom added so
        the x-axis "Number of Tweets" label is never clipped on the language panel.

B.  _plot_histogram (engagement histograms)
    B1. Median line zorder raised to 4 and color darkened (was same hue at
        alpha 0.55) — now drawn in a near-black (#1E293B) so it is always
        visible against both light and saturated bars.
    B2. Median line only drawn when med > xlim_left * 1.05 so it never
        renders flush against or behind the y-axis spine (Retweets / Likes /
        Quotes median = 1–3, right at the edge).
    B3. Stats box width: switched to a fixed-width monospace column alignment
        using ljust/rjust so values never overflow the rounded box on any panel.

C.  get_univariate_for_authors
    C1. _fmt_iqr fix for sub-1K q1 with K-range q3 — e.g. Following Count
        IQR q1=200, q3=1100 produced "0.2K–1.1K"; now returns "200–1.1K"
        (q1 shown as integer when < 1K even if q3 is in K range).
    C2. Top Author Locations — "Others" label detection hardened: the check
        now compares against the raw (pre-truncation) "Others" string so it
        still matches after label normalisation.
    C3. Location chart left margin widened (left=0.10 in GridSpec) so long
        y-tick labels like "National Capital Region, …" are not clipped.

D.  get_temporal_distribution
    D1. Spike annotation arrow removed (arrowstyle "-" produced a hairline
        white vertical rule that cut through the area fill). Replaced with
        a small filled circle marker at the peak and the date label above it,
        using xytext=(0,14) with no arrowprops. Cleaner and no artefacts.
    D2. Legend moved to upper right when data is sparse on the right side
        (checked by comparing last-quarter mean vs first-quarter mean); falls
        back to upper left. Hard-coded "upper left" was overlapping data.
    D3. Y-axis gridlines: linewidth raised 0.6 → 1.0, color darkened to
        "#CBD5E1" so they are clearly visible (was nearly invisible on screen).
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import (
    FuncFormatter,
    LogLocator,
    MaxNLocator,
    NullLocator,
)

# ──────────────────────────────────────────────────────────────────
# 🎨  Design tokens
# ──────────────────────────────────────────────────────────────────

BG      = "#F8FAFC"
CARD_BG = "#FFFFFF"
TXT     = "#0F172A"
TXT_MED = "#475569"
TXT_LT  = "#94A3B8"
RULE    = "#E2E8F0"
GRID    = "#CBD5E1"          # D3: slightly darker grid lines
MEDIAN_LINE = "#1E293B"      # B1: near-black for median line visibility

PALETTE = {
    "retweetCount":     "#3B82F6",
    "likeCount":        "#14B8A6",
    "viewCount":        "#8B5CF6",
    "quoteCount":       "#F59E0B",
    "replyCount":       "#EF4444",
    "bookmarkCount":    "#10B981",
    "author_followers": "#14B8A6",
    "author_following": "#8B5CF6",
}

LANG_LABELS = {
    "tl":  "Filipino (tl)",
    "en":  "English (en)",
    "und": "Undetermined",
}

LANG_COLORS = {
    "Filipino (tl)": "#3B82F6",
    "English (en)":  "#14B8A6",
    "Other":         "#CBD5E1",
}

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    RULE,
    "axes.labelcolor":   TXT_MED,
    "xtick.color":       TXT_MED,
    "ytick.color":       TXT_MED,
    "font.family":       "sans-serif",
    "font.size":         10,
    "axes.titlesize":    12,
    "axes.titleweight":  "bold",
    "axes.titlecolor":   TXT,
})

# ──────────────────────────────────────────────────────────────────
# 🔧  Private helpers
# ──────────────────────────────────────────────────────────────────

def _fmt_k(x, _=None):
    if x >= 1_000_000:
        v = x / 1_000_000
        return f"{v:.1f}M" if v < 10 and v != int(v) else f"{int(v)}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{int(x)}"


def _fmt_stat(x):
    """Scale-aware stat formatter (consistent K/M)."""
    if x >= 1_000_000:
        v = x / 1_000_000
        return f"{v:.2f}M" if v < 10 else f"{int(v)}M"
    if x >= 1_000:
        v = x / 1_000
        return f"{v:.1f}K" if v < 100 else f"{int(v)}K"
    if x != int(x):
        return f"{x:.1f}"
    return f"{int(x)}"


def _fmt_iqr(q1, q3):
    """
    FIX C1: q1 shown as plain integer when < 1 000, even if q3 is in K range.
    Eliminates the confusing '0.2K–1.1K' output for Following Count.
    """
    if q3 >= 1_000_000:
        return f"{_fmt_stat(q1)}–{_fmt_stat(q3)}"
    elif q3 >= 1_000:
        # q3 in K range — format q3 as K; q1 as integer if < 1K
        v3 = q3 / 1_000
        fmt3 = f"{v3:.1f}K" if v3 != int(v3) else f"{int(v3)}K"
        if q1 < 1_000:
            fmt1 = f"{int(q1)}" if q1 == int(q1) else f"{q1:.0f}"
        else:
            v1 = q1 / 1_000
            fmt1 = f"{v1:.1f}K" if v1 != int(v1) else f"{int(v1)}K"
        return f"{fmt1}–{fmt3}"
    else:
        fmt1 = f"{int(q1)}" if q1 == int(q1) else f"{q1:.1f}"
        fmt3 = f"{int(q3)}" if q3 == int(q3) else f"{q3:.1f}"
        return f"{fmt1}–{fmt3}"


def _truncate_label(label, max_len=32):
    """
    Truncate at a word boundary so we never cut mid-word.
    e.g. 'National Capital Region, Republic of the Philippines'
    → 'National Capital Region, …'  (breaks before 'Republic')
    """
    if len(label) <= max_len:
        return label
    # Walk backwards from max_len to find a space or comma
    cut = label[:max_len].rfind(" ")
    if cut <= 0:
        cut = label[:max_len].rfind(",")
    if cut <= 0:
        cut = max_len - 1
    return label[:cut].rstrip(", ") + " …"


def _safe_show(fig, save_path=None):
    if save_path:
        save_path = os.path.normpath(save_path)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Chart saved → {save_path}")
    is_gui = matplotlib.get_backend().lower() not in (
        "agg", "cairo", "pdf", "ps", "svg", "template"
    )
    if is_gui:
        plt.show()
    else:
        if not save_path:
            fig.savefig("univariate_output.png", dpi=300, bbox_inches="tight")
            print("[viz] Non-interactive — saved → univariate_output.png")
        plt.close(fig)


def _style_ax(ax, grid_axis="y", xlabel=""):
    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom", "left"))
        if loc in ("bottom", "left"):
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    ax.tick_params(which="major", length=5, width=0.8, labelsize=9)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    if grid_axis == "y":
        # D3: darker, slightly thicker grid lines
        ax.yaxis.grid(True, which="major", color=GRID, linewidth=1.0, zorder=0)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, which="major", color=GRID, linewidth=1.0, zorder=0)
        ax.yaxis.grid(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=5, color=TXT_MED)


def _normalize_dtypes(df):
    for col in ("isReply", "author_isBlueVerified"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes")
                if pd.notna(v) else False
            ).astype(bool)
    for col in ("createdAt", "author_createdAt"):
        if col in df.columns:
            df[col] = (
                pd.to_datetime(df[col], errors="coerce", utc=True)
                .dt.tz_localize(None)
            )
    if "pseudo_inReplyToUsername" in df.columns:
        df["pseudo_inReplyToUsername"] = (
            df["pseudo_inReplyToUsername"]
            .astype(str)
            .replace({"nan": np.nan, "<NA>": np.nan})
        )
    return df


def _normalize_tweet_columns(df):
    rename = {
        "retweet_count":  "retweetCount",
        "reply_count":    "replyCount",
        "like_count":     "likeCount",
        "quote_count":    "quoteCount",
        "view_count":     "viewCount",
        "bookmark_count": "bookmarkCount",
    }
    return df.rename(columns={k: v for k, v in rename.items() if k in df.columns})


def _validate_columns(df, required):
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _plot_histogram(ax, data, color, title):
    """
    Fixes B1–B3 applied here.
    """
    data    = pd.to_numeric(data, errors="coerce").dropna()
    N       = len(data)
    nonzero = data[data > 0]
    n_zeros = int((data == 0).sum())

    ax.set_title(title)

    if N == 0 or len(nonzero) == 0:
        ax.text(0.5, 0.5, "All zeros" if N > 0 else "No data",
                ha="center", va="center", transform=ax.transAxes,
                color=TXT_MED, fontsize=9)
        _style_ax(ax)
        return

    use_log = (nonzero.max() / max(nonzero.min(), 1)) > 50

    if use_log:
        log_min = int(np.floor(np.log10(nonzero.min())))
        log_max = int(np.ceil(np.log10(nonzero.max())))
        if log_max <= log_min:
            log_max = log_min + 1
        bins = np.logspace(log_min, log_max, 30)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
        ax.xaxis.set_minor_locator(NullLocator())
    else:
        bins = 30
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

    counts, edges = np.histogram(nonzero, bins=bins)
    ax.bar(edges[:-1], counts, width=np.diff(edges) * 0.88,
           align="edge", color=color, alpha=0.85, zorder=2)

    if use_log:
        xlim_left = 10 ** (np.log10(nonzero.min()) - 0.5)
        ax.set_xlim(xlim_left, 10 ** (log_max + 0.15))
    else:
        data_range = nonzero.max() - nonzero.min()
        ax.set_xlim(
            max(0, nonzero.min() - data_range * 0.05),
            nonzero.max() + data_range * 0.08,
        )

    n_decades = (log_max - log_min) if use_log else 0
    if use_log and n_decades > 4:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=7)
    else:
        plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)

    med = float(nonzero.median())

    # B1 + B2: near-black median line, only drawn when visually meaningful
    xlim_left_val = ax.get_xlim()[0]
    if use_log:
        med_visible = med > xlim_left_val * 2.0   # must be at least 1 tick in from spine
    else:
        med_visible = med > xlim_left_val + (nonzero.max() - xlim_left_val) * 0.02
    if med_visible:
        ax.axvline(med, color=MEDIAN_LINE, linewidth=1.2, linestyle="--",
                   alpha=0.70, zorder=4)

    # B3: fixed-width stats block using column padding
    q1  = float(nonzero.quantile(0.25))
    q3  = float(nonzero.quantile(0.75))
    COL = 7   # label column width
    rows = []
    if n_zeros > 0:
        rows.append(f"{'Zeros':<{COL}} {n_zeros / N * 100:.1f}%  (n={n_zeros:,})")
    rows += [
        f"{'Median':<{COL}} {_fmt_stat(med)}",
        f"{'Mean':<{COL}} {_fmt_stat(float(nonzero.mean()))}",
        f"{'IQR':<{COL}} {_fmt_iqr(q1, q3)}",
        f"{'Max':<{COL}} {_fmt_stat(float(nonzero.max()))}",
    ]
    ax.text(0.98, 0.96, "\n".join(rows),
            transform=ax.transAxes, fontsize=8,
            va="top", ha="right", family="monospace",
            zorder=5,
            bbox=dict(facecolor="#F9FAFB", edgecolor=RULE,
                      boxstyle="round,pad=0.4", alpha=0.95))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_ylabel("Number of Tweets", fontsize=8.5, color=TXT_MED)
    _style_ax(ax)


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def get_univariate_for_authors(data_source, save_path=None):
    """
    Dataset 1 · Authors.
    New fixes: C1, C2, C3.
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(
        df, ["author_followers", "author_following",
             "author_isBlueVerified", "author_location"]
    )

    N   = len(df)
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Univariate Analysis \u2014 Author Profiles",
        fontsize=17, fontweight="bold", color=TXT, y=0.98,
    )

    # C3: left margin widened so long y-axis labels don't clip
    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        top=0.90, bottom=0.07, left=0.17, right=0.97,   # widened for long ytick labels
        hspace=0.38, wspace=0.35,
        height_ratios=[1, 1.1],
    )
    ax_fol  = fig.add_subplot(gs[0, 0])
    ax_fing = fig.add_subplot(gs[0, 1])
    ax_ver  = fig.add_subplot(gs[0, 2])
    ax_loc  = fig.add_subplot(gs[1, :])

    _plot_histogram(ax_fol,  df["author_followers"], PALETTE["author_followers"], "Follower Count")
    _plot_histogram(ax_fing, df["author_following"], PALETTE["author_following"], "Following Count")

    # Verification bar
    vcounts = (
        df["author_isBlueVerified"]
        .map({False: "Not Verified", True: "Verified"})
        .value_counts()
        .reindex(["Not Verified", "Verified"])
    )
    bars = ax_ver.bar(vcounts.index, vcounts.values, color="#3B82F6", width=0.5, zorder=2)
    for bar, val in zip(bars, vcounts.values):
        ax_ver.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + N * 0.012,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax_ver.set_title("Author Verification")
    ax_ver.set_ylim(0, vcounts.max() * 1.20)
    ax_ver.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax_ver)

    # Top Author Locations
    loc_raw    = df["author_location"].fillna("Unknown").str.strip().replace({"": "Unknown"})
    loc_counts = loc_raw.value_counts()
    top_n      = 15
    top        = loc_counts.head(top_n)
    others     = loc_counts.iloc[top_n:].sum()

    # Build series with raw labels first, then truncate
    loc_series = pd.concat([top, pd.Series({"Others": others})]).sort_values()
    raw_labels  = list(loc_series.index)           # keep "Others" detectable — C2
    disp_labels = [_truncate_label(str(lbl)) for lbl in raw_labels]

    bar_colors = [
        "#CBD5E1" if lbl == "Others" else "#3B82F6"
        for lbl in raw_labels                       # C2: check raw, not truncated
    ]

    ax_loc.tick_params(axis="y", labelsize=8.5)
    bars = ax_loc.barh(disp_labels, loc_series.values, color=bar_colors, height=0.6, zorder=2)
    max_val   = loc_series.max()
    right_pad = 1.30 if max_val > 50 else 1.40
    for bar, val in zip(bars, loc_series.values):
        ax_loc.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({val/N*100:.1f}%)",
            va="center", fontsize=9, color=TXT_MED,
        )
    ax_loc.set_title("Top Author Locations", pad=10)
    ax_loc.set_xlabel("Number of Authors", fontsize=9, color=TXT_MED)
    ax_loc.set_xlim(0, max_val * right_pad)
    ax_loc.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax_loc, grid_axis="x")

    _safe_show(fig, save_path)


def get_univariate_for_tweets(data_source, save_path=None):
    """Dataset 2 · Tweets — engagement histograms."""
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_tweet_columns(df)
    df = _normalize_dtypes(df)

    metrics = [
        ("retweetCount",  "Retweets"),
        ("likeCount",     "Likes"),
        ("viewCount",     "Views"),
        ("quoteCount",    "Quotes"),
        ("replyCount",    "Replies"),
        ("bookmarkCount", "Bookmarks"),
    ]
    _validate_columns(df, [m for m, _ in metrics])

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(
        "Univariate Analysis \u2014 Tweet Engagement",
        fontsize=17, fontweight="bold", color=TXT, y=0.99,
    )
    for ax, (col, title) in zip(axes.flat, metrics):
        _plot_histogram(ax, df[col], PALETTE[col], title)

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    plt.subplots_adjust(top=0.95)
    _safe_show(fig, save_path)


def get_univariate_for_tweet_categoricals(data_source, save_path=None):
    """
    Dataset 2 · Tweets — categorical columns.
    New fixes: A1, A2, A3.
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(df, ["isReply", "lang", "author_isBlueVerified"])

    N = len(df)

    # Distinct palette per panel — each chart measures a different concept
    C_REPLY  = "#3B82F6"   # blue   — Reply Status
    C_LANG   = "#14B8A6"   # teal   — Languages
    C_VERIFY = "#8B5CF6"   # violet — Verification

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(
        "Univariate Analysis \u2014 Tweet Categoricals",
        fontsize=17, fontweight="bold", color=TXT, y=1.02,
    )

    # 1. Reply Status
    ax = axes[0]
    counts = (
        df["isReply"]
        .map({False: "Original", True: "Reply"})
        .value_counts()
        .reindex(["Original", "Reply"])
    )
    bars = ax.bar(counts.index, counts.values, color=C_REPLY, width=0.5, zorder=2)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.02,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax.set_title("Reply Status")
    ax.set_ylabel("Number of Tweets", fontsize=9, color=TXT_MED)
    ax.set_ylim(0, counts.max() * 1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)

    # 2. Top Languages
    ax = axes[1]
    lang_counts = (
        df["lang"].map(lambda x: LANG_LABELS.get(x, x))
        .value_counts().head(5)
    )
    bars = ax.barh(
        lang_counts.index[::-1], lang_counts.values[::-1],
        color=C_LANG, height=0.55, zorder=2,
    )
    label_offset = lang_counts.max() * 0.01
    for bar, val in zip(bars, lang_counts.values[::-1]):
        ax.text(
            val + label_offset,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({val/N*100:.1f}%)",
            va="center", fontsize=9, color=TXT_MED,
        )
    ax.set_title("Top Languages")
    ax.set_xlabel("Number of Tweets", fontsize=9, color=TXT_MED, labelpad=8)
    ax.set_xlim(0, lang_counts.max() * 1.28)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax, grid_axis="x")

    # 3. Author Verification
    ax = axes[2]
    vcounts = (
        df["author_isBlueVerified"]
        .map({False: "Not Verified", True: "Verified"})
        .value_counts()
        .reindex(["Not Verified", "Verified"])
    )
    bars = ax.bar(vcounts.index, vcounts.values, color=C_VERIFY, width=0.5, zorder=2)
    for bar, val in zip(bars, vcounts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vcounts.max() * 0.02,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax.set_title("Author Verification")
    ax.set_ylabel("Number of Tweets", fontsize=9, color=TXT_MED)
    ax.set_ylim(0, vcounts.max() * 1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.12)
    _safe_show(fig, save_path)


def get_temporal_distribution(data_source, save_path=None,
                               freq="D", top_n_lang=2):
    """
    Dataset 2 · Tweets — volume over time.
    New fixes: D1, D2, D3.
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(df, ["createdAt", "lang"])
    df = df.dropna(subset=["createdAt"])

    df["_period"]      = df["createdAt"].dt.to_period(freq).dt.to_timestamp()
    df["_lang_mapped"] = df["lang"].map(lambda x: LANG_LABELS.get(x, x))
    top_langs          = df["_lang_mapped"].value_counts().head(top_n_lang).index.tolist()
    df["_lang_grp"]    = df["_lang_mapped"].where(df["_lang_mapped"].isin(top_langs), other="Other")

    pivot = (
        df.groupby(["_period", "_lang_grp"])
        .size()
        .unstack(fill_value=0)
    )
    ordered = top_langs + [c for c in pivot.columns if c not in top_langs]
    pivot   = pivot.reindex(columns=ordered, fill_value=0)
    colors  = [LANG_COLORS.get(c, "#8B5CF6") for c in pivot.columns]

    freq_label = "Day" if freq == "D" else "Week"

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.suptitle(
        "Univariate Analysis \u2014 Tweet Volume Over Time",
        fontsize=17, fontweight="bold", color=TXT, y=1.02,
    )

    # Use ax.stackplot() directly — pivot.plot.area() with linewidth=0 can
    # still render faint white boundary lines between stacked layers in some
    # matplotlib versions. stackplot() with linewidth=0 never does this.
    total_by_period_pre = pivot.sum(axis=1)
    ax.stackplot(
        pivot.index,
        [pivot[col].values for col in pivot.columns],
        labels=list(pivot.columns),
        colors=colors, alpha=0.85,
        linewidth=0, edgecolor="none",
    )

    # D1: no arrowprops — avoids white hairline cutting through area fill
    # D1 fix: ax.scatter() used instead of ax.plot() to avoid the
    # "axis already has a converter set" UserWarning that ax.plot() triggers
    # after pivot.plot.area() has registered pandas' datetime converter.
    total_by_period = total_by_period_pre.sort_values(ascending=False)
    top3 = total_by_period.head(3)
    for date, val in top3.items():
        ax.annotate(
            pd.Timestamp(date).strftime("%b %d"),
            xy=(date, val),
            xytext=(0, 14),
            textcoords="offset points",
            ha="center", fontsize=8.5,
            color=TXT_MED, fontweight="bold",
            arrowprops=None,
        )
    # Batch all dots in a single scatter call — no converter conflict
    ax.scatter(
        top3.index, top3.values,
        color=TXT_MED, s=18, zorder=5,
    )

    ax.set_xlabel(f"Date (per {freq_label})", fontsize=10, color=TXT_MED)
    ax.set_ylabel("Number of Tweets",         fontsize=10, color=TXT_MED)
    # Restore clean date formatting lost when switching from pandas .plot.area()
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    # D2: choose legend position based on data density in last quarter
    n_periods = len(pivot)
    last_q    = pivot.iloc[-(n_periods // 4):].sum().sum() if n_periods >= 4 else 1
    first_q   = pivot.iloc[:(n_periods // 4)].sum().sum()  if n_periods >= 4 else 0
    leg_loc   = "upper right" if last_q < first_q * 0.5 else "upper left"
    ax.legend(
        title="Language", fontsize=9, title_fontsize=9,
        framealpha=0.92, edgecolor=RULE, loc=leg_loc,
    )

    _style_ax(ax)
    plt.tight_layout()
    _safe_show(fig, save_path)
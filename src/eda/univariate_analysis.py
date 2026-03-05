"""
univariate_analysis.py
======================
Portfolio-grade univariate visualisation for the PH Flood Control
Twitter/X EDA — two datasets, four public functions.

Datasets
--------
Dataset 1 · Authors  (227 rows × 8 columns)
Dataset 2 · Tweets   (195,744 rows × 16 columns)

Public API
----------
get_univariate_for_tweets(data_source, save_path=None)
    2×3 grid — all 6 engagement metrics with zero-inflation annotations.

get_univariate_for_tweet_categoricals(data_source, save_path=None)
    Reply status · Language breakdown · Author verification.

get_univariate_for_authors(data_source, save_path=None)
    Follower / following histograms + verification bar + location chart.

get_temporal_distribution(data_source, save_path=None, freq="D")
    Tweet volume over time, annotated with top spike dates.
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
    AutoMinorLocator,
    MaxNLocator,
    NullLocator,
    NullFormatter,
)

# ──────────────────────────────────────────────────────────────────
# 🎨  Design tokens
# ──────────────────────────────────────────────────────────────────

BG       = "#F8FAFC"
CARD_BG  = "#FFFFFF"
TXT      = "#0F172A"
TXT_MED  = "#475569"
TXT_LT   = "#94A3B8"
RULE     = "#E2E8F0"

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

def _fmt_k(x, _):
    """Format numbers as K / M for axis ticks."""
    if x >= 1_000_000:
        v = x/1_000_000; return f"{v:.1f}M" if v < 10 and v != int(v) else f"{int(v)}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{int(x)}"
def _fmt_iqr(q1, q3):
    """Format IQR bounds at the same scale so they never mix e.g. 175-1K."""
    if q3 >= 1_000_000:
        return f"{_fmt_k(q1, None)}–{_fmt_k(q3, None)}"
    elif q3 >= 1_000:
        # Both in K territory — express q1 in K too
        v1 = q1 / 1_000
        v3 = q3 / 1_000
        fmt1 = f"{v1:.1f}K" if v1 != int(v1) else f"{int(v1)}K"
        fmt3 = f"{v3:.1f}K" if v3 != int(v3) else f"{int(v3)}K"
        return f"{fmt1}–{fmt3}"
    else:
        return f"{int(q1)}–{int(q3)}"




def _truncate_label(label, max_len=28):
    """Truncate long location labels with an ellipsis."""
    return label if len(label) <= max_len else label[:max_len - 1] + "…"


def _safe_show(fig, save_path=None):
    """
    Display or save without the FigureCanvasAgg UserWarning.
    Falls back to auto-save when no GUI backend is available.
    """
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
    """Minimal, consistent spine + grid styling."""
    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom", "left"))
        if loc in ("bottom", "left"):
            spine.set_color(RULE)
            spine.set_linewidth(0.8)

    ax.tick_params(which="major", length=5, width=0.8, labelsize=9)

    if grid_axis == "y":
        ax.yaxis.grid(True, which="major", color=RULE, linewidth=0.6, zorder=0)
        ax.xaxis.grid(False)
    else:
        ax.xaxis.grid(True, which="major", color=RULE, linewidth=0.6, zorder=0)
        ax.yaxis.grid(False)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, labelpad=5, color=TXT_MED)


def _normalize_dtypes(df):
    """
    Fix mixed-type columns flagged by the data quality report.

    Tweets  → isReply, author_isBlueVerified: bool
            → createdAt: datetime64 (tz stripped)
            → pseudo_inReplyToUsername: str (NaN preserved)
    Authors → author_isBlueVerified: bool
            → author_createdAt: datetime64 (tz stripped)
    """
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
    """Accept either camelCase or snake_case engagement column names."""
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


def _set_clean_log_ticks(ax):
    """
    Apply sparse, readable log-scale ticks.

    Only labels decade boundaries (1, 10, 100, 1K, 10K, 100K, 1M, 10M).
    Suppresses the intermediate ×2 and ×5 sub-ticks that cause crowding.
    """
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.xaxis.set_minor_locator(NullLocator())


def _plot_histogram(ax, data, color, title):
    """
    Histogram for a single count column.

    Collision-proof design
    ----------------------
    - Zero-inflation shown as first row of stats panel (inside axes).
      Nothing is rendered outside the axes box — no y > 1 transAxes text.
    - No separate median label — the stats panel already shows it.
    - xlim set with explicit left + right AFTER bar() is drawn.
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

    # xlim — explicit both bounds, set AFTER bar()
    if use_log:
        # Anchor left edge to actual data min — eliminates dead space before first bar.
        # Floor at tick_safe so the first decade label always stays visible.
        data_anchored = 10 ** (np.log10(nonzero.min()) - 0.15)
        tick_safe     = 10 ** (log_min - 0.2)
        xlim_left     = max(data_anchored, tick_safe)
        ax.set_xlim(xlim_left, 10 ** (log_max + 0.15))

    # Scale tick font down for wide log ranges to prevent crowding
    n_decades = (log_max - log_min) if use_log else 0
    if use_log and n_decades > 4:
        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", fontsize=7)
    else:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    med = float(nonzero.median())

    # Stats panel — zero info as first row when present, fully inside axes
    q1 = float(nonzero.quantile(0.25))
    q3 = float(nonzero.quantile(0.75))
    rows = []
    if n_zeros > 0:
        rows.append(f"Zeros  {n_zeros / N * 100:.1f}%  (n={n_zeros:,})")
    rows += [
        f"Median {_fmt_k(med, None)}",
        f"Mean   {_fmt_k(nonzero.mean(), None)}",
        f"IQR    {_fmt_iqr(q1, q3)}",
        f"Max    {_fmt_k(nonzero.max(), None)}",
    ]
    ax.text(0.98, 0.96, "\n".join(rows),
            transform=ax.transAxes, fontsize=7.5,
            va="top", ha="right", family="monospace",
            zorder=5,
            bbox=dict(facecolor="#F9FAFB", edgecolor=RULE,
                      boxstyle="round,pad=0.4", alpha=0.95))

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)


# 📊  PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def get_univariate_for_authors(data_source, save_path=None):
    """
    Dataset 1 · Authors — all relevant columns.

    Produces a 2-row layout:
      Row 1 (3 panels): Follower Count · Following Count · Verification
      Row 2 (full width): Top Author Locations (horizontal bar)

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path : str, optional
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
        fontsize=17, fontweight="bold", color=TXT, y=1.01,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.65, wspace=0.35,
        height_ratios=[1, 0.9],
    )
    ax_fol  = fig.add_subplot(gs[0, 0])
    ax_fing = fig.add_subplot(gs[0, 1])
    ax_ver  = fig.add_subplot(gs[0, 2])
    ax_loc  = fig.add_subplot(gs[1, :])

    # Follower Count
    _plot_histogram(
        ax_fol, df["author_followers"],
        PALETTE["author_followers"], "Follower Count",
    )

    # Following Count
    _plot_histogram(
        ax_fing, df["author_following"],
        PALETTE["author_following"], "Following Count",
    )

    # Verification bar
    vcounts = (
        df["author_isBlueVerified"]
        .map({False: "Not Verified", True: "Verified"})
        .value_counts()
        .reindex(["Not Verified", "Verified"])
    )
    bars = ax_ver.bar(
        vcounts.index, vcounts.values,
        color="#3B82F6", width=0.5, zorder=2,
    )
    for bar, val in zip(bars, vcounts.values):
        ax_ver.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + N * 0.012,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax_ver.set_title("Author Verification")
    ax_ver.set_ylim(0, vcounts.max() * 1.18)
    ax_ver.yaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    _style_ax(ax_ver)

    # Top Author Locations
    loc_raw    = df["author_location"].fillna("Unknown").str.strip()
    loc_raw    = loc_raw.replace({"": "Unknown"})
    loc_counts = loc_raw.value_counts()

    top_n  = 10
    top    = loc_counts.head(top_n)
    others = loc_counts.iloc[top_n:].sum()

    loc_plot = pd.concat(
        [top, pd.Series({"Others": others})]
    ).sort_values()

    loc_plot.index = [_truncate_label(str(lbl)) for lbl in loc_plot.index]

    colors = ["#3B82F6"] * len(top) + ["#94A3B8"]
    bars   = ax_loc.barh(
        loc_plot.index, loc_plot.values,
        color=colors, height=0.6, zorder=2,
    )
    for bar, val in zip(bars, loc_plot.values):
        ax_loc.text(
            val + loc_plot.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({val/N*100:.1f}%)",
            va="center", fontsize=9, color=TXT_MED,
        )
    ax_loc.set_title("Top Author Locations", pad=10)
    ax_loc.set_xlabel("Number of Authors", fontsize=9, color=TXT_MED)
    ax_loc.set_xlim(0, loc_plot.max() * 1.22)
    ax_loc.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    _style_ax(ax_loc, grid_axis="x")

    _safe_show(fig, save_path)


def get_univariate_for_tweets(data_source, save_path=None):
    """
    Dataset 2 · Tweets — numeric engagement columns.

    Produces a 2×3 grid of histograms (non-zero values only):
      Row 1: Retweets · Likes · Views
      Row 2: Quotes   · Replies · Bookmarks

    Zero-inflation rate shown as italic text above each panel title.

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path : str, optional
    """
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
        fontsize=17, fontweight="bold", color=TXT, y=1.01,
    )

    for ax, (col, title) in zip(axes.flat, metrics):
        _plot_histogram(ax, df[col], PALETTE[col], title)

    # Extra top margin so the zero subtitles don't get clipped
    plt.tight_layout(h_pad=5.0, w_pad=2.5)
    plt.subplots_adjust(top=0.93)
    _safe_show(fig, save_path)


def get_univariate_for_tweet_categoricals(data_source, save_path=None):
    """
    Dataset 2 · Tweets — categorical columns.

    Produces three bar charts side by side:
      1. Reply Status        (Original vs Reply)
      2. Top Languages       (sorted horizontal bars)
      3. Author Verification (Not Verified vs Verified)

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path : str, optional
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(df, ["isReply", "lang", "author_isBlueVerified"])

    N     = len(df)
    C_BAR = "#3B82F6"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Univariate Analysis \u2014 Tweet Categoricals",
        fontsize=17, fontweight="bold", color=TXT, y=1.03,
    )

    # 1. Reply Status
    ax = axes[0]
    counts = (
        df["isReply"]
        .map({False: "Original", True: "Reply"})
        .value_counts()
        .reindex(["Original", "Reply"])
    )
    bars = ax.bar(counts.index, counts.values,
                  color=C_BAR, width=0.5, zorder=2)
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + N * 0.005,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax.set_title("Reply Status")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)

    # 2. Top Languages
    ax = axes[1]
    lang_counts = (df["lang"]
                  .map(lambda x: LANG_LABELS.get(x, x))
                  .value_counts().head(5))
    bars = ax.barh(
        lang_counts.index[::-1], lang_counts.values[::-1],
        color=C_BAR, height=0.55, zorder=2,
    )
    for bar, val in zip(bars, lang_counts.values[::-1]):
        ax.text(
            val + N * 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,} ({val/N*100:.1f}%)",
            va="center", fontsize=9, color=TXT_MED,
        )
    ax.set_title("Top Languages")
    ax.set_xlabel("Number of Tweets", fontsize=9, color=TXT_MED)
    ax.set_xlim(0, lang_counts.max() * 1.22)
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
    bars = ax.bar(vcounts.index, vcounts.values,
                  color=C_BAR, width=0.5, zorder=2)
    for bar, val in zip(bars, vcounts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + N * 0.005,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax.set_title("Author Verification")
    ax.set_ylim(0, vcounts.max() * 1.15)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)

    plt.tight_layout()
    _safe_show(fig, save_path)


def get_temporal_distribution(data_source, save_path=None,
                               freq="D", top_n_lang=2):
    """
    Dataset 2 · Tweets — tweet volume over time.

    Area chart stacked by language, with top-3 spike dates annotated
    so narrative context (e.g. typhoon events) is immediately visible.

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path : str, optional
    freq : str
        Pandas offset alias: 'D' daily (default), 'W' weekly.
    top_n_lang : int
        Number of top languages shown individually; rest grouped as 'Other'.
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(df, ["createdAt", "lang"])

    df = df.dropna(subset=["createdAt"])

    # tz already stripped in _normalize_dtypes — no UserWarning here
    df["_period"]   = df["createdAt"].dt.to_period(freq).dt.to_timestamp()
    df["_lang_mapped"] = df["lang"].map(lambda x: LANG_LABELS.get(x, x))
    top_langs         = df["_lang_mapped"].value_counts().head(top_n_lang).index.tolist()
    df["_lang_grp"]   = df["_lang_mapped"].where(df["_lang_mapped"].isin(top_langs), other="Other")

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

    pivot.plot.area(ax=ax, color=colors, alpha=0.82, linewidth=0)

    # Annotate top-3 spike dates
    total_by_period = pivot.sum(axis=1).sort_values(ascending=False)
    for date, val in total_by_period.head(3).items():
        ax.annotate(
            pd.Timestamp(date).strftime("%b %d"),
            xy=(date, val),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center", fontsize=8.5,
            color=TXT_MED, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=RULE, lw=0.8),
        )

    ax.set_xlabel(f"Date (per {freq_label})", fontsize=10, color=TXT_MED)
    ax.set_ylabel("Number of Tweets",         fontsize=10, color=TXT_MED)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(
        title="Language", fontsize=9, title_fontsize=9,
        framealpha=0.9, edgecolor=RULE, loc="upper left",
    )

    _style_ax(ax)
    plt.tight_layout()
    _safe_show(fig, save_path)
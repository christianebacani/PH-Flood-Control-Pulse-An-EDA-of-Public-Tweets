"""
univariate_analysis.py
======================
Portfolio-grade univariate visualisation for the PH Flood Control
Twitter/X EDA — two datasets, four public functions.

Datasets
--------
Dataset 1 · Tweets   (195,744 rows × 16 columns)
Dataset 2 · Authors  (227 rows × 8 columns)

Public API
----------
get_univariate_for_tweets(data_source, save_path=None)
    2×3 grid — all 6 engagement metrics with zero-inflation labels.

get_univariate_for_tweet_categoricals(data_source, save_path=None)
    Reply status · Language breakdown · Author verification.

get_univariate_for_authors(data_source, save_path=None)
    Follower / following histograms + verification bar + location chart.

get_temporal_distribution(data_source, save_path=None, freq="D")
    Tweet volume over time, annotated with top spike dates.
"""

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
ZERO_CLR = "#CBD5E1"

# Per-metric palette
PALETTE = {
    "retweetCount":     "#3B82F6",   # blue
    "likeCount":        "#14B8A6",   # teal
    "viewCount":        "#8B5CF6",   # purple
    "quoteCount":       "#F59E0B",   # amber
    "replyCount":       "#EF4444",   # red
    "bookmarkCount":    "#10B981",   # emerald
    "author_followers": "#14B8A6",
    "author_following": "#8B5CF6",
}

LANG_COLORS = {
    "tl":    "#3B82F6",
    "en":    "#14B8A6",
    "Other": "#CBD5E1",
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
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}K"
    return f"{int(x)}"


def _safe_show(fig, save_path=None):
    """
    Display or save without the FigureCanvasAgg UserWarning.
    Falls back to auto-save when no GUI backend is available.
    """
    if save_path:
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
    ax.tick_params(which="minor", length=3, width=0.5)

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

    Tweets dataset
    --------------
    isReply, author_isBlueVerified  → bool
    createdAt                       → datetime64[utc]
    pseudo_inReplyToUsername        → str (NaN preserved)

    Authors dataset
    ---------------
    author_isBlueVerified           → bool
    author_createdAt                → datetime64[utc]
    """
    for col in ("isReply", "author_isBlueVerified"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes")
                if pd.notna(v) else False
            ).astype(bool)

    for col in ("createdAt", "author_createdAt"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

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


def _plot_zero_inflated_histogram(ax, data, color, title):
    """
    Single zero-inflated count histogram.

    - Grey bar for zeros with 'Zero N (X%)' label
    - Log x-axis when value range > 50×
    - Dashed median line + rotated label (only when median > 0)
    - Stats panel: Median / Mean / IQR / Max
    """
    data = pd.to_numeric(data, errors="coerce").dropna()
    N    = len(data)
    ax.set_title(title)

    if N == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                transform=ax.transAxes, color=TXT_MED)
        return

    n_zeros  = int((data == 0).sum())
    pct_zero = n_zeros / N * 100
    nonzero  = data[data > 0]

    # ── Zero bar ──────────────────────────────────────────────────
    if n_zeros > 0:
        zbar = ax.bar(0.5, n_zeros, width=0.5,
                      color=ZERO_CLR, alpha=0.9, zorder=2)
        ax.text(
            zbar[0].get_x() + zbar[0].get_width() / 2,
            n_zeros,
            f"Zero\n{n_zeros:,} ({pct_zero:.1f}%)",
            ha="center", va="bottom", fontsize=8,
            color=TXT_MED, fontweight="bold",
        )

    # ── Non-zero histogram ────────────────────────────────────────
    if len(nonzero) > 0:
        use_log = nonzero.max() / max(nonzero.min(), 1) > 50

        if use_log:
            bins = np.logspace(
                np.floor(np.log10(nonzero.min())),
                np.ceil(np.log10(nonzero.max())),
                26,
            )
            ax.set_xscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1, 2, 5]))
            ax.xaxis.set_minor_locator(
                LogLocator(base=10.0, subs=np.arange(1, 10) * 0.1)
            )
        else:
            bins = 25
            ax.set_xscale("linear")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.xaxis.set_minor_locator(AutoMinorLocator())

        counts, edges = np.histogram(nonzero, bins=bins)
        ax.bar(
            edges[:-1], counts,
            width=np.diff(edges) * 0.88,
            align="edge", color=color, alpha=0.85, zorder=2,
        )

        # Median line
        med = data.median()
        if med > 0:
            ax.axvline(med, linestyle="--", linewidth=1.5,
                       color=color, zorder=3)
            ax.text(
                med, ax.get_ylim()[1] * 0.97,
                f"Median: {_fmt_k(med, None)}",
                rotation=90, va="top", ha="right",
                fontsize=8, color=color,
            )

    # ── Stats panel ───────────────────────────────────────────────
    med    = data.median()
    q1, q3 = data.quantile([0.25, 0.75])
    panel  = (
        f"Median  {_fmt_k(med, None)}\n"
        f"Mean    {_fmt_k(data.mean(), None)}\n"
        f"IQR     {_fmt_k(q1, None)}–{_fmt_k(q3, None)}\n"
        f"Max     {_fmt_k(data.max(), None)}"
    )
    ax.text(
        0.98, 0.98, panel,
        transform=ax.transAxes, fontsize=8.5,
        va="top", ha="right", family="monospace",
        bbox=dict(facecolor="#F9FAFB", edgecolor=RULE,
                  boxstyle="round,pad=0.45", alpha=0.9),
    )

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    _style_ax(ax)


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def get_univariate_for_tweets(data_source, save_path=None):
    """
    Dataset 1 · Tweets — numeric engagement columns.

    Produces a 2×3 grid of zero-inflated histograms:
      Row 1: Retweets · Likes · Views
      Row 2: Quotes   · Replies · Bookmarks

    Parameters
    ----------
    data_source : str or pd.DataFrame
        Path to tweets CSV or a pre-loaded DataFrame.
    save_path : str, optional
        Output path (e.g. 'outputs/tweet_engagement.png').
        Auto-saves to 'univariate_output.png' when running
        in a non-interactive (script) environment.
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
        "Univariate Analysis — Tweet Engagement",
        fontsize=17, fontweight="bold", color=TXT, y=1.01,
    )

    for ax, (col, title) in zip(axes.flat, metrics):
        _plot_zero_inflated_histogram(ax, df[col], PALETTE[col], title)

    plt.tight_layout(h_pad=3.5, w_pad=2.5)
    _safe_show(fig, save_path)


def get_univariate_for_tweet_categoricals(data_source, save_path=None):
    """
    Dataset 1 · Tweets — categorical columns.

    Produces three bar charts side by side:
      1. Reply Status       (Original vs Reply)
      2. Top Languages      (sorted horizontal bars)
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

    N   = len(df)
    C_BAR = "#3B82F6"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(
        "Univariate Analysis — Tweet Categoricals",
        fontsize=17, fontweight="bold", color=TXT, y=1.03,
    )

    # ── 1. Reply Status ───────────────────────────────────────────
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

    # ── 2. Top Languages ──────────────────────────────────────────
    ax = axes[1]
    lang_counts = df["lang"].value_counts().head(5)
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

    # ── 3. Author Verification ────────────────────────────────────
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


def get_univariate_for_authors(data_source, save_path=None):
    """
    Dataset 2 · Authors — all relevant columns.

    Produces a 2-row layout:
      Row 1 (3 panels): Follower Count · Following Count · Verification
      Row 2 (full width): Top Author Locations (horizontal bar)

    Parameters
    ----------
    data_source : str or pd.DataFrame
        Path to authors CSV or a pre-loaded DataFrame.
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
        "Univariate Analysis — Author Profiles",
        fontsize=17, fontweight="bold", color=TXT, y=1.01,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.55, wspace=0.35,
        height_ratios=[1, 0.9],
    )
    ax_fol  = fig.add_subplot(gs[0, 0])
    ax_fing = fig.add_subplot(gs[0, 1])
    ax_ver  = fig.add_subplot(gs[0, 2])
    ax_loc  = fig.add_subplot(gs[1, :])

    # ── Follower Count ────────────────────────────────────────────
    _plot_zero_inflated_histogram(
        ax_fol, df["author_followers"],
        PALETTE["author_followers"], "Follower Count",
    )

    # ── Following Count ───────────────────────────────────────────
    _plot_zero_inflated_histogram(
        ax_fing, df["author_following"],
        PALETTE["author_following"], "Following Count",
    )

    # ── Verification bar ──────────────────────────────────────────
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

    # ── Top Author Locations ──────────────────────────────────────
    loc_raw    = df["author_location"].fillna("Unknown").str.strip()
    loc_raw    = loc_raw.replace({"": "Unknown"})
    loc_counts = loc_raw.value_counts()

    top_n  = 10
    top    = loc_counts.head(top_n)
    others = loc_counts.iloc[top_n:].sum()

    loc_plot = pd.concat(
        [top, pd.Series({"Others": others})]
    ).sort_values()

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


def get_temporal_distribution(data_source, save_path=None,
                               freq="D", top_n_lang=2):
    """
    Dataset 1 · Tweets — tweet volume over time.

    Area chart stacked by language, with top-3 spike dates annotated
    so narrative context (e.g. typhoon events) is immediately visible.

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path : str, optional
    freq : str
        Pandas offset alias — 'D' daily (default), 'W' weekly.
    top_n_lang : int
        Number of top languages shown individually; rest → 'Other'.
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(df, ["createdAt", "lang"])

    df = df.dropna(subset=["createdAt"])
    df["_period"] = df["createdAt"].dt.tz_localize(None).dt.to_period(freq).dt.to_timestamp()
    top_langs       = df["lang"].value_counts().head(top_n_lang).index.tolist()
    df["_lang_grp"] = df["lang"].where(df["lang"].isin(top_langs), other="Other")

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
        "Univariate Analysis — Tweet Volume Over Time",
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
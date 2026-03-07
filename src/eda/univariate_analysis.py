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
GRID    = "#CBD5E1"
MEDIAN_LINE = "#1E293B"

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
    "Filipino (tl)": "#60A5FA",
    "English (en)":  "#34D399",
    "Other":         "#E2E8F0",
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
    if q3 >= 1_000_000:
        return f"{_fmt_stat(q1)}–{_fmt_stat(q3)}"
    elif q3 >= 1_000:
        v3   = q3 / 1_000
        fmt3 = f"{v3:.1f}K" if v3 != int(v3) else f"{int(v3)}K"
        if q1 < 1_000:
            fmt1 = f"{int(q1)}" if q1 == int(q1) else f"{q1:.0f}"
        else:
            v1   = q1 / 1_000
            fmt1 = f"{v1:.1f}K" if v1 != int(v1) else f"{int(v1)}K"
        return f"{fmt1}–{fmt3}"
    else:
        fmt1 = f"{int(q1)}" if q1 == int(q1) else f"{q1:.1f}"
        fmt3 = f"{int(q3)}" if q3 == int(q3) else f"{q3:.1f}"
        return f"{fmt1}–{fmt3}"


def _truncate_label(label: str, max_len: int = 35) -> str:
    """
    Truncate at a word boundary.
    max_len raised from 28 → 35 so labels like
    'National Capital Region, Republic of the Philippines' fit without
    cutting mid-word on most common Philippine location strings.
    """
    if len(label) <= max_len:
        return label
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


def _plot_histogram(ax, data, color, title, ylabel="Number of Tweets"):
    data    = pd.to_numeric(data, errors="coerce").dropna()
    N       = len(data)
    nonzero = data[data > 0]
    n_zeros = int((data == 0).sum())

    ax.set_title(title)
    _COUNT_LABELS = {
        "Retweets":       "Retweet Count",
        "Likes":          "Like Count",
        "Views":          "View Count",
        "Quotes":         "Quote Count",
        "Replies":        "Reply Count",
        "Bookmarks":      "Bookmark Count",
        "Follower Count": "Follower Count",
        "Following Count":"Following Count",
    }
    fallback = title if title.endswith("Count") else f"{title} Count"
    ax.set_xlabel(_COUNT_LABELS.get(title, fallback),
                  fontsize=9, color=TXT_MED, labelpad=4)

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

    xlim_left_val  = ax.get_xlim()[0]
    xlim_right_val = ax.get_xlim()[1]
    if use_log:
        log_span    = np.log10(xlim_right_val) - np.log10(xlim_left_val)
        med_frac    = (np.log10(max(med, 1e-9)) - np.log10(xlim_left_val)) / log_span
        med_visible = med_frac > 0.20
    else:
        lin_span    = xlim_right_val - xlim_left_val
        med_visible = (med - xlim_left_val) / lin_span > 0.05

    if med_visible:
        ax.axvline(med, color=MEDIAN_LINE, linewidth=1.2, linestyle="--",
                   alpha=0.70, zorder=4)

    q1  = float(nonzero.quantile(0.25))
    q3  = float(nonzero.quantile(0.75))
    COL = 7
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
    ax.set_ylabel(ylabel, fontsize=8.5, color=TXT_MED)
    _style_ax(ax)


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def get_univariate_for_authors(data_source, save_path=None):
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

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        top=0.90, bottom=0.07, left=0.24, right=0.97,
        hspace=0.38, wspace=0.35,
        height_ratios=[1, 1.1],
    )
    ax_fol  = fig.add_subplot(gs[0, 0])
    ax_fing = fig.add_subplot(gs[0, 1])
    ax_ver  = fig.add_subplot(gs[0, 2])
    ax_loc  = fig.add_subplot(gs[1, :])

    _plot_histogram(ax_fol,  df["author_followers"], PALETTE["author_followers"],
                   "Follower Count",  ylabel="Number of Authors")
    _plot_histogram(ax_fing, df["author_following"], PALETTE["author_following"],
                   "Following Count", ylabel="Number of Authors")

    # ── Verification bar ──────────────────────────────────────────────────────
    vcounts = (
        df["author_isBlueVerified"]
        .map({False: "Not Verified", True: "Verified"})
        .value_counts()
        .reindex(["Not Verified", "Verified"])
    )
    bars = ax_ver.bar(vcounts.index, vcounts.values,
                      color="#3B82F6", width=0.5, zorder=2)
    for bar, val in zip(bars, vcounts.values):
        ax_ver.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + vcounts.max() * 0.02,
            f"{val:,} ({val/N*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, color=TXT_MED,
        )
    ax_ver.set_title("Author Verification")
    ax_ver.set_ylabel("Number of Authors", fontsize=9, color=TXT_MED)
    ax_ver.set_xlabel("Verification Status", fontsize=9, color=TXT_MED)
    ax_ver.set_ylim(0, vcounts.max() * 1.30)
    ax_ver.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax_ver)

    # ── Top Author Locations ───────────────────────────────────────────────────
    loc_raw    = df["author_location"].fillna("Unknown").str.strip().replace({"": "Unknown"})
    loc_counts = loc_raw.value_counts()

    # FIX: only keep locations appearing more than once to avoid inflating "Others"
    meaningful  = loc_counts[loc_counts > 1]
    top_n       = min(len(meaningful), 20)
    top         = meaningful.head(top_n)
    others      = loc_counts.iloc[top_n:].sum()

    loc_series  = pd.concat([top, pd.Series({"Others": others})]).sort_values()
    raw_labels  = list(loc_series.index)
    disp_labels = [_truncate_label(str(lbl)) for lbl in raw_labels]

    bar_colors  = [
        "#CBD5E1" if lbl == "Others" else "#3B82F6"
        for lbl in raw_labels
    ]

    ax_loc.tick_params(axis="y", labelsize=8.5)
    bars = ax_loc.barh(disp_labels, loc_series.values,
                       color=bar_colors, height=0.6, zorder=2)
    for lbl in ax_loc.get_yticklabels():
        lbl.set_clip_on(False)

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

    plt.subplots_adjust(left=0.24)
    _safe_show(fig, save_path)


def get_univariate_for_tweets(data_source, save_path=None):
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
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize_dtypes(df)
    _validate_columns(df, ["isReply", "lang", "author_isBlueVerified"])

    N = len(df)

    C_REPLY  = "#3B82F6"
    C_LANG   = "#14B8A6"
    C_VERIFY = "#8B5CF6"

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
    ax.set_xlabel("Tweet Type", fontsize=9, color=TXT_MED)
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
    min_visible  = lang_counts.max() * 0.005
    for bar, val in zip(bars, lang_counts.values[::-1]):
        display_x = max(val, min_visible)
        if val < min_visible:
            ax.barh(
                bar.get_y() + bar.get_height() / 2,
                min_visible, height=0.55,
                left=0, color=C_LANG, alpha=0.35, zorder=3,
            )
        ax.text(
            display_x + label_offset,
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
    ax.set_xlabel("Verification Status", fontsize=9, color=TXT_MED)
    ax.set_ylim(0, vcounts.max() * 1.22)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    _style_ax(ax)

    plt.tight_layout(pad=2.0)
    plt.subplots_adjust(bottom=0.12)
    _safe_show(fig, save_path)


def get_temporal_distribution(data_source, save_path=None,
                               freq="D", top_n_lang=2):
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

    total_by_period_pre = pivot.sum(axis=1)
    x_vals      = pivot.index
    cumulative  = np.zeros(len(pivot))
    handles     = []
    for col, color in zip(pivot.columns, colors):
        y_vals = pivot[col].values
        poly   = ax.fill_between(
            x_vals, cumulative, cumulative + y_vals,
            color=color, alpha=0.88,
            linewidth=0, label=col,
        )
        ax.plot(x_vals, cumulative + y_vals,
                color=color, linewidth=0.8, alpha=0.88, zorder=3)
        handles.append(poly)
        cumulative = cumulative + y_vals

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
    ax.scatter(
        top3.index, top3.values,
        color=TXT_MED, s=18, zorder=5,
    )

    ax.set_xlabel(f"Date (per {freq_label})", fontsize=10, color=TXT_MED)
    ax.set_ylabel("Number of Tweets",         fontsize=10, color=TXT_MED)

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate(rotation=0, ha="center")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))

    n_periods = len(pivot)
    last_q    = pivot.iloc[-(n_periods // 4):].sum().sum() if n_periods >= 4 else 1
    first_q   = pivot.iloc[:(n_periods // 4)].sum().sum()  if n_periods >= 4 else 0
    leg_loc   = "upper right" if last_q < first_q * 0.5 else "upper left"
    ax.legend(
        handles=handles,
        title="Language", fontsize=9, title_fontsize=9,
        framealpha=0.92, edgecolor=RULE, loc=leg_loc,
    )

    _style_ax(ax)
    plt.tight_layout()
    _safe_show(fig, save_path)
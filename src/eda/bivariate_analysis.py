import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, NullLocator
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# 🎨  Design tokens  (identical to univariate suite)
# ──────────────────────────────────────────────────────────────────

BG          = "#F8FAFC"
CARD_BG     = "#FFFFFF"
TXT         = "#0F172A"
TXT_MED     = "#475569"
TXT_LT      = "#94A3B8"
RULE        = "#E2E8F0"
GRID        = "#CBD5E1"

VERIFY_COLORS = {
    "Verified":     "#8B5CF6",
    "Not Verified": "#CBD5E1",
}
LANG_COLORS = {
    "Filipino (tl)": "#60A5FA",
    "English (en)":  "#34D399",
    "other":         "#E2E8F0",
}
REPLY_COLORS = {
    "Original": "#3B82F6",
    "Reply":    "#94A3B8",
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
# 🔧  Helpers
# ──────────────────────────────────────────────────────────────────

def _fmt_k(x, _=None):
    if x >= 1_000_000:
        v = x / 1_000_000
        return f"{v:.1f}M" if v != int(v) else f"{int(v)}M"
    if x >= 1_000:
        v = x / 1_000
        return f"{v:.1f}K" if v != int(v) else f"{int(v)}K"
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


def _normalize(df):
    for col in ("isReply", "author_isBlueVerified"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes")
                if pd.notna(v) else False
            ).astype(bool)
    if "lang" in df.columns:
        df["lang"] = df["lang"].replace({
            "tl": "Filipino (tl)",
            "en": "English (en)",
            "und": "other",
        })
    return df


def _nonzero_stats(series):
    """Return mean, median, count, and pct_nonzero for a numeric series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    nz = s[s > 0]
    return {
        "mean":       float(nz.mean())   if len(nz) else 0.0,
        "median":     float(nz.median()) if len(nz) else 0.0,
        "n_total":    int(len(s)),
        "n_nonzero":  int(len(nz)),
        "pct_nonzero": float(len(nz) / len(s) * 100) if len(s) else 0.0,
    }


def _style_ax(ax, grid_axis="y"):
    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom", "left"))
        if loc in ("bottom", "left"):
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    ax.tick_params(which="major", length=4, width=0.8, labelsize=9)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.xaxis.grid(False)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.9, zorder=0)
        ax.yaxis.grid(False)


def _safe_show(fig, save_path=None):
    if save_path:
        save_path = os.path.normpath(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Chart saved → {save_path}")
    is_gui = matplotlib.get_backend().lower() not in (
        "agg", "cairo", "pdf", "ps", "svg", "template"
    )
    if is_gui:
        plt.show()
    else:
        if not save_path:
            fig.savefig("bivariate_output.png", dpi=300, bbox_inches="tight")
            print("[viz] Non-interactive — saved → bivariate_output.png")
        plt.close(fig)


def _pill(ax, text, color, desc=""):
    ax.plot([0, 0], [0, 1], color=color, linewidth=3,
            transform=ax.transAxes, clip_on=False, zorder=10)
    ax.annotate(
        f"  {text}  ",
        xy=(0, 1), xycoords="axes fraction",
        xytext=(0, 46), textcoords="offset points",
        fontsize=7.5, fontweight="bold", color="white",
        va="bottom", ha="left", annotation_clip=False,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=color, edgecolor="none"),
    )
    if desc:
        ax.annotate(
            desc,
            xy=(0, 1), xycoords="axes fraction",
            xytext=(62, 46), textcoords="offset points",
            fontsize=8, color=TXT_MED, style="italic",
            va="bottom", ha="left", annotation_clip=False,
        )


# ──────────────────────────────────────────────────────────────────
# 🖊  Panel renderers
# ──────────────────────────────────────────────────────────────────

def _panel_grouped_bars(ax, df, group_col, group_order, metrics,
                        metric_labels, metric_colors,
                        title, ylabel, pill_text, pill_color, pill_desc,
                        colors_map):
    """
    Grouped vertical bar chart.
    For each group shows mean of non-zero values for each metric.
    Annotates bar top with mean value + sub-label with tweet count.
    """
    _pill(ax, pill_text, pill_color, pill_desc)

    # Compute stats
    data = {}
    counts = {}
    for g in group_order:
        mask = df[group_col] == g
        counts[g] = int(mask.sum())
        data[g] = {}
        for m in metrics:
            data[g][m] = _nonzero_stats(df.loc[mask, m])["mean"]

    x     = np.arange(len(group_order))
    width = 0.80 / len(metrics)
    max_v = max(
        (data[g][m] for g in group_order for m in metrics),
        default=1
    )
    max_v = max(max_v, 1)

    for i, (m, label, color) in enumerate(
            zip(metrics, metric_labels, metric_colors)):
        offset = (i - (len(metrics) - 1) / 2) * width
        vals   = [data[g][m] for g in group_order]
        bars   = ax.bar(x + offset, vals, width * 0.88,
                        color=color, alpha=0.90, label=label, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * 0.02,
                _fmt_stat(val),
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=TXT,
            )

    # Tweet count sub-labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{g}\n({counts[g]:,} tweets)" for g in group_order
    ], fontsize=8.5)

    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel, fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_ylim(0, max_v * 1.30)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_k))
    if len(metrics) > 1:
        ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE)
    _style_ax(ax, grid_axis="y")


def _panel_hbar(ax, df, group_col, group_order, metric,
                colors_map, title, xlabel,
                pill_text, pill_color, pill_desc, total):
    """
    Horizontal bar chart — mean of non-zero values per group.
    Sub-label shows tweet count and % of total.
    """
    _pill(ax, pill_text, pill_color, pill_desc)

    stats  = {}
    counts = {}
    for g in group_order:
        mask = df[group_col] == g
        counts[g] = int(mask.sum())
        stats[g]  = _nonzero_stats(df.loc[mask, metric])

    vals   = [stats[g]["mean"] for g in group_order]
    colors = [colors_map.get(g, "#94A3B8") for g in group_order]
    max_v  = max(max(vals), 1)

    ax.barh(range(len(group_order)), vals,
            color=colors, height=0.52, alpha=0.90, zorder=2)

    for i, (g, val) in enumerate(zip(group_order, vals)):
        n   = counts[g]
        pct = n / total * 100
        # Mean value — bold
        ax.text(val + max_v * 0.015, i,
                _fmt_stat(val),
                va="center", fontsize=9,
                fontweight="bold", color=TXT)
        # Sub-label — tweet count
        ax.text(val + max_v * 0.015, i - 0.28,
                f"{n:,} tweets  ({pct:.1f}% of dataset)",
                va="center", fontsize=7.5, color=TXT_MED)

    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order, fontsize=9.5)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, max_v * 1.70)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    _style_ax(ax, grid_axis="x")


def _panel_stacked_100(ax, df, group_col, group_order,
                       split_col, split_order, colors_map,
                       title, pill_text, pill_color, pill_desc):
    """
    100% stacked horizontal bars — composition of split_col within each group.
    """
    _pill(ax, pill_text, pill_color, pill_desc)

    rows = {}
    for g in group_order:
        sub  = df.loc[df[group_col] == g, split_col]
        pcts = sub.value_counts(normalize=True) * 100
        rows[g] = {s: float(pcts.get(s, 0.0)) for s in split_order}

    lefts = np.zeros(len(group_order))
    for s in split_order:
        vals  = np.array([rows[g][s] for g in group_order])
        color = colors_map.get(s, "#94A3B8")
        ax.barh(range(len(group_order)), vals, left=lefts,
                color=color, height=0.52, alpha=0.90,
                label=s, zorder=2)
        for i, (v, lft) in enumerate(zip(vals, lefts)):
            if v > 8:
                ax.text(lft + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order, fontsize=9.5)
    ax.set_title(title, pad=8)
    ax.set_xlabel("Share of Tweets (%)", fontsize=8.5,
                  color=TXT_MED, labelpad=5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE,
              loc="lower right")
    _style_ax(ax, grid_axis="x")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_bivariate_analysis(data_source, save_path=None):
    """
    Bivariate Analysis — relationships within the tweets dataset.

    Uses author metadata already embedded in each tweet row
    (author_isBlueVerified, lang, isReply) paired against engagement
    metrics (viewCount, retweetCount, likeCount).

    All engagement values computed as mean of non-zero tweets only,
    clearly labelled, to avoid zero-inflation bias.

    Layout (2 × 2):
      Panel 1 — Verification × Views & Retweets  (grouped bars)
      Panel 2 — Language × Mean Views            (horizontal bars)
      Panel 3 — Reply Status × Mean Views        (horizontal bars)
      Panel 4 — Language × Reply Composition     (100% stacked)

    Parameters
    ----------
    data_source : str or DataFrame   tweets dataset
    save_path   : str, optional
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize(df)
    N  = len(df)

    # Derived labels
    df["verify_label"] = df["author_isBlueVerified"].map(
        {False: "Not Verified", True: "Verified"})
    df["reply_label"]  = df["isReply"].map(
        {False: "Original", True: "Reply"})

    # Filter 'other' lang for cleaner comparisons (only 0.2%)
    df_lang = df[df["lang"].isin(["Filipino (tl)", "English (en)"])].copy()

    # Ordered groups
    verify_order = ["Not Verified", "Verified"]
    lang_order   = ["Filipino (tl)", "English (en)"]
    reply_order  = ["Original", "Reply"]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 18), facecolor=BG)

    # Header band
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.965), 1, 0.035, boxstyle="square,pad=0",
        facecolor=TXT, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))
    fig.text(
        0.5, 0.983,
        "Bivariate Analysis \u2014 Author & Content Traits \u00d7 Tweet Engagement",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="white", zorder=3,
    )

    # KPI strip
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.930), 1, 0.035, boxstyle="square,pad=0",
        facecolor="#1E293B", edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))

    # KPI values
    verified_pct   = df["author_isBlueVerified"].mean() * 100
    original_pct   = (~df["isReply"]).mean() * 100
    fil_pct        = (df["lang"] == "Filipino (tl)").mean() * 100

    kpis = [
        (f"{N:,}",             "tweets analysed"),
        (f"{verified_pct:.1f}%", "from verified authors"),
        (f"{original_pct:.1f}%", "original posts (not replies)"),
        (f"{fil_pct:.1f}%",    "tweets in Filipino"),
    ]
    for i, (val, lbl) in enumerate(kpis):
        xf = 0.07 + i * 0.23
        fig.text(xf, 0.954, val, ha="left", va="center",
                 fontsize=11, fontweight="bold", color="#60A5FA", zorder=3)
        fig.text(xf, 0.937, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_LT, zorder=3)
        if i > 0:
            fig.add_artist(mpatches.FancyBboxPatch(
                (xf - 0.018, 0.933), 0.0015, 0.026,
                boxstyle="square,pad=0", facecolor="#334155", edgecolor="none",
                transform=fig.transFigure, clip_on=False, zorder=3))

    # GridSpec 2 × 2
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.12, right=0.97,
        top=0.908, bottom=0.06,
        hspace=0.52, wspace=0.36,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1 — Verification × Views & Retweets ─────────────────────────────
    _panel_grouped_bars(
        ax1, df,
        group_col     = "verify_label",
        group_order   = verify_order,
        metrics       = ["viewCount", "retweetCount"],
        metric_labels = ["Views", "Retweets"],
        metric_colors = ["#8B5CF6", "#A78BFA"],
        title         = "Verification \u00d7 Mean Views & Retweets",
        ylabel        = "Mean Count (non-zero tweets only)",
        pill_text     = "VERIFICATION",
        pill_color    = "#8B5CF6",
        pill_desc     = "Does Blue Verified status translate to higher reach?",
        colors_map    = VERIFY_COLORS,
    )

    # ── Panel 2 — Language × Mean Views ───────────────────────────────────────
    _panel_hbar(
        ax2, df_lang,
        group_col   = "lang",
        group_order = lang_order,
        metric      = "viewCount",
        colors_map  = LANG_COLORS,
        title       = "Language \u00d7 Mean Views",
        xlabel      = "Mean View Count (non-zero tweets only)",
        pill_text   = "LANGUAGE",
        pill_color  = LANG_COLORS["Filipino (tl)"],
        pill_desc   = "Do Filipino or English tweets reach wider audiences?",
        total       = N,
    )

    # ── Panel 3 — Reply Status × Mean Views ───────────────────────────────────
    _panel_hbar(
        ax3, df,
        group_col   = "reply_label",
        group_order = reply_order,
        metric      = "viewCount",
        colors_map  = REPLY_COLORS,
        title       = "Reply Status \u00d7 Mean Views",
        xlabel      = "Mean View Count (non-zero tweets only)",
        pill_text   = "REPLY STATUS",
        pill_color  = REPLY_COLORS["Original"],
        pill_desc   = "Do original posts reach wider audiences than replies?",
        total       = N,
    )

    # ── Panel 4 — Language × Reply Composition ────────────────────────────────
    _panel_stacked_100(
        ax4, df_lang,
        group_col    = "lang",
        group_order  = lang_order,
        split_col    = "reply_label",
        split_order  = reply_order,
        colors_map   = REPLY_COLORS,
        title        = "Language \u00d7 Reply Composition",
        pill_text    = "LANGUAGE \u00d7 BEHAVIOUR",
        pill_color   = LANG_COLORS["English (en)"],
        pill_desc    = "Are Filipino tweets more conversational than English?",
    )

    # Footer
    fig.text(
        0.5, 0.025,
        "All engagement values are means computed on non-zero tweets only — "
        "zero-value tweets are excluded to avoid inflation bias from silent impressions.  "
        "Language \u2018other\u2019 / undetermined (0.2%) excluded from language comparisons.",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
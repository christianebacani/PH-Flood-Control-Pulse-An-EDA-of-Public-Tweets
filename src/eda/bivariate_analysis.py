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

BG      = "#F8FAFC"
CARD_BG = "#FFFFFF"
TXT     = "#0F172A"
TXT_MED = "#475569"
TXT_LT  = "#94A3B8"
RULE    = "#E2E8F0"
GRID    = "#CBD5E1"

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


def _nonzero_mean(series):
    s  = pd.to_numeric(series, errors="coerce").dropna()
    nz = s[s > 0]
    return float(nz.mean()) if len(nz) else 0.0


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
    """Left accent bar + pill badge sitting just above the axes top."""
    ax.plot([0, 0], [0, 1], color=color, linewidth=3,
            transform=ax.transAxes, clip_on=False, zorder=10)
    ax.annotate(
        f"  {text}  ",
        xy=(0, 1), xycoords="axes fraction",
        xytext=(0, 6), textcoords="offset points",
        fontsize=7.5, fontweight="bold", color="white",
        va="bottom", ha="left", annotation_clip=False,
        bbox=dict(boxstyle="round,pad=0.28", facecolor=color, edgecolor="none"),
    )
    if desc:
        ax.annotate(
            desc,
            xy=(0, 1), xycoords="axes fraction",
            xytext=(62, 6), textcoords="offset points",
            fontsize=8, color=TXT_MED, style="italic",
            va="bottom", ha="left", annotation_clip=False,
        )


def _hbar(ax, group_order, vals, counts, colors, N,
          title, xlabel, pill_text, pill_color, pill_desc=""):
    """
    Reusable horizontal bar — mean value per group.
    Annotates each bar with the mean and a tweet-count sub-label.
    """
    _pill(ax, pill_text, pill_color, pill_desc)
    max_v = max(max(vals), 1)

    ax.barh(range(len(group_order)), vals,
            color=colors, height=0.52, alpha=0.90, zorder=2)

    for i, (g, val) in enumerate(zip(group_order, vals)):
        ax.text(val + max_v * 0.015, i,
                _fmt_stat(val),
                va="center", fontsize=9, fontweight="bold", color=TXT)
        ax.text(val + max_v * 0.015, i - 0.30,
                f"{counts[g]:,} tweets  ({counts[g] / N * 100:.1f}% of dataset)",
                va="center", fontsize=7.5, color=TXT_MED)

    ax.set_yticks(range(len(group_order)))
    ax.set_yticklabels(group_order, fontsize=9.5)
    ax.set_title(title, pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, max_v * 1.72)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    _style_ax(ax, grid_axis="x")


def _stacked_100(ax, df, group_col, group_order,
                 split_col, split_order, colors_map,
                 title, pill_text, pill_color, pill_desc=""):
    """100% stacked horizontal bars."""
    _pill(ax, pill_text, pill_color, pill_desc)

    rows = {}
    for g in group_order:
        sub  = df.loc[df[group_col] == g, split_col]
        pcts = sub.value_counts(normalize=True) * 100
        rows[g] = {s: float(pcts.get(s, 0.0)) for s in split_order}

    lefts = np.zeros(len(group_order))
    for s in split_order:
        vals  = np.array([rows[g][s] for g in group_order])
        ax.barh(range(len(group_order)), vals, left=lefts,
                color=colors_map[s], height=0.52, alpha=0.90,
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
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE, loc="lower right")
    _style_ax(ax, grid_axis="x")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_bivariate_analysis(data_source, save_path=None):
    """
    Bivariate Analysis — relationships within the tweets dataset.

    Uses author metadata already embedded in each tweet row
    (author_isBlueVerified, lang, isReply) paired against engagement
    metrics (viewCount, retweetCount).

    All engagement values are means of non-zero tweets only to avoid
    zero-inflation bias — clearly labelled in axis titles and footer.

    Layout (2 × 2):
      Panel 1 (top-left)  — Verification × Mean Views
      Panel 2 (top-right) — Verification × Mean Retweets
      Panel 3 (bot-left)  — Language × Mean Views
      Panel 4 (bot-right) — Reply Status × Mean Views
                            + Language × Reply Composition (inset annotation)

    Parameters
    ----------
    data_source : str or DataFrame   tweets CSV path or DataFrame
    save_path   : str, optional      output PNG path
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

    df_lang = df[df["lang"].isin(["Filipino (tl)", "English (en)"])].copy()

    verify_order = ["Not Verified", "Verified"]
    lang_order   = ["Filipino (tl)", "English (en)"]
    reply_order  = ["Original", "Reply"]

    # Pre-compute counts
    verify_counts = {g: int((df["verify_label"] == g).sum()) for g in verify_order}
    lang_counts   = {g: int((df_lang["lang"] == g).sum())    for g in lang_order}
    reply_counts  = {g: int((df["reply_label"] == g).sum())  for g in reply_order}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 17), facecolor=BG)

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

    kpis = [
        (f"{N:,}",
         "tweets analysed"),
        (f"{verify_counts['Verified']:,}",
         f"from verified authors ({verify_counts['Verified']/N*100:.1f}%)"),
        (f"{reply_counts['Original']:,}",
         f"original posts ({reply_counts['Original']/N*100:.1f}%)"),
        (f"{lang_counts['Filipino (tl)']:,}",
         f"tweets in Filipino ({lang_counts['Filipino (tl)']/N*100:.1f}%)"),
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
        top=0.912, bottom=0.06,
        hspace=0.38, wspace=0.36,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # ── Panel 1 — Verification × Mean Views ───────────────────────────────────
    _hbar(
        ax1,
        group_order = verify_order,
        vals        = [_nonzero_mean(df.loc[df["verify_label"] == g, "viewCount"])
                       for g in verify_order],
        counts      = verify_counts,
        colors      = [VERIFY_COLORS[g] for g in verify_order],
        N           = N,
        title       = "Verification \u00d7 Mean Views",
        xlabel      = "Mean View Count (non-zero tweets only)",
        pill_text   = "VERIFICATION",
        pill_color  = "#8B5CF6",
        pill_desc   = "Does Blue Verified status translate to higher reach?",
    )

    # ── Panel 2 — Verification × Mean Retweets ────────────────────────────────
    _hbar(
        ax2,
        group_order = verify_order,
        vals        = [_nonzero_mean(df.loc[df["verify_label"] == g, "retweetCount"])
                       for g in verify_order],
        counts      = verify_counts,
        colors      = [VERIFY_COLORS[g] for g in verify_order],
        N           = N,
        title       = "Verification \u00d7 Mean Retweets",
        xlabel      = "Mean Retweet Count (non-zero tweets only)",
        pill_text   = "VERIFICATION",
        pill_color  = "#8B5CF6",
        pill_desc   = "Are verified authors more likely to be retweeted?",
    )

    # ── Panel 3 — Language × Mean Views ───────────────────────────────────────
    _hbar(
        ax3,
        group_order = lang_order,
        vals        = [_nonzero_mean(df_lang.loc[df_lang["lang"] == g, "viewCount"])
                       for g in lang_order],
        counts      = lang_counts,
        colors      = [LANG_COLORS[g] for g in lang_order],
        N           = N,
        title       = "Language \u00d7 Mean Views",
        xlabel      = "Mean View Count (non-zero tweets only)",
        pill_text   = "LANGUAGE",
        pill_color  = LANG_COLORS["Filipino (tl)"],
        pill_desc   = "Do Filipino or English tweets reach wider audiences?",
    )

    # ── Panel 4 — Reply Status × Mean Views ───────────────────────────────────
    _hbar(
        ax4,
        group_order = reply_order,
        vals        = [_nonzero_mean(df.loc[df["reply_label"] == g, "viewCount"])
                       for g in reply_order],
        counts      = reply_counts,
        colors      = [REPLY_COLORS[g] for g in reply_order],
        N           = N,
        title       = "Reply Status \u00d7 Mean Views",
        xlabel      = "Mean View Count (non-zero tweets only)",
        pill_text   = "REPLY STATUS",
        pill_color  = REPLY_COLORS["Original"],
        pill_desc   = "Do original posts reach wider audiences than replies?",
    )

    # Footer
    fig.text(
        0.5, 0.022,
        "All engagement values are means computed on non-zero tweets only \u2014 "
        "zero-value tweets excluded to avoid inflation bias from silent impressions.  "
        "Language \u2018other\u2019 / undetermined (0.2%) excluded from language comparisons.",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
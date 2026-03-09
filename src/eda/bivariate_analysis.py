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
MEDIAN_LINE = "#1E293B"

TIER_COLORS = {
    "Micro  (< 10K)":    "#BFDBFE",
    "Mid    (10K–100K)": "#60A5FA",
    "Macro  (100K–1M)":  "#2563EB",
    "Mega   (1M+)":      "#1E3A8A",
}
TIER_ORDER = [
    "Micro  (< 10K)",
    "Mid    (10K–100K)",
    "Macro  (100K–1M)",
    "Mega   (1M+)",
]

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


def _normalize_tweets(df):
    for col in ("isReply", "author_isBlueVerified"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes")
                if pd.notna(v) else False
            ).astype(bool)
    if "createdAt" in df.columns:
        df["createdAt"] = pd.to_datetime(
            df["createdAt"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    if "lang" in df.columns:
        df["lang"] = df["lang"].replace({
            "tl": "Filipino (tl)", "en": "English (en)", "und": "other",
        })
    return df


def _normalize_authors(df):
    if "author_isBlueVerified" in df.columns:
        df["author_isBlueVerified"] = df["author_isBlueVerified"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
            if pd.notna(v) else False
        ).astype(bool)
    if "obfuscated_userName" in df.columns:
        df["obfuscated_userName"] = (
            df["obfuscated_userName"]
            .astype(str)
            .str.replace("@", "", regex=False)
            .str.strip()
        )
    return df


def _assign_tier(followers):
    if followers >= 1_000_000:
        return "Mega   (1M+)"
    elif followers >= 100_000:
        return "Macro  (100K–1M)"
    elif followers >= 10_000:
        return "Mid    (10K–100K)"
    else:
        return "Micro  (< 10K)"


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

def _panel_tier_vs_views(ax, df):
    _pill(ax, "FOLLOWER TIER × REACH",
          TIER_COLORS["Mega   (1M+)"],
          "Do tweets from larger accounts reach wider audiences?")

    medians, means, counts = [], [], []
    for t in TIER_ORDER:
        sub = df.loc[df["follower_tier"] == t, "viewCount"].dropna()
        sub = sub[sub > 0]
        medians.append(float(sub.median()) if len(sub) else 0.0)
        means.append(float(sub.mean())   if len(sub) else 0.0)
        counts.append(int(df["follower_tier"].eq(t).sum()))

    colors = [TIER_COLORS[t] for t in TIER_ORDER]
    max_v  = max(max(medians), 1)

    ax.barh(range(len(TIER_ORDER)), medians, color=colors,
            height=0.55, alpha=0.92, zorder=2)

    for i, (med, mn, n) in enumerate(zip(medians, means, counts)):
        ax.text(med + max_v * 0.015, i,
                _fmt_stat(med),
                va="center", fontsize=9, fontweight="bold", color=TXT)
        ax.text(med + max_v * 0.015, i - 0.28,
                f"mean {_fmt_stat(mn)}  ·  {n:,} tweets",
                va="center", fontsize=7.5, color=TXT_MED)

    ax.set_yticks(range(len(TIER_ORDER)))
    ax.set_yticklabels(TIER_ORDER, fontsize=9)
    ax.set_title("Follower Tier vs Median Views", pad=8)
    ax.set_xlabel("Median View Count (non-zero tweets only)",
                  fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, max_v * 1.70)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    _style_ax(ax, grid_axis="x")


def _panel_verification_vs_engagement(ax, df):
    _pill(ax, "VERIFICATION × ENGAGEMENT",
          "#8B5CF6",
          "Does Blue Verified status translate to higher reach?")

    metrics       = ["viewCount", "retweetCount"]
    metric_names  = ["Views", "Retweets"]
    metric_colors = ["#8B5CF6", "#A78BFA"]
    groups        = ["Not Verified", "Verified"]

    data = {}
    for g in groups:
        mask = df["verify_label"] == g
        data[g] = {}
        for m in metrics:
            sub = df.loc[mask, m].dropna()
            sub = sub[sub > 0]
            data[g][m] = float(sub.median()) if len(sub) else 0.0

    x     = np.arange(len(groups))
    width = 0.32
    max_all = max(
        (data[g][m] for g in groups for m in metrics), default=1
    )

    for i, (m, name, color) in enumerate(zip(metrics, metric_names, metric_colors)):
        vals   = [data[g][m] for g in groups]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        color=color, alpha=0.90, label=name, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_all * 0.02,
                    _fmt_stat(val),
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold", color=TXT)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9.5)
    ax.set_title("Verification vs Median Views & Retweets", pad=8)
    ax.set_ylabel("Median Count (non-zero tweets only)",
                  fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_ylim(0, max(max_all * 1.30, 1))
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE)
    _style_ax(ax, grid_axis="y")


def _panel_tier_vs_language(ax, df):
    _pill(ax, "FOLLOWER TIER × LANGUAGE",
          LANG_COLORS["Filipino (tl)"],
          "Do larger accounts tweet in Filipino or English?")

    lang_order = ["Filipino (tl)", "English (en)", "other"]
    lang_short = {"Filipino (tl)": "Filipino", "English (en)": "English", "other": "Other"}

    rows  = {}
    for t in TIER_ORDER:
        sub = df.loc[df["follower_tier"] == t, "lang"] \
                .value_counts(normalize=True) * 100
        rows[t] = {lang: float(sub.get(lang, 0.0)) for lang in lang_order}

    lefts = np.zeros(len(TIER_ORDER))
    for lang in lang_order:
        vals  = np.array([rows[t][lang] for t in TIER_ORDER])
        ax.barh(range(len(TIER_ORDER)), vals, left=lefts,
                color=LANG_COLORS[lang], height=0.55, alpha=0.90,
                label=lang_short[lang], zorder=2)
        for i, (v, lft) in enumerate(zip(vals, lefts)):
            if v > 8:
                ax.text(lft + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(len(TIER_ORDER)))
    ax.set_yticklabels(TIER_ORDER, fontsize=9)
    ax.set_title("Follower Tier × Language Mix", pad=8)
    ax.set_xlabel("Share of Tweets (%)", fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE, loc="lower right")
    _style_ax(ax, grid_axis="x")


def _panel_tier_vs_reply(ax, df):
    _pill(ax, "FOLLOWER TIER × BEHAVIOUR",
          REPLY_COLORS["Original"],
          "Do larger accounts broadcast or engage in conversation?")

    reply_order = ["Original", "Reply"]
    rows = {}
    for t in TIER_ORDER:
        sub = df.loc[df["follower_tier"] == t, "reply_label"] \
                .value_counts(normalize=True) * 100
        rows[t] = {r: float(sub.get(r, 0.0)) for r in reply_order}

    lefts = np.zeros(len(TIER_ORDER))
    for reply in reply_order:
        vals  = np.array([rows[t][reply] for t in TIER_ORDER])
        ax.barh(range(len(TIER_ORDER)), vals, left=lefts,
                color=REPLY_COLORS[reply], height=0.55, alpha=0.90,
                label=reply, zorder=2)
        for i, (v, lft) in enumerate(zip(vals, lefts)):
            if v > 8:
                ax.text(lft + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(len(TIER_ORDER)))
    ax.set_yticklabels(TIER_ORDER, fontsize=9)
    ax.set_title("Follower Tier × Tweet Behaviour", pad=8)
    ax.set_xlabel("Share of Tweets (%)", fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE, loc="lower right")
    _style_ax(ax, grid_axis="x")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_bivariate_analysis(tweets_source, authors_source, save_path=None):
    """
    Bivariate Analysis — author traits × tweet engagement.

    Joins tweets to authors on the shared pseudonymised author ID,
    then examines four cross-dataset variable pairs:

      Panel 1 — Follower Tier vs Median Views
      Panel 2 — Verification vs Median Views & Retweets
      Panel 3 — Follower Tier × Language Mix
      Panel 4 — Follower Tier × Tweet Behaviour

    Parameters
    ----------
    tweets_source  : str or DataFrame
    authors_source : str or DataFrame
    save_path      : str, optional
    """
    tweets  = pd.read_csv(tweets_source)  if isinstance(tweets_source,  str) \
              else tweets_source.copy()
    authors = pd.read_csv(authors_source) if isinstance(authors_source, str) \
              else authors_source.copy()

    tweets  = _normalize_tweets(tweets)
    authors = _normalize_authors(authors)

    # Join on pseudonymised author ID
    tweets["_jk"]  = tweets["pseudo_author_userName"].astype(str).str.strip()
    authors["_jk"] = authors["obfuscated_userName"].astype(str).str.strip()

    merged = tweets.merge(
        authors[["_jk", "author_followers",
                 "author_following", "author_isBlueVerified"]],
        on="_jk", how="left",
        suffixes=("", "_auth"),
    )

    # Resolve potential duplicate bool column from both datasets
    if "author_isBlueVerified_auth" in merged.columns:
        merged["author_isBlueVerified"] = merged["author_isBlueVerified_auth"]
        merged.drop(columns=["author_isBlueVerified_auth"], inplace=True)

    N         = len(merged)
    n_matched = int(merged["author_followers"].notna().sum())

    # Derived columns
    merged["follower_tier"] = merged["author_followers"].apply(
        lambda x: _assign_tier(x) if pd.notna(x) else "Unknown"
    )
    merged["reply_label"]  = merged["isReply"].map(
        {False: "Original", True: "Reply"})
    merged["verify_label"] = merged["author_isBlueVerified"].map(
        {False: "Not Verified", True: "Verified"})

    df = merged[merged["follower_tier"] != "Unknown"].copy()

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 18), facecolor=BG)

    # Header
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.965), 1, 0.035, boxstyle="square,pad=0",
        facecolor=TXT, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))
    fig.text(
        0.5, 0.983,
        "Bivariate Analysis \u2014 Author Traits \u00d7 Tweet Engagement",
        ha="center", va="center",
        fontsize=16, fontweight="bold", color="white", zorder=3,
    )

    # KPI strip
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.930), 1, 0.035, boxstyle="square,pad=0",
        facecolor="#1E293B", edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))

    kpis = [
        (f"{N:,}",         "tweets in merged dataset"),
        (f"{n_matched:,}", "tweets matched to an author"),
        (f"{len(df):,}",   "tweets with follower data"),
        ("4",              "cross-dataset pairs examined"),
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

    # GridSpec
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.14, right=0.97,
        top=0.908, bottom=0.06,
        hspace=0.52, wspace=0.38,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    _panel_tier_vs_views(ax1, df)
    _panel_verification_vs_engagement(ax2, df)
    _panel_tier_vs_language(ax3, df)
    _panel_tier_vs_reply(ax4, df)

    # Footer
    fig.text(
        0.5, 0.025,
        "Tweets joined to authors on pseudonymised author ID (left join).  "
        "Engagement medians computed on non-zero values only to avoid zero-inflation bias.  "
        "Tiers: Micro < 10K \u00b7 Mid 10K\u2013100K \u00b7 Macro 100K\u20131M \u00b7 Mega 1M+.",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
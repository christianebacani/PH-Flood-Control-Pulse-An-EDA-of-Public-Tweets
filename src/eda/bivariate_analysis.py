import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# 🎨  Design tokens  (matches the rest of the EDA suite)
# ──────────────────────────────────────────────────────────────────

BG      = "#F8FAFC"
CARD_BG = "#FFFFFF"
TXT     = "#0F172A"
TXT_MED = "#475569"
TXT_LT  = "#94A3B8"
RULE    = "#E2E8F0"
GRID    = "#CBD5E1"

C_FILIPINO   = "#60A5FA"
C_ENGLISH    = "#34D399"
C_ORIGINAL   = "#3B82F6"
C_REPLY      = "#94A3B8"
C_VERIFIED   = "#8B5CF6"
C_UNVERIFIED = "#CBD5E1"

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


def _normalize(df):
    for col in ("isReply", "author_isBlueVerified"):
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in ("true", "1", "yes")
                if pd.notna(v) else False
            ).astype(bool)
    if "createdAt" in df.columns:
        df["createdAt"] = pd.to_datetime(df["createdAt"], errors="coerce", utc=True)
    if "lang" in df.columns:
        df["lang"] = df["lang"].replace({
            "tl":  "Filipino (tl)",
            "en":  "English (en)",
            "und": "other",
        })
    return df


def _style_ax(ax, grid_axis="y"):
    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom", "left"))
        if loc in ("bottom", "left"):
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    ax.tick_params(which="major", length=4, width=0.8, labelsize=9)
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
    elif grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.yaxis.grid(False)
    else:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)


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


def _pill_badge(ax, text, color, desc=""):
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


def _median_bars(ax, df, group_col, metric, colors, title, xlabel,
                 total=None):
    N = total if total is not None else len(df)
    grp_counts = df[group_col].value_counts()
    data = (
        df.groupby(group_col)[metric]
        .median()
        .sort_values(ascending=False)
    )
    groups = list(data.index)
    vals   = [float(v) for v in data.values]
    cols   = [colors.get(g, "#94A3B8") for g in groups]
    max_v  = max(max(vals) if vals else 1, 1)  # floor at 1 — prevents xlim(0, 0)

    ax.barh(range(len(groups)), vals, color=cols, height=0.52, alpha=0.90, zorder=2)
    for i, (g, v) in enumerate(zip(groups, vals)):
        pct = grp_counts.get(g, 0) / N * 100
        ax.text(v + max_v * 0.015, i,
                f"{_fmt_k(v)}  ({pct:.1f}% of tweets)",
                va="center", fontsize=8.5, color=TXT_MED)

    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TXT, pad=8)
    ax.set_xlabel(xlabel, fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, max_v * 1.65)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    _style_ax(ax, grid_axis="x")


def _stacked_100(ax, df, group_col, split_col, split_order, colors, title):
    groups = [g for g in ["Filipino (tl)", "English (en)"]
              if g in df[group_col].dropna().unique()]
    if not groups:
        groups = list(df[group_col].dropna().unique())

    rows = {}
    for g in groups:
        sub = df.loc[df[group_col] == g, split_col].value_counts(normalize=True) * 100
        rows[g] = {s: float(sub.get(s, 0.0)) for s in split_order}

    sorted_g = sorted(groups, key=lambda g: rows[g].get(split_order[0], 0), reverse=True)
    lefts = np.zeros(len(sorted_g))

    for s in split_order:
        vals = np.array([rows[g][s] for g in sorted_g])
        ax.barh(range(len(sorted_g)), vals, left=lefts,
                color=colors[s], height=0.52, alpha=0.90, label=s, zorder=2)
        for i, (v, lft) in enumerate(zip(vals, lefts)):
            if v > 9:
                ax.text(lft + v / 2, i, f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(len(sorted_g)))
    ax.set_yticklabels(sorted_g, fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TXT, pad=8)
    ax.set_xlabel("Share of Tweets (%)", fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE, loc="lower right")
    _style_ax(ax, grid_axis="x")


def _grouped_lang_verify(ax, df, title):
    lang_order   = ["Filipino (tl)", "English (en)"]
    verify_order = ["Not Verified", "Verified"]
    verify_colors = {"Not Verified": C_UNVERIFIED, "Verified": C_VERIFIED}

    data = {}
    for v in verify_order:
        sub = df.loc[df["verify_label"] == v, "lang"]
        tot = len(sub)
        data[v] = {lang: sub.value_counts().get(lang, 0) / tot * 100
                   for lang in lang_order}

    x     = np.arange(len(lang_order))
    width = 0.35

    for i, v in enumerate(verify_order):
        vals   = [data[v][lang] for lang in lang_order]
        offset = (i - 0.5) * width
        bars   = ax.bar(x + offset, vals, width,
                        color=verify_colors[v], alpha=0.88,
                        label=v, zorder=2)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.8,
                    f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=8.5, color=TXT_MED)

    ax.set_xticks(x)
    ax.set_xticklabels(lang_order, fontsize=9.5)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TXT, pad=8)
    ax.set_ylabel("% of tweets (within group)", fontsize=8.5, color=TXT_MED)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE)
    _style_ax(ax, grid_axis="y")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_bivariate_analysis(data_source, save_path=None):
    """
    Bivariate Analysis chart — 6 variable-pair panels, 3 rows × 2 cols.

    Row 1 — Language vs Median Likes  |  Language vs Median Retweets
    Row 2 — Reply Status vs Median Views  |  Language × Reply Composition
    Row 3 — Verification vs Median Likes  |  Verification × Language Mix
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize(df)
    N  = len(df)

    df["reply_label"]  = df["isReply"].map({False: "Original", True: "Reply"})
    df["verify_label"] = df["author_isBlueVerified"].map(
        {False: "Not Verified", True: "Verified"})

    df_lang = df[df["lang"].isin(["Filipino (tl)", "English (en)"])].copy()

    lang_colors  = {"Filipino (tl)": C_FILIPINO, "English (en)": C_ENGLISH}
    reply_colors = {"Original": C_ORIGINAL, "Reply": C_REPLY}
    verify_colors= {"Verified": C_VERIFIED, "Not Verified": C_UNVERIFIED}

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 21), facecolor=BG)

    # Header band
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.968), 1, 0.032, boxstyle="square,pad=0",
        facecolor=TXT, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))
    fig.text(0.5, 0.984,
             "Bivariate Analysis — Relationships Between Tweet Variables",
             ha="center", va="center",
             fontsize=16, fontweight="bold", color="white", zorder=3)

    # KPI strip
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.934), 1, 0.034, boxstyle="square,pad=0",
        facecolor="#1E293B", edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))

    kpis = [
        (f"{N:,}",  "tweets analysed"),
        ("3",       "categorical variables"),
        ("3",       "engagement metrics"),
        ("6",       "variable pairs examined"),
    ]
    for i, (val, lbl) in enumerate(kpis):
        xf = 0.07 + i * 0.23
        fig.text(xf, 0.957, val, ha="left", va="center",
                 fontsize=11, fontweight="bold", color="#60A5FA", zorder=3)
        fig.text(xf, 0.940, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_LT, zorder=3)
        if i > 0:
            fig.add_artist(mpatches.FancyBboxPatch(
                (xf - 0.018, 0.936), 0.0015, 0.026,
                boxstyle="square,pad=0", facecolor="#334155", edgecolor="none",
                transform=fig.transFigure, clip_on=False, zorder=3))

    # ── GridSpec ─────────────────────────────────────────────────────────────
    gs = gridspec.GridSpec(
        3, 2, figure=fig,
        left=0.14, right=0.97,
        top=0.912, bottom=0.055,
        hspace=0.54, wspace=0.38,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # ── Row 1: Language × engagement ─────────────────────────────────────────
    _pill_badge(ax1, "LANGUAGE", C_FILIPINO,
                "Does tweet language relate to engagement?")
    _median_bars(ax1, df_lang, "lang", "likeCount", lang_colors,
                 "Language vs Median Likes", "Median Like Count", total=N)

    _pill_badge(ax2, "LANGUAGE", C_FILIPINO)
    _median_bars(ax2, df_lang, "lang", "retweetCount", lang_colors,
                 "Language vs Median Retweets", "Median Retweet Count", total=N)

    # ── Row 2: Reply status × engagement + composition ────────────────────────
    _pill_badge(ax3, "REPLY STATUS", C_ORIGINAL,
                "Do original posts reach wider audiences than replies?")
    _median_bars(ax3, df, "reply_label", "viewCount", reply_colors,
                 "Reply Status vs Median Views", "Median View Count", total=N)

    _pill_badge(ax4, "REPLY STATUS", C_ORIGINAL)
    _stacked_100(ax4, df_lang, "lang", "reply_label",
                 split_order=["Original", "Reply"],
                 colors={"Original": C_ORIGINAL, "Reply": C_REPLY},
                 title="Language × Reply Composition")

    # ── Row 3: Verification × engagement + language mix ───────────────────────
    _pill_badge(ax5, "VERIFICATION", C_VERIFIED,
                "Does author verification predict higher engagement?")
    _median_bars(ax5, df, "verify_label", "likeCount", verify_colors,
                 "Verification vs Median Likes", "Median Like Count", total=N)

    _pill_badge(ax6, "VERIFICATION", C_VERIFIED)
    _grouped_lang_verify(ax6, df_lang, "Verification × Language Mix")

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.022,
        "All engagement comparisons use median (not mean) to reduce the distorting influence of viral outliers.  "
        "Language 'other' / undetermined (0.2%) excluded from language comparisons.",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
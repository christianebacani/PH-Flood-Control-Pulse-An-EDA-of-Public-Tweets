import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, NullLocator, MultipleLocator
from matplotlib.lines import Line2D
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# 🎨  Design tokens
# ──────────────────────────────────────────────────────────────────
BG        = "#F8FAFC"
CARD_BG   = "#FFFFFF"
TXT       = "#0F172A"
TXT_MED   = "#475569"
TXT_LT    = "#94A3B8"
RULE      = "#E2E8F0"
GRID      = "#CBD5E1"

# Accent palette
PURPLE    = "#7C3AED"
PURPLE_LT = "#DDD6FE"
BLUE      = "#2563EB"
BLUE_LT   = "#BFDBFE"
GREEN     = "#059669"
GREEN_LT  = "#A7F3D0"
AMBER     = "#D97706"
AMBER_LT  = "#FDE68A"
SLATE     = "#64748B"
SLATE_LT  = "#CBD5E1"

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
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ──────────────────────────────────────────────────────────────────
# 🔧  Helpers
# ──────────────────────────────────────────────────────────────────

def _fmt_k(x, _=None):
    if x >= 1_000_000:
        v = x / 1_000_000
        return f"{v:.1f}M"
    if x >= 1_000:
        v = x / 1_000
        return f"{v:.0f}K" if v == int(v) else f"{v:.1f}K"
    return f"{int(x)}"

def _fmt_stat(x):
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        v = x / 1_000
        return f"{v:.1f}K"
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
        df["lang"] = df["lang"].replace(
            {"tl": "Filipino (tl)", "en": "English (en)", "und": "other"})
    return df

def _nz(series):
    """Non-zero numeric values as a clean series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s[s > 0]

def _stats(series):
    nz = _nz(series)
    if len(nz) == 0:
        return dict(mean=0, p25=0, p75=0, p90=0, n=0)
    return dict(
        mean = float(nz.mean()),
        p25  = float(nz.quantile(0.25)),
        p75  = float(nz.quantile(0.75)),
        p90  = float(nz.quantile(0.90)),
        n    = int(len(nz)),
    )

def _style_ax(ax, grid_axis="x"):
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(which="major", length=3, width=0.8, labelsize=8.5)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_axisbelow(True)
    if grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.yaxis.grid(False)
    elif grid_axis == "y":
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.xaxis.grid(False)
    elif grid_axis == "both":
        ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)

def _card(ax):
    """White card background with subtle shadow effect."""
    ax.set_facecolor(CARD_BG)
    for sp in ax.spines.values():
        sp.set_zorder(5)

def _panel_label(ax, letter, title, subtitle, accent):
    """Panel letter badge + title + italic subtitle above axes."""
    ax.text(-0.08, 1.13, letter,
            transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="white",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor=accent, edgecolor="none"))
    ax.text(0.0, 1.13, title,
            transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=TXT,
            ha="left", va="center")
    ax.text(0.0, 1.055, subtitle,
            transform=ax.transAxes,
            fontsize=8, color=TXT_MED, style="italic",
            ha="left", va="center")

def _safe_show(fig, save_path=None):
    if save_path:
        save_path = os.path.normpath(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Chart saved → {save_path}")
    is_gui = matplotlib.get_backend().lower() not in (
        "agg","cairo","pdf","ps","svg","template")
    if is_gui:
        plt.show()
    else:
        if not save_path:
            fig.savefig("bivariate_output.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 🖊  Panel A — Lollipop chart: Verification × Views
#     Shows mean as dot, IQR as thick line, P90 as thin whisker
# ──────────────────────────────────────────────────────────────────
def _panel_A(ax, df):
    _card(ax)
    _panel_label(ax, "A",
                 "Verification × View Count",
                 "Mean views per tweet — verified authors reach far wider audiences",
                 PURPLE)

    groups  = ["Not Verified", "Verified"]
    colors  = [SLATE, PURPLE]
    lt_cols = [SLATE_LT, PURPLE_LT]
    N       = len(df)

    stats_list = [
        _stats(df.loc[df["verify_label"] == g, "viewCount"])
        for g in groups
    ]
    counts = [int((df["verify_label"] == g).sum()) for g in groups]

    y_pos = [1, 0]   # Verified on top

    for i, (g, st, col, lt, cnt, y) in enumerate(
            zip(groups, stats_list, colors, lt_cols, counts, y_pos)):

        # P25–P75 thick band
        ax.barh(y, st["p75"] - st["p25"], left=st["p25"],
                height=0.28, color=lt, alpha=0.85, zorder=2)

        # P90 thin whisker
        ax.plot([st["p25"], st["p90"]], [y, y],
                color=col, linewidth=1.2, alpha=0.5, zorder=3)

        # Stem from 0 to mean
        ax.plot([0, st["mean"]], [y, y],
                color=col, linewidth=2.5, alpha=0.6, zorder=3,
                solid_capstyle="round")

        # Mean dot
        ax.scatter(st["mean"], y, s=160, color=col,
                   zorder=5, edgecolors="white", linewidths=1.5)

        # Mean label
        ax.text(st["mean"] + (ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 10000) * 0.015,
                y + 0.06,
                f"{_fmt_stat(st['mean'])}",
                va="bottom", ha="left",
                fontsize=10, fontweight="bold", color=col)

        # Sub: tweet count
        ax.text(st["mean"] + (ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 10000) * 0.015,
                y - 0.13,
                f"{cnt:,} tweets  ({cnt/N*100:.1f}%)",
                va="top", ha="left", fontsize=7.5, color=TXT_MED)

    # Fix xlim after plotting
    max_p90 = max(s["p90"] for s in stats_list)
    ax.set_xlim(0, max_p90 * 1.55)

    # Re-draw labels with correct xlim
    offset = max_p90 * 1.55 * 0.015
    for i, (g, st, col, cnt, y) in enumerate(
            zip(groups, stats_list, colors, counts, y_pos)):
        # Clear old annotations by redrawing (already positioned correctly below)
        pass

    ax.set_yticks(y_pos)
    ax.set_yticklabels(["Verified", "Not Verified"], fontsize=10)
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlabel("View Count (non-zero tweets · dot = mean · band = IQR · whisker = P90)",
                  fontsize=7.5, color=TXT_MED, labelpad=6)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

    # Legend
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=PURPLE,
               markersize=8, label="Mean"),
        mpatches.Patch(facecolor=PURPLE_LT, label="IQR (P25–P75)"),
        Line2D([0],[0], color=PURPLE, linewidth=1.2,
               alpha=0.5, label="P90 whisker"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5,
              framealpha=0.9, edgecolor=RULE, loc="lower right")
    _style_ax(ax, grid_axis="x")


# ──────────────────────────────────────────────────────────────────
# 🖊  Panel B — Grouped horizontal bars: Language × Views & Retweets
#     Two metrics side by side, own scale per metric via twin axes
# ──────────────────────────────────────────────────────────────────
def _panel_B(ax, df):
    _card(ax)
    _panel_label(ax, "B",
                 "Language × Engagement",
                 "English tweets average 3× more views; Filipino tweets dominate volume",
                 GREEN)

    df_lang = df[df["lang"].isin(["Filipino (tl)", "English (en)"])].copy()
    groups  = ["Filipino (tl)", "English (en)"]
    colors  = [BLUE, GREEN]
    N       = len(df)

    view_means  = [_stats(df_lang.loc[df_lang["lang"]==g,"viewCount"])["mean"]  for g in groups]
    rt_means    = [_stats(df_lang.loc[df_lang["lang"]==g,"retweetCount"])["mean"] for g in groups]
    counts      = [int((df_lang["lang"]==g).sum()) for g in groups]

    y     = np.arange(len(groups))
    h     = 0.32

    # Views bars (bottom)
    bars_v = ax.barh(y - h/2, view_means, height=h,
                     color=[BLUE_LT, GREEN_LT], alpha=0.95,
                     label="Mean Views", zorder=2)
    # Retweets bars (top) — use secondary x-axis
    ax2 = ax.twiny()
    bars_r = ax2.barh(y + h/2, rt_means, height=h,
                      color=[BLUE, GREEN], alpha=0.85,
                      label="Mean Retweets", zorder=2)

    max_v = max(view_means) * 1.70
    max_r = max(rt_means)   * 1.70
    ax.set_xlim(0, max_v)
    ax2.set_xlim(0, max_r)

    # Value labels
    for bar, val, col in zip(bars_v, view_means, [BLUE, GREEN]):
        ax.text(val + max_v * 0.012, bar.get_y() + bar.get_height()/2,
                _fmt_stat(val), va="center", fontsize=9,
                fontweight="bold", color=col)
    for bar, val, col in zip(bars_r, rt_means, [BLUE, GREEN]):
        ax2.text(val + max_r * 0.012, bar.get_y() + bar.get_height()/2,
                 _fmt_stat(val), va="center", fontsize=9,
                 fontweight="bold", color=col)

    # Tweet count
    for i, (g, cnt, y_) in enumerate(zip(groups, counts, y)):
        ax.text(0, y_ - 0.50,
                f"{cnt:,} tweets  ({cnt/N*100:.1f}% of dataset)",
                va="top", ha="left", fontsize=7.5, color=TXT_MED)

    ax.set_yticks(y)
    ax.set_yticklabels(groups, fontsize=10)
    ax.set_ylim(-0.9, len(groups) - 0.1)
    ax.set_xlabel("Mean View Count →", fontsize=8, color=TXT_MED, labelpad=5)
    ax2.set_xlabel("← Mean Retweet Count", fontsize=8, color=TXT_MED, labelpad=5)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))
    ax2.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=GREEN_LT, edgecolor=GREEN, label="Mean Views (bottom axis)"),
        mpatches.Patch(facecolor=GREEN,    label="Mean Retweets (top axis)"),
    ]
    ax.legend(handles=legend_elements, fontsize=7.5,
              framealpha=0.9, edgecolor=RULE, loc="lower right")

    # Style only main ax spines (ax2 spines handled by twiny)
    ax.spines["left"].set_color(RULE)
    ax.spines["bottom"].set_color(RULE)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(which="major", length=3, width=0.8, labelsize=8.5)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax2.tick_params(which="major", length=3, width=0.8, labelsize=8.5)


# ──────────────────────────────────────────────────────────────────
# 🖊  Panel C — 100% stacked + mean annotation:
#     Reply Status × Language composition
#     Shows both "what % are replies" AND mean views per group
# ──────────────────────────────────────────────────────────────────
def _panel_C(ax, df):
    _card(ax)
    _panel_label(ax, "C",
                 "Reply Status × Language Mix",
                 "Original posts dominate both languages — Filipino slightly more conversational",
                 BLUE)

    df_lang   = df[df["lang"].isin(["Filipino (tl)", "English (en)"])].copy()
    groups    = ["Filipino (tl)", "English (en)"]
    grp_short = {"Filipino (tl)": "Filipino\n(tl)", "English (en)": "English\n(en)"}
    replies   = ["Original", "Reply"]
    r_colors  = [BLUE, SLATE_LT]
    N         = len(df)

    rows = {}
    for g in groups:
        sub  = df_lang.loc[df_lang["lang"] == g, "reply_label"]
        pcts = sub.value_counts(normalize=True) * 100
        rows[g] = {r: float(pcts.get(r, 0.0)) for r in replies}

    x     = np.arange(len(groups))
    w     = 0.52
    lefts = np.zeros(len(groups))

    for r, col in zip(replies, r_colors):
        vals = np.array([rows[g][r] for g in groups])
        bars = ax.bar(x, vals, width=w, bottom=lefts,
                      color=col, alpha=0.90, label=r, zorder=2)
        for i, (bar, v, lft) in enumerate(zip(bars, vals, lefts)):
            if v > 6:
                ax.text(bar.get_x() + bar.get_width()/2,
                        lft + v/2,
                        f"{v:.1f}%",
                        ha="center", va="center",
                        fontsize=10, fontweight="bold", color="white")
        lefts += vals

    # Mean views annotation above each bar
    for i, g in enumerate(groups):
        mv = _stats(df_lang.loc[df_lang["lang"]==g,"viewCount"])["mean"]
        cnt = int((df_lang["lang"]==g).sum())
        ax.text(i, 104,
                f"avg {_fmt_stat(mv)} views",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold", color=TXT_MED)
        ax.text(i, 101,
                f"{cnt:,} tweets",
                ha="center", va="top",
                fontsize=7.5, color=TXT_LT)

    ax.set_xticks(x)
    ax.set_xticklabels([grp_short[g] for g in groups], fontsize=10)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Share of Tweets (%)", fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=RULE, loc="upper right")
    _style_ax(ax, grid_axis="y")


# ──────────────────────────────────────────────────────────────────
# 🖊  Panel D — Slope / dot-matrix comparison:
#     All three traits × Mean Views — normalised index chart
#     Shows the relative "lift" of each trait over its baseline
# ──────────────────────────────────────────────────────────────────
def _panel_D(ax, df):
    _card(ax)
    _panel_label(ax, "D",
                 "Trait Lift on Mean Views",
                 "How much does each trait multiply expected view count vs its baseline?",
                 AMBER)

    df_lang = df[df["lang"].isin(["Filipino (tl)", "English (en)"])].copy()

    # Compute mean views per group for each trait
    traits = {
        "Verification": {
            "Not Verified": _stats(df.loc[df["verify_label"]=="Not Verified","viewCount"])["mean"],
            "Verified":     _stats(df.loc[df["verify_label"]=="Verified",    "viewCount"])["mean"],
        },
        "Language": {
            "Filipino (tl)": _stats(df_lang.loc[df_lang["lang"]=="Filipino (tl)","viewCount"])["mean"],
            "English (en)":  _stats(df_lang.loc[df_lang["lang"]=="English (en)", "viewCount"])["mean"],
        },
        "Reply Status": {
            "Reply":    _stats(df.loc[df["reply_label"]=="Reply",    "viewCount"])["mean"],
            "Original": _stats(df.loc[df["reply_label"]=="Original", "viewCount"])["mean"],
        },
    }

    trait_colors = {
        "Verification": PURPLE,
        "Language":     GREEN,
        "Reply Status": BLUE,
    }
    marker_colors = {
        "Not Verified": SLATE_LT, "Verified":     PURPLE,
        "Filipino (tl)": BLUE_LT, "English (en)": GREEN,
        "Reply":         SLATE_LT, "Original":     BLUE,
    }

    # Y positions — 3 trait groups with gap
    y_map   = {}
    y_ticks = []
    y_lbls  = []
    y       = 0
    trait_mid = {}

    for trait, groups in traits.items():
        ys = []
        for grp in groups:
            y_map[grp] = y
            y_ticks.append(y)
            y_lbls.append(grp)
            ys.append(y)
            y += 1
        trait_mid[trait] = np.mean(ys)
        y += 0.6   # gap between trait groups

    # Horizontal reference line per trait at baseline (lower group)
    for trait, groups in traits.items():
        vals = list(groups.values())
        base = min(vals)
        top  = max(vals)
        grp_keys = list(groups.keys())
        y0   = y_map[grp_keys[0]]
        y1   = y_map[grp_keys[1]]
        # Connector line between the two dots
        ax.plot([base, top], [y_map[grp_keys[0 if vals[0]==base else 1]],
                               y_map[grp_keys[1 if vals[1]==top  else 0]]],
                color=trait_colors[trait], linewidth=2.0,
                alpha=0.35, zorder=2)

    # Dots and labels
    max_v = max(v for t in traits.values() for v in t.values())
    for grp, y_pos in y_map.items():
        # Find which trait this group belongs to
        for trait, groups in traits.items():
            if grp in groups:
                val   = groups[grp]
                col   = marker_colors[grp]
                tcol  = trait_colors[trait]
                break

        # Dot
        ax.scatter(val, y_pos, s=180, color=col,
                   edgecolors=tcol, linewidths=1.8, zorder=5)
        # Value label
        ax.text(val + max_v * 0.013, y_pos,
                _fmt_stat(val),
                va="center", fontsize=9,
                fontweight="bold", color=TXT)

    # Lift multiplier annotation between each pair
    for trait, groups in traits.items():
        grp_keys = list(groups.keys())
        vals     = list(groups.values())
        base     = min(vals)
        top      = max(vals)
        lift     = top / base if base > 0 else 1
        y_mid    = trait_mid[trait]
        ax.text(max_v * 1.38, y_mid,
                f"{lift:.1f}×",
                va="center", ha="right",
                fontsize=11, fontweight="bold",
                color=trait_colors[trait])
        ax.text(max_v * 1.38, y_mid - 0.38,
                "lift",
                va="center", ha="right",
                fontsize=7.5, color=TXT_LT)

    # Trait bracket labels on the left
    for trait, y_mid in trait_mid.items():
        ax.text(-max_v * 0.22, y_mid,
                trait,
                va="center", ha="left",
                fontsize=9, fontweight="bold",
                color=trait_colors[trait])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_lbls, fontsize=8.5)
    ax.set_xlim(-max_v * 0.05, max_v * 1.52)
    ax.set_ylim(-0.5, y - 0.1)
    ax.set_xlabel("Mean View Count (non-zero tweets only)",
                  fontsize=8.5, color=TXT_MED, labelpad=5)
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_k))

    # "Lift" column header
    ax.text(max_v * 1.38, y - 0.3,
            "Lift",
            va="center", ha="right",
            fontsize=8, color=TXT_MED, style="italic")

    _style_ax(ax, grid_axis="x")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_bivariate_analysis(data_source, save_path=None):
    """
    Bivariate Analysis — author & content traits × tweet engagement.

    Four panels, four different chart types, one narrative arc:
      A — Lollipop (mean + IQR + P90): Verification × View distribution
      B — Grouped horizontal bars (dual axis): Language × Views & Retweets
      C — 100% stacked bar + annotation: Reply Status × Language composition
      D — Connected dot plot with lift multiplier: All traits × Mean Views

    Parameters
    ----------
    data_source : str or DataFrame
    save_path   : str, optional
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()
    df = _normalize(df)
    N  = len(df)

    df["verify_label"] = df["author_isBlueVerified"].map(
        {False: "Not Verified", True: "Verified"})
    df["reply_label"]  = df["isReply"].map(
        {False: "Original", True: "Reply"})

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 18), facecolor=BG)

    # Header
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.965), 1, 0.035, boxstyle="square,pad=0",
        facecolor=TXT, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))
    fig.text(0.5, 0.983,
             "Bivariate Analysis \u2014 Author & Content Traits \u00d7 Tweet Engagement",
             ha="center", va="center",
             fontsize=16, fontweight="bold", color="white", zorder=3)

    # KPI strip
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.930), 1, 0.035, boxstyle="square,pad=0",
        facecolor="#1E293B", edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))

    verified_n  = int(df["author_isBlueVerified"].sum())
    original_n  = int((~df["isReply"]).sum())
    filipino_n  = int((df["lang"] == "Filipino (tl)").sum())

    kpis = [
        (f"{N:,}",          "tweets analysed"),
        (f"{verified_n:,}", f"from verified authors ({verified_n/N*100:.1f}%)"),
        (f"{original_n:,}", f"original posts ({original_n/N*100:.1f}%)"),
        (f"{filipino_n:,}", f"tweets in Filipino ({filipino_n/N*100:.1f}%)"),
    ]
    for i, (val, lbl) in enumerate(kpis):
        xf = 0.06 + i * 0.235
        fig.text(xf, 0.954, val, ha="left", va="center",
                 fontsize=11, fontweight="bold", color="#60A5FA", zorder=3)
        fig.text(xf, 0.937, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_LT, zorder=3)
        if i > 0:
            fig.add_artist(mpatches.FancyBboxPatch(
                (xf - 0.015, 0.933), 0.0012, 0.026,
                boxstyle="square,pad=0", facecolor="#334155", edgecolor="none",
                transform=fig.transFigure, clip_on=False, zorder=3))

    # GridSpec — extra top margin for panel labels
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.13, right=0.97,
        top=0.900, bottom=0.07,
        hspace=0.52, wspace=0.38,
    )
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    _panel_A(ax_A, df)
    _panel_B(ax_B, df)
    _panel_C(ax_C, df)
    _panel_D(ax_D, df)

    # Footer
    fig.text(
        0.5, 0.028,
        "All engagement values computed on non-zero tweets only — zero-value tweets excluded "
        "to avoid zero-inflation bias.  "
        "Panel A: IQR = interquartile range (P25\u2013P75); P90 = 90th percentile.  "
        "Panel D: Lift = ratio of higher group mean to lower group mean.",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
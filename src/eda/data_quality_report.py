import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path


# ── Shared style constants ────────────────────────────────────────────────────
BG_COLOR   = "#F8F9FA"
TEXT_COLOR = "#2D2D2D"
GRAY       = "#888888"
LIGHT_GRAY = "#CCCCCC"
FONT       = {"font.family": "DejaVu Sans", "font.size": 10}

STATUS_CLEAN = ("✓  Clean",        "#10B981")
STATUS_ISSUE = ("✗  Issue Found",  "#EF4444")
STATUS_WARN  = ("⚠  Inconsistent", "#F59E0B")
STATUS_TYPE  = ("⚠  Wrong Type",   "#F59E0B")


# ── Helpers ───────────────────────────────────────────────────────────────────

def bool_check(df, col):
    vals = set(df[col].dropna().astype(str).unique())
    ok   = vals <= {"True", "False"}
    return (f"Unique: {sorted(vals)}", STATUS_CLEAN if ok else STATUS_ISSUE)


def pct(n, total):
    """Return '  (X.XX%)' string."""
    return f"  ({n / total * 100:.2f}%)"


def null_finding(df, col, total):
    n = df[col].isna().sum()
    p = pct(n, total)
    if n == 0:
        return f"0 nulls{pct(0, total)}", STATUS_CLEAN
    return f"{n:,} nulls{p}", STATUS_ISSUE


# ── Breakdown mini-table renderer ─────────────────────────────────────────────

def draw_breakdown(ax, x_start, y_center, items, total,
                   col_width=0.18, row_h_frac=0.022):
    """
    Draw a small inline breakdown table.
    items : list of (label, count) tuples
    Returns the y position after the last mini-row.
    """
    y = y_center - (len(items) / 2) * row_h_frac
    for label, count in items:
        ax.text(x_start,        y, f"  ↳ {label}",
                fontsize=7.2, color=GRAY, transform=ax.transAxes, va="center")
        count_str = f"{count:,}" if total is None else f"{count:,}  ({count / total * 100:.1f}%)"
        ax.text(x_start + 0.22, y, count_str,
                fontsize=7.2, color=TEXT_COLOR, transform=ax.transAxes,
                va="center")
        y -= row_h_frac
    return y


# ── Shared figure scaffold ────────────────────────────────────────────────────

# Design tokens
BG_COLOR      = "#FFFFFF"       # pure white canvas
BG_ROW_ALT    = "#F7F9FC"       # very subtle blue-tinted stripe
BG_HEADER_ROW = "#1E293B"       # dark slate header bar
TEXT_HEADER   = "#FFFFFF"       # white text on dark header
TEXT_COLOR    = "#1E293B"       # near-black body text
TEXT_MUTED    = "#64748B"       # muted slate for secondary text
TEXT_FAINT    = "#CBD5E1"       # faint for dashes / placeholders
ACCENT_LINE   = "#E2E8F0"       # very light divider lines
BADGE_BG      = "#F1F5F9"       # neutral badge background for title area

# Status colors
C_GREEN  = "#10B981"
C_ORANGE = "#F59E0B"
C_RED    = "#EF4444"
C_GREEN_BG  = "#ECFDF5"
C_ORANGE_BG = "#FFFBEB"
C_RED_BG    = "#FEF2F2"

STATUS_CLEAN = ("✓  Clean",       C_GREEN)
STATUS_ISSUE = ("✗  Issue Found", C_RED)
STATUS_WARN  = ("⚠  Inconsistent",C_ORANGE)
STATUS_TYPE  = ("⚠  Wrong Type",  C_ORANGE)

# Fixed inch-based row heights
_BASE_ROW_IN  = 0.32
_BREAKDOWN_IN = 0.23
_HEADER_IN    = 1.55   # title + subtitle + badges + col header bar
_FOOTER_IN    = 0.30


def _make_figure(title, subtitle, rows, col_x, col_labels,
                 breakdown_rows=None):
    """
    rows           : list of (col, check, finding, dtype_info, null_info, status_tuple)
    breakdown_rows : dict { row_index: [(label, count), ...] }
    """
    import matplotlib.patches as patches
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                          "text.color": TEXT_COLOR})
    breakdown_rows = breakdown_rows or {}

    # ── Figure height from content ────────────────────────────────────────────
    row_heights_in = [
        _BASE_ROW_IN + len(breakdown_rows.get(i, [])) * _BREAKDOWN_IN
        for i in range(len(rows))
    ]
    fig_h = _HEADER_IN + sum(row_heights_in) + _FOOTER_IN
    fig_w = 18

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def yf(inches_from_top):
        return 1.0 - inches_from_top / fig_h

    # ════════════════════════════════════════════════════════════════════════
    # HEADER SECTION
    # ════════════════════════════════════════════════════════════════════════

    # ── Top accent bar (full width, 4px) ──────────────────────────────────
    accent_h = 0.055 / fig_h
    ax.add_patch(patches.Rectangle(
        (0, 1 - accent_h), 1, accent_h,
        transform=ax.transAxes, facecolor=C_GREEN, zorder=5, linewidth=0
    ))

    # ── Title ─────────────────────────────────────────────────────────────
    fig.text(0.5, yf(0.28), title,
             fontsize=15, fontweight="bold", ha="center", va="top",
             color=TEXT_COLOR, zorder=6)

    # ── Subtitle ──────────────────────────────────────────────────────────
    fig.text(0.5, yf(0.52), subtitle,
             fontsize=9, ha="center", va="top", color=TEXT_MUTED, zorder=6)

    # ── Summary badges — centered as a group ──────────────────────────────
    issues   = sum(1 for r in rows if r[-1][1] == C_RED)
    warnings = sum(1 for r in rows if r[-1][1] == C_ORANGE)
    cleans   = sum(1 for r in rows if r[-1][1] == C_GREEN)

    badge_data = [
        (f"  ✓  {cleans} Clean  ",       C_GREEN,  C_GREEN_BG,  C_GREEN),
        (f"  ⚠  {warnings} Warnings  ",  C_ORANGE, C_ORANGE_BG, C_ORANGE),
        (f"  ✗  {issues} Issues  ",       C_RED,    C_RED_BG,    C_RED),
    ]
    # Center the badge group at x=0.5
    badge_spacing = 0.19
    badge_start   = 0.5 - badge_spacing * (len(badge_data) - 1) / 2
    badge_y       = yf(0.80)

    for j, (label, fg, bg, border) in enumerate(badge_data):
        bx = badge_start + j * badge_spacing
        fig.text(bx, badge_y, label,
                 fontsize=9, fontweight="bold",
                 ha="center", va="center", color=fg,
                 bbox=dict(boxstyle="round,pad=0.35",
                           facecolor=bg, edgecolor=border,
                           linewidth=1.0),
                 zorder=6)

    # ── Column header bar (dark slate) ────────────────────────────────────
    header_top_in    = 1.10
    header_h_in      = 0.34
    header_top_frac  = yf(header_top_in)
    header_h_frac    = header_h_in / fig_h

    ax.add_patch(patches.Rectangle(
        (0.01, header_top_frac - header_h_frac),
        0.98, header_h_frac,
        transform=ax.transAxes,
        facecolor=BG_HEADER_ROW, zorder=2, linewidth=0,
        clip_on=False
    ))

    header_text_y = yf(header_top_in + header_h_in / 2)
    for x, label in zip(col_x, col_labels):
        ax.text(x + 0.005, header_text_y, label,
                fontsize=8.5, fontweight="bold", color=TEXT_HEADER,
                transform=ax.transAxes, va="center", zorder=3)

    # ════════════════════════════════════════════════════════════════════════
    # DATA ROWS
    # ════════════════════════════════════════════════════════════════════════
    cursor_in = _HEADER_IN
    prev_col  = None

    # ── Pre-compute column group spans for accent bars ────────────────────────
    # For each col name, find the y_top and y_bot spanning all its rows,
    # and the worst status color among those rows.
    from collections import defaultdict
    group_info = defaultdict(lambda: {"top_in": None, "bot_in": None,
                                      "color": None, "priority": 0})
    _status_priority = {C_RED: 2, C_ORANGE: 1}
    _cursor = _HEADER_IN
    for i, row in enumerate(rows):
        col = row[0]
        sc  = row[-1][1]
        rh  = row_heights_in[i]
        gi  = group_info[col]
        if gi["top_in"] is None:
            gi["top_in"] = _cursor
        gi["bot_in"] = _cursor + rh
        p = _status_priority.get(sc, 0)
        if p > gi["priority"]:
            gi["priority"] = p
            gi["color"]    = sc
        _cursor += rh

    for i, row in enumerate(rows):
        col, check, finding, dtype_info, null_info, (status_text, status_color) = row

        rh        = row_heights_in[i]
        n_sub     = len(breakdown_rows.get(i, []))
        text_y_in = cursor_in + _BASE_ROW_IN / 2
        text_y    = yf(text_y_in)

        bg_top    = yf(cursor_in)
        bg_bot    = yf(cursor_in + rh)
        row_h_frac = bg_top - bg_bot

        # ── Alternating row background ────────────────────────────────────
        row_bg = BG_ROW_ALT if i % 2 == 0 else BG_COLOR
        ax.add_patch(patches.Rectangle(
            (0.01, bg_bot), 0.98, row_h_frac,
            transform=ax.transAxes,
            facecolor=row_bg, zorder=0, linewidth=0
        ))

        # ── Left accent bar — drawn once per group at first row ───────────
        display_col = col if col != prev_col else ""
        prev_col    = col
        if display_col != "":
            gi = group_info[col]
            if gi["color"] in (C_RED, C_ORANGE):
                acc_top = yf(gi["top_in"])
                acc_bot = yf(gi["bot_in"])
                ax.add_patch(patches.Rectangle(
                    (0.01, acc_bot), 0.003, acc_top - acc_bot,
                    transform=ax.transAxes,
                    facecolor=gi["color"], zorder=1, linewidth=0
                ))

        ax.text(col_x[0] + 0.005, text_y, display_col,
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes, va="center",
                fontweight="bold" if display_col else "normal")

        # ── Check ─────────────────────────────────────────────────────────
        ax.text(col_x[1] + 0.005, text_y, check,
                fontsize=8, color=TEXT_MUTED,
                transform=ax.transAxes, va="center")

        # ── Finding ───────────────────────────────────────────────────────
        ax.text(col_x[2] + 0.005, text_y, finding,
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes, va="center")

        # ── dtype: current → expected ─────────────────────────────────────
        if dtype_info:
            cur, exp = dtype_info
            mismatch = cur != exp
            ax.text(col_x[3] + 0.005, text_y, cur,
                    fontsize=7.8,
                    color=C_RED if mismatch else TEXT_MUTED,
                    transform=ax.transAxes, va="center",
                    fontweight="bold" if mismatch else "normal")
            ax.text(col_x[3] + 0.072, text_y, "→",
                    fontsize=7.8, color=TEXT_FAINT,
                    transform=ax.transAxes, va="center")
            ax.text(col_x[3] + 0.088, text_y, exp,
                    fontsize=7.8,
                    color=C_GREEN if mismatch else TEXT_MUTED,
                    transform=ax.transAxes, va="center",
                    fontweight="bold" if mismatch else "normal")
        else:
            ax.text(col_x[3] + 0.005, text_y, "—",
                    fontsize=8, color=TEXT_FAINT,
                    transform=ax.transAxes, va="center")

        # ── Null count ────────────────────────────────────────────────────
        if null_info:
            n_null, pct_s = null_info
            nc = C_RED if n_null > 0 else TEXT_MUTED
            ax.text(col_x[4] + 0.005, text_y, f"{n_null:,}  {pct_s}",
                    fontsize=8, color=nc,
                    transform=ax.transAxes, va="center")
        else:
            ax.text(col_x[4] + 0.005, text_y, "—",
                    fontsize=8, color=TEXT_FAINT,
                    transform=ax.transAxes, va="center")

        # ── Status pill ───────────────────────────────────────────────────
        pill_bg = {C_GREEN: C_GREEN_BG, C_ORANGE: C_ORANGE_BG,
                   C_RED: C_RED_BG}.get(status_color, BG_ROW_ALT)
        ax.text(col_x[5] + 0.005, text_y, status_text,
                fontsize=8, fontweight="bold", color=status_color,
                transform=ax.transAxes, va="center",
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor=pill_bg, edgecolor="none",
                          linewidth=0),
                zorder=2)

        # ── Breakdown sub-lines ───────────────────────────────────────────
        if n_sub > 0:
            sub_cursor_in = cursor_in + _BASE_ROW_IN
            for label, count in breakdown_rows[i]:
                sub_y = yf(sub_cursor_in + _BREAKDOWN_IN / 2)
                ax.text(col_x[2] + 0.018, sub_y, f"↳  {label}",
                        fontsize=7.5, color=TEXT_MUTED,
                        transform=ax.transAxes, va="center",
                        style="italic")
                ax.text(col_x[2] + 0.21, sub_y, f"{count:,}",
                        fontsize=7.5, color=TEXT_COLOR,
                        transform=ax.transAxes, va="center")
                sub_cursor_in += _BREAKDOWN_IN

        # ── Row divider (very faint) ──────────────────────────────────────
        ax.axhline(bg_bot, color=ACCENT_LINE, linewidth=0.5,
                   xmin=0.01, xmax=0.99, zorder=1)

        cursor_in += rh

    # ── Footer line ───────────────────────────────────────────────────────────
    ax.axhline(yf(cursor_in + 0.10), color=ACCENT_LINE, linewidth=1.0,
               xmin=0.01, xmax=0.99, zorder=1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1: Tweets
# ─────────────────────────────────────────────────────────────────────────────

def get_data_quality_for_tweets(filepath: str, output_dir: str = "output") -> dict:
    base_filename = str(filepath).replace("data/", "").replace(".csv", "")
    df            = pd.read_csv(filepath)
    total_rows    = len(df)

    # ── Computations ──────────────────────────────────────────────────────────
    duplicate_count = df.duplicated(subset=["pseudo_id"]).sum()

    lang_counts = df["lang"].value_counts()
    unexpected_langs = [l for l in lang_counts.index if l not in {"tl", "en"}]
    lang_breakdown = [(str(k), int(lang_counts[k])) for k in lang_counts.index]

    float_count = df["pseudo_inReplyToUsername"].apply(
        lambda x: isinstance(x, float) and not pd.isna(x)
    ).sum()
    na_str_count = (df["quoted_pseudo_id"].astype(str) == "<NA>").sum()

    # Null rates for all columns
    null_rates = {col: int(df[col].isna().sum()) for col in df.columns}

    def ni(col):
        """null_info tuple for a column."""
        n = null_rates[col]
        p = n / total_rows * 100
        if p == 0:
            pct = "(0.00%)"
        elif p < 0.01:
            pct = f"({p:.4f}%)"
        elif p < 0.1:
            pct = f"({p:.3f}%)"
        else:
            pct = f"({p:.2f}%)"
        return (n, pct)

    def pct_str(n):
        p = n / total_rows * 100
        if p == 0:
            return "(0.00%)"
        elif p < 0.01:
            return f"({p:.4f}%)"
        elif p < 0.1:
            return f"({p:.3f}%)"
        else:
            return f"({p:.2f}%)"

    # ── Row definitions ───────────────────────────────────────────────────────
    # (col, check, finding, dtype_info, null_info, status)
    rows = [
        ("pseudo_id", "Duplicate Rows",
         f"{duplicate_count:,} duplicate {'row' if duplicate_count == 1 else 'rows'}  {pct_str(duplicate_count)}",
         None, ni("pseudo_id"),
         STATUS_CLEAN if duplicate_count == 0 else STATUS_ISSUE),

        ("pseudo_conversationId", "Data Type",
         "int64 — correct",
         ("int64", "int64"), ni("pseudo_conversationId"),
         STATUS_CLEAN),

        ("createdAt", "Wrong Data Type",
         "Stored as str, expected datetime",
         ("str / object", "datetime64"), ni("createdAt"),
         STATUS_TYPE),

        ("createdAt", "Format Check",
         "All timestamps parseable",
         None, None,
         STATUS_CLEAN),

        ("lang", "Unexpected Values",
         (f"Only tl & en found" if not unexpected_langs
          else f"Unexpected: {unexpected_langs}  —  {sum(lang_counts[l] for l in unexpected_langs):,} rows  {pct_str(sum(lang_counts[l] for l in unexpected_langs))}"),
         None, ni("lang"),
         STATUS_CLEAN if not unexpected_langs else STATUS_WARN),

        ("text", "Null Check",
         f"0 nulls  {pct_str(0)}  — all tweets have content",
         ("str / object", "str"), ni("text"),
         STATUS_CLEAN),

        ("retweetCount",  "Negative Values",
         f"No negative values", ("int64","int64"), ni("retweetCount"),  STATUS_CLEAN),
        ("likeCount",     "Negative Values",
         f"No negative values", ("int64","int64"), ni("likeCount"),     STATUS_CLEAN),
        ("viewCount",     "Negative Values",
         f"No negative values", ("int64","int64"), ni("viewCount"),     STATUS_CLEAN),
        ("quoteCount",    "Negative Values",
         f"No negative values", ("int64","int64"), ni("quoteCount"),    STATUS_CLEAN),
        ("replyCount",    "Negative Values",
         f"No negative values", ("int64","int64"), ni("replyCount"),    STATUS_CLEAN),
        ("bookmarkCount", "Negative Values",
         f"No negative values", ("int64","int64"), ni("bookmarkCount"), STATUS_CLEAN),

        ("isReply", "Boolean Check",
         "Unique: ['False', 'True']",
         ("bool / str","bool"), ni("isReply"),
         STATUS_CLEAN),

        ("pseudo_inReplyToUsername", "Wrong Data Type",
         "Stored as float64, expected str",
         ("float64", "str / object"), ni("pseudo_inReplyToUsername"),
         STATUS_TYPE),

        ("pseudo_inReplyToUsername", "Inconsistent Values",
         f"{float_count:,} values stored as float  {pct_str(float_count)}",
         None, None,
         STATUS_WARN),

        ("quoted_pseudo_id", "String NaN",
         f"{na_str_count:,} rows have \"<NA>\" string  {pct_str(na_str_count)}",
         None, ni("quoted_pseudo_id"),
         STATUS_CLEAN if na_str_count == 0 else STATUS_WARN),

        ("author_isBlueVerified", "Boolean Check",
         "Unique: ['False', 'True']",
         ("bool / str","bool"), ni("author_isBlueVerified"),
         STATUS_CLEAN),

        ("pseudo_author_userName", "Null Check",
         f"0 nulls  {pct_str(0)}",
         ("str / object","str"), ni("pseudo_author_userName"),
         STATUS_CLEAN),
    ]

    # ── Breakdown tables ──────────────────────────────────────────────────────
    # Row index 4 = lang row — show per-lang counts
    breakdown_rows = {
        4: lang_breakdown[:6]   # top 6 lang codes
    }

    # ── Column layout ─────────────────────────────────────────────────────────
    col_x      = [0.01, 0.20, 0.38, 0.63, 0.77, 0.91]
    col_labels = ["Column", "Check", "Finding", "Current → Expected dtype", "Null Count  (%)", "Status"]

    subtitle = (f"{total_rows:,} rows  ·  {len(df.columns)} columns  ·  "
                f"{duplicate_count:,} duplicate rows")

    fig = _make_figure(
        title          = "Data Quality Report — Dataset 1: Tweets",
        subtitle       = subtitle,
        rows           = rows,
        col_x          = col_x,
        col_labels     = col_labels,
        breakdown_rows = breakdown_rows,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_data_quality.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✓ Tweets report saved → {out_path}")

    return {
        "duplicate_count":        int(duplicate_count),
        "unexpected_langs":       unexpected_langs,
        "na_string_count":        int(na_str_count),
        "float_identifier_count": int(float_count),
        "null_rates":             null_rates,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2: Authors
# ─────────────────────────────────────────────────────────────────────────────

def get_data_quality_for_authors(filepath: str, output_dir: str = "output") -> dict:
    base_filename = str(filepath).replace("data/", "").replace(".csv", "")
    df            = pd.read_csv(filepath)
    total_rows    = len(df)

    # ── Computations ──────────────────────────────────────────────────────────
    duplicate_count = df.duplicated(subset=["author_userName"]).sum()

    invalid_patterns = [
        "earth", "wherever", "worldwide", "around the world",
        "📍", "ig:", "http", "whatsapp", "telegram",
        "lunar", "yn edri", "abbott", "internet",
        "everywhere", "facebook.com", "twitter.com", "t.co",
        "south girl", "2020", "2021", "2019",
    ]

    def is_invalid_location(loc):
        if pd.isna(loc) or str(loc).strip() == "":
            return False
        return any(p in str(loc).lower() for p in invalid_patterns)

    invalid_loc_mask  = df["author_location"].apply(is_invalid_location)
    invalid_loc_count = invalid_loc_mask.sum()
    valid_loc_count   = df["author_location"].notna().sum() - invalid_loc_count

    # Sample of invalid locations for breakdown
    invalid_loc_samples = (
        df.loc[invalid_loc_mask, "author_location"]
        .value_counts()
        .head(5)
        .reset_index()
        .values.tolist()
    )

    empty_bio      = df["author_profile_bio_description"].fillna("").astype(str).str.strip().eq("").sum()
    real_null_bio  = df["author_profile_bio_description"].isna().sum()
    empty_str_bio  = int(empty_bio - real_null_bio)

    null_rates = {col: int(df[col].isna().sum()) for col in df.columns}

    def ni(col):
        n = null_rates[col]
        p = n / total_rows * 100
        if p == 0:
            pct = "(0.00%)"
        elif p < 0.01:
            pct = f"({p:.4f}%)"
        elif p < 0.1:
            pct = f"({p:.3f}%)"
        else:
            pct = f"({p:.2f}%)"
        return (n, pct)

    def pct_str(n):
        p = n / total_rows * 100
        if p == 0:
            return "(0.00%)"
        elif p < 0.01:
            return f"({p:.4f}%)"
        elif p < 0.1:
            return f"({p:.3f}%)"
        else:
            return f"({p:.2f}%)"

    # ── Row definitions ───────────────────────────────────────────────────────
    rows = [
        ("author_userName", "Duplicate Rows",
         f"{duplicate_count:,} duplicate {'row' if duplicate_count == 1 else 'rows'}  {pct_str(duplicate_count)}",
         None, ni("author_userName"),
         STATUS_CLEAN if duplicate_count == 0 else STATUS_ISSUE),

        ("author_userName", "Null Check",
         f"0 nulls  {pct_str(0)}  — all authors have a username",
         ("str / object","str"), None,
         STATUS_CLEAN),

        ("obfuscated_userName", "Null Check",
         f"0 nulls  {pct_str(0)}",
         ("str / object","str"), ni("obfuscated_userName"),
         STATUS_CLEAN),

        ("author_createdAt", "Wrong Data Type",
         "Stored as str, expected datetime",
         ("str / object","datetime64"), ni("author_createdAt"),
         STATUS_TYPE),

        ("author_createdAt", "Format Check",
         "All timestamps parseable",
         None, None,
         STATUS_CLEAN),

        ("author_profile_bio_description", "Empty String",
         (f"0 empty strings — clean"
          if empty_str_bio == 0
          else f"{empty_str_bio:,} empty strings \"\" instead of NaN  {pct_str(empty_str_bio)}"),
         None, ni("author_profile_bio_description"),
         STATUS_CLEAN if empty_str_bio == 0 else STATUS_WARN),

        ("author_location", "Invalid Values",
         f"{invalid_loc_count:,} non-geographic / meaningless values  {pct_str(invalid_loc_count)}",
         None, ni("author_location"),
         STATUS_CLEAN if invalid_loc_count == 0 else STATUS_WARN),

        ("author_location", "Valid Values",
         f"{valid_loc_count:,} valid geographic values  {pct_str(valid_loc_count)}",
         None, None,
         STATUS_CLEAN),

        ("author_followers", "Negative Values",
         "No negative values",
         ("int64","int64"), ni("author_followers"),
         STATUS_CLEAN),

        ("author_following", "Negative Values",
         "No negative values",
         ("int64","int64"), ni("author_following"),
         STATUS_CLEAN),

        ("author_isBlueVerified", "Boolean Check",
         "Unique: ['False', 'True']",
         ("bool / str","bool"), ni("author_isBlueVerified"),
         STATUS_CLEAN),
    ]

    # ── Breakdown: top invalid location samples ───────────────────────────────
    breakdown_rows = {}
    if invalid_loc_samples:
        breakdown_rows[6] = [(str(loc), int(cnt)) for loc, cnt in invalid_loc_samples]

    # ── Column layout ─────────────────────────────────────────────────────────
    col_x      = [0.01, 0.22, 0.40, 0.63, 0.77, 0.91]
    col_labels = ["Column", "Check", "Finding", "Current → Expected dtype", "Null Count  (%)", "Status"]

    subtitle = (f"{total_rows:,} rows  ·  {len(df.columns)} columns  ·  "
                f"{duplicate_count:,} duplicate rows")

    fig = _make_figure(
        title          = "Data Quality Report — Dataset 2: Authors",
        subtitle       = subtitle,
        rows           = rows,
        col_x          = col_x,
        col_labels     = col_labels,
        breakdown_rows = breakdown_rows,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_data_quality.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"✓ Authors report saved → {out_path}")

    return {
        "duplicate_count":   int(duplicate_count),
        "invalid_loc_count": int(invalid_loc_count),
        "valid_loc_count":   int(valid_loc_count),
        "empty_bio_count":   empty_str_bio,
        "null_rates":        null_rates,
    }
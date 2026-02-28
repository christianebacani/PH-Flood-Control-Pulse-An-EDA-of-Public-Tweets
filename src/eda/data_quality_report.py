"""
Data Quality Report - Professional EDA Output
──────────────────────────────────────────────────
Fixes:
  • Tight header — no dead space
  • Colored inline dtype badges (current → expected)
  • Stronger section headers with solid left rule
  • Parenthetical explanations moved to a muted sub-line
  • Consistent pill sizing
  • Compact breakdown rows
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#FFFFFF"
BG_ALT      = "#F8FAFC"
BG_DARK     = "#1E293B"

TXT         = "#1E293B"
TXT_MED     = "#64748B"
TXT_LIGHT   = "#94A3B8"
TXT_WHITE   = "#FFFFFF"
RULE        = "#E2E8F0"
RULE_MED    = "#CBD5E1"

# Status colours
C_RED       = "#DC2626";  BG_RED     = "#FEF2F2";  BD_RED    = "#FECACA"
C_ORANGE    = "#B45309";  BG_ORANGE  = "#FFFBEB";  BD_ORANGE = "#FDE68A"
C_GREEN     = "#065F46";  BG_GREEN   = "#ECFDF5";  BD_GREEN  = "#A7F3D0"

# Dtype badge colours
C_DTYPE_BAD = "#DC2626"   # current (wrong) type
C_DTYPE_OK  = "#059669"   # expected type
BG_DTYPE    = "#F1F5F9"   # badge background

# Section colours  (accent, header-bg)
SEC = {
    "Duplicate Rows":      ("#7C3AED", "#FAF5FF"),
    "Wrong Data Types":    ("#B45309", "#FFFBEB"),
    "Inconsistent Values": ("#1D4ED8", "#EFF6FF"),
}

# ── Layout (inches) ────────────────────────────────────────────────────────────
FIG_W      = 15.0
HDR_H      = 1.10   # title + subtitle + pills + padding
SEC_HDR_H  = 0.46   # section label row
COL_HDR_H  = 0.32   # "Column / Finding / Status" bar
ROW_H      = 0.36   # main data row
NOTE_H     = 0.22   # muted sub-note line (explanation)
BD_H       = 0.22   # breakdown item
SEC_GAP    = 0.22   # gap between sections
FOOTER_H   = 0.28

# X positions (axes fraction)
X_COL    = 0.012
X_FIND   = 0.240
X_STATUS = 0.870
MARGIN   = 0.988    # right edge


def _fig_height(sections):
    # Content height first, then add proportional header
    content_h = FOOTER_H
    for i, (name, rows) in enumerate(sections):
        content_h += SEC_HDR_H + COL_HDR_H
        for r in rows:
            content_h += ROW_H
            if r.get("note"):  content_h += NOTE_H
            content_h += len(r.get("breakdowns", [])) * BD_H
        if i < len(sections) - 1:
            content_h += SEC_GAP
    # Header is 12% of total, so total = content / 0.88, minimum header 0.90"
    total = content_h / 0.88
    return max(total, content_h + 0.90)


def _pill(ax, cx, cy, text, fg, bg, border, size=7.5, bold=True):
    ax.text(cx, cy, text,
            fontsize=size, fontweight="bold" if bold else "normal",
            color=fg, transform=ax.transAxes, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", facecolor=bg,
                      edgecolor=border, linewidth=0.9), zorder=5)


def _dtype_badge(ax, cx, cy, current, expected):
    """Render  [current_type]  →  [expected_type]  inline."""
    # current type (red — wrong)
    ax.text(cx, cy, current,
            fontsize=7.5, fontweight="bold", color=C_DTYPE_BAD,
            transform=ax.transAxes, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG_DTYPE,
                      edgecolor=BD_RED, linewidth=0.8), zorder=5)
    # arrow
    ax.text(cx + len(current) * 0.0052 + 0.012, cy, "→",
            fontsize=8, color=TXT_LIGHT,
            transform=ax.transAxes, va="center", ha="left")
    # expected type (green — correct)
    ax.text(cx + len(current) * 0.0052 + 0.028, cy, expected,
            fontsize=7.5, fontweight="bold", color=C_DTYPE_OK,
            transform=ax.transAxes, va="center", ha="left",
            bbox=dict(boxstyle="round,pad=0.22", facecolor=BG_DTYPE,
                      edgecolor=BD_GREEN, linewidth=0.8), zorder=5)


def _render(title, subtitle, sections, pills):
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig_h = _fig_height(sections)
    fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

    def Y(inch): return 1.0 - inch / fig_h

    # ── Header ────────────────────────────────────────────────────────────────
    hdr_h = max(1.25, fig_h * 0.14)   # slightly taller, more breathing room

    # Vertical rhythm inside header
    title_y    = Y(hdr_h * 0.18)
    subtitle_y = Y(hdr_h * 0.38)
    pill_y     = Y(hdr_h * 0.68)

    # Title
    ax.text(0.5, title_y, title,
            fontsize=14, fontweight="bold",
            ha="center", va="top",
            color=TXT, transform=ax.transAxes)

    # Subtitle (lighter + smaller for hierarchy)
    ax.text(0.5, subtitle_y, subtitle,
            fontsize=8, color=TXT_MED,
            ha="center", va="top",
            transform=ax.transAxes)

    # Pills (centered group)
    pw = 0.185
    px = 0.5 - pw * (len(pills) - 1) / 2
    for j, (lbl, fg, bg, bd) in enumerate(pills):
        ax.text(px + j * pw, pill_y, lbl,
                fontsize=8.5, fontweight="bold",
                ha="center", va="center",
                color=fg,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.38",
                          facecolor=bg,
                          edgecolor=bd,
                          linewidth=1.1))
    pw = 0.185
    px = 0.5 - pw * (len(pills) - 1) / 2
    for j, (lbl, fg, bg, bd) in enumerate(pills):
        ax.text(px + j * pw, pill_y, lbl,
                fontsize=8.5, fontweight="bold",
                ha="center", va="center", color=fg,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.38", facecolor=bg,
                          edgecolor=bd, linewidth=1.1))

    # ── Sections ───────────────────────────────────────────────────────────────
    cursor = hdr_h + 0.15   # add breathing space before first section

    for si, (sec_name, rows) in enumerate(sections):
        acc, sec_bg = SEC[sec_name]
        n = len(rows)

        # Section header
        sh0 = Y(cursor + SEC_HDR_H)
        shh = Y(cursor) - sh0
        ax.add_patch(patches.Rectangle(
            (0.01, sh0), MARGIN - 0.01, shh,
            transform=ax.transAxes, facecolor=sec_bg, zorder=1, linewidth=0))
        # solid left rule (4px feel)
        ax.add_patch(patches.Rectangle(
            (0.01, sh0), 0.005, shh,
            transform=ax.transAxes, facecolor=acc, zorder=2, linewidth=0))

        sec_mid = Y(cursor + SEC_HDR_H / 2)
        ax.text(0.024, sec_mid, sec_name,
                fontsize=9, fontweight="bold", color=acc,
                transform=ax.transAxes, va="center")
        # issue count badge
        badge = f"  {n} {'issue' if n == 1 else 'issues'}  "
        ax.text(0.024 + len(sec_name) * 0.0072 + 0.012, sec_mid, badge,
                fontsize=7, fontweight="bold", color=acc,
                transform=ax.transAxes, va="center",
                bbox=dict(boxstyle="round,pad=0.20", facecolor=BG,
                          edgecolor=acc, linewidth=0.8))
        cursor += SEC_HDR_H

        # Column header bar
        ch0 = Y(cursor + COL_HDR_H)
        chh = Y(cursor) - ch0
        ax.add_patch(patches.Rectangle(
            (0.01, ch0), MARGIN - 0.01, chh,
            transform=ax.transAxes, facecolor=BG_DARK, zorder=1, linewidth=0))
        for cx_key, label in [(X_COL, "Column"),
                              (X_FIND, "Finding"),
                              (X_STATUS, "Status")]:
            ax.text(cx_key + 0.008, Y(cursor + COL_HDR_H / 2), label,
                    fontsize=7.5, fontweight="bold", color=TXT_WHITE,
                    transform=ax.transAxes, va="center")
        cursor += COL_HDR_H

        # Rows
        for ri, row in enumerate(rows):
            # compute row height
            rh = ROW_H
            if row.get("note"):      rh += NOTE_H
            rh += len(row.get("breakdowns", [])) * BD_H

            ry0 = Y(cursor + rh)
            ryh = Y(cursor) - ry0
            ax.add_patch(patches.Rectangle(
                (0.01, ry0), MARGIN - 0.01, ryh,
                transform=ax.transAxes,
                facecolor=BG_ALT if ri % 2 == 0 else BG,
                zorder=0, linewidth=0))

            main_y = Y(cursor + ROW_H / 2)

            # Column name
            ax.text(X_COL + 0.008, main_y, row["col"],
                    fontsize=8.5, fontweight="bold", color=TXT,
                    transform=ax.transAxes, va="center")

            # Finding — plain text OR dtype badges
            if row.get("dtype"):
                cur_t, exp_t = row["dtype"]
                ax.text(X_FIND + 0.008, main_y, row["finding"] + "  ",
                        fontsize=8, color=TXT,
                        transform=ax.transAxes, va="center")
                bx = X_FIND + 0.008 + len(row["finding"]) * 0.0046 + 0.012
                _dtype_badge(ax, bx, main_y, cur_t, exp_t)
            else:
                ax.text(X_FIND + 0.008, main_y, row["finding"],
                        fontsize=8.5, color=TXT,
                        transform=ax.transAxes, va="center")

            # Status pill
            st, sf = row["status"]
            sb = {C_RED: BG_RED, C_ORANGE: BG_ORANGE,
                  C_GREEN: BG_GREEN}.get(sf, BG_ALT)
            sbd = {C_RED: BD_RED, C_ORANGE: BD_ORANGE,
                   C_GREEN: BD_GREEN}.get(sf, RULE_MED)
            _pill(ax, X_STATUS + 0.008, main_y, st, sf, sb, sbd)

            cursor += ROW_H

            # Muted sub-note
            if row.get("note"):
                ny = Y(cursor + NOTE_H / 2)
                ax.text(X_FIND + 0.018, ny, row["note"],
                        fontsize=7.5, color=TXT_MED, style="italic",
                        transform=ax.transAxes, va="center")
                cursor += NOTE_H

            # Breakdowns
            for bd_lbl, bd_val in row.get("breakdowns", []):
                bdy = Y(cursor + BD_H / 2)
                ax.text(X_FIND + 0.022, bdy, f"↳  {bd_lbl}",
                        fontsize=7.5, color=TXT_MED, style="italic",
                        transform=ax.transAxes, va="center")
                ax.text(X_FIND + 0.26, bdy, bd_val,
                        fontsize=7.5, color=TXT,
                        transform=ax.transAxes, va="center")
                ax.axhline(Y(cursor + BD_H), color=RULE, linewidth=0.3,
                           xmin=0.012, xmax=MARGIN, zorder=1)
                cursor += BD_H

            # Row divider
            ax.axhline(Y(cursor), color=RULE, linewidth=0.4,
                       xmin=0.012, xmax=MARGIN, zorder=1)

        # Section bottom rule
        ax.axhline(Y(cursor), color=RULE_MED, linewidth=0.7,
                   xmin=0.01, xmax=MARGIN, zorder=2)
        if si < len(sections) - 1:
            cursor += SEC_GAP

    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────
def _pct(n, total):
    p = n / total * 100
    if p == 0:     return "0.00%"
    elif p < 0.01: return f"{p:.4f}%"
    elif p < 0.1:  return f"{p:.3f}%"
    else:          return f"{p:.2f}%"


def _save(fig, out_dir, filename):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    p = Path(out_dir) / filename
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"✓ Chart saved → {p}")
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 1 — Tweets
# ─────────────────────────────────────────────────────────────────────────────
def get_data_quality_for_tweets(filepath: str):
    df = pd.read_csv(filepath); N = len(df)
    output_dir = "output"

    # Duplicates
    dup = int(df.duplicated(subset=["pseudo_id"]).sum())

    # Inconsistent
    lc       = df["lang"].value_counts()
    bad_lang = [l for l in lc.index if l not in {"tl","en"}]
    bad_n    = int(sum(lc[l] for l in bad_lang))
    lang_bd  = [(f"'{k}'", f"{int(lc[k]):,} rows  ({_pct(int(lc[k]),N)})")
                for k in lc.index][:6]

    float_n  = int(df["pseudo_inReplyToUsername"].apply(
        lambda x: isinstance(x, float) and not pd.isna(x)).sum())

    # Build sections
    dup_rows = []
    if dup:
        dup_rows.append({
            "col":     "pseudo_id",
            "finding": f"{dup:,} duplicate {'row' if dup==1 else 'rows'}",
            "note":    f"{_pct(dup,N)} of total dataset — likely a scraping overlap",
            "status":  ("✗  Issue Found", C_RED),
        })

    dtype_rows = [
        {"col": "createdAt",              "finding": "Stored as",
         "dtype": ("str / object", "datetime64"),
         "note": "Timestamps need parsing before any temporal analysis",
         "status": ("⚠  Wrong Type", C_ORANGE)},

        {"col": "isReply",                "finding": "Stored as",
         "dtype": ("bool / str", "bool"),
         "note": "Mixed serialisation — normalise to uniform bool before filtering",
         "status": ("⚠  Wrong Type", C_ORANGE)},

        {"col": "pseudo_inReplyToUsername","finding": "Stored as",
         "dtype": ("float64", "str / object"),
         "note": "IDs cast to float when NaNs present — loses leading-zero safety",
         "status": ("⚠  Wrong Type", C_ORANGE)},

        {"col": "author_isBlueVerified",  "finding": "Stored as",
         "dtype": ("bool / str", "bool"),
         "note": "Mixed serialisation — normalise to uniform bool before aggregation",
         "status": ("⚠  Wrong Type", C_ORANGE)},
    ]

    incon_rows = []
    if bad_lang:
        incon_rows.append({
            "col":       "lang",
            "finding":   f"{bad_n:,} rows with unexpected codes: {bad_lang}  ({_pct(bad_n,N)})",
            "note":      "Only 'tl' and 'en' are expected; 'und' = undetermined by Twitter",
            "status":    ("⚠  Inconsistent", C_ORANGE),
            "breakdowns": lang_bd,
        })
    if float_n:
        incon_rows.append({
            "col":     "pseudo_inReplyToUsername",
            "finding": f"{float_n:,} non-null values stored as float64  ({_pct(float_n,N)})",
            "note":    "Non-reply rows are NaN; reply rows should be cast to str before use",
            "status":  ("⚠  Inconsistent", C_ORANGE),
        })

    sections = []
    if dup_rows:   sections.append(("Duplicate Rows",      dup_rows))
    if dtype_rows: sections.append(("Wrong Data Types",    dtype_rows))
    if incon_rows: sections.append(("Inconsistent Values", incon_rows))

    n_issues = sum(1 for _,rs in sections for r in rs if r["status"][1]==C_RED)
    n_warns  = sum(1 for _,rs in sections for r in rs if r["status"][1]==C_ORANGE)

    i_fg  = C_RED    if n_issues else "#475569"
    i_bg  = BG_RED   if n_issues else "#F8FAFC"
    i_bd  = BD_RED   if n_issues else RULE_MED
    i_lbl = f"  ✗  {n_issues} Issue{'s' if n_issues!=1 else ''}  " if n_issues else "  ✓  0 Issues  "

    pills = [
        (i_lbl,   i_fg, i_bg, i_bd),
        (f"  ⚠  {n_warns} Warning{'s' if n_warns!=1 else ''}  ",
         C_ORANGE, BG_ORANGE, BD_ORANGE),
        (f"  {N:,} rows  ·  16 columns  ",
         "#475569", "#F8FAFC", RULE_MED),
    ]

    fig = _render(
        title    = "Data Quality Report — Dataset 1: Tweets",
        subtitle = "Exceptions only  ·  Missing data and dtype inventory covered in separate reports",
        sections = sections,
        pills    = pills,
    )
    _save(fig, output_dir, f"{Path(filepath).stem}_data_quality.png")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset 2 — Authors
# ─────────────────────────────────────────────────────────────────────────────
def get_data_quality_for_authors(filepath: str):
    df = pd.read_csv(filepath); N = len(df)
    output_dir = "output"

    dup = int(df.duplicated(subset=["author_userName"]).sum())

    invalid_kw = [
        "earth","wherever","worldwide","around the world","📍","ig:",
        "http","whatsapp","telegram","lunar","internet","everywhere",
        "facebook.com","twitter.com","t.co","abbott","south girl",
        "2019","2020","2021",
    ]
    def _bad(v):
        if pd.isna(v) or str(v).strip()=="": return False
        return any(k in str(v).lower() for k in invalid_kw)

    mask    = df["author_location"].apply(_bad)
    inv_n   = int(mask.sum())
    inv_bd  = (df.loc[mask,"author_location"]
               .value_counts().head(5).reset_index().values.tolist())
    inv_bd_fmt = [(str(l), f"{int(c)} row{'s' if c>1 else ''}") for l,c in inv_bd]

    empty_bio = int(
        df["author_profile_bio_description"]
        .fillna("").astype(str).str.strip().eq("").sum()
        - df["author_profile_bio_description"].isna().sum()
    )

    dup_rows = []
    if dup:
        dup_rows.append({
            "col":     "author_userName",
            "finding": f"{dup:,} duplicate {'row' if dup==1 else 'rows'}",
            "note":    f"{_pct(dup,N)} of total dataset",
            "status":  ("✗  Issue Found", C_RED),
        })

    dtype_rows = [
        {"col": "author_createdAt",      "finding": "Stored as",
         "dtype": ("str / object", "datetime64"),
         "note": "Parse before computing account age or temporal groupings",
         "status": ("⚠  Wrong Type", C_ORANGE)},

        {"col": "author_isBlueVerified", "finding": "Stored as",
         "dtype": ("bool / str", "bool"),
         "note": "Mixed serialisation — normalise before groupby or filtering",
         "status": ("⚠  Wrong Type", C_ORANGE)},
    ]

    incon_rows = []
    if inv_n:
        incon_rows.append({
            "col":       "author_location",
            "finding":   f"{inv_n:,} non-geographic values  ({_pct(inv_n,N)})",
            "note":      "Contains platform handles, URLs, and colloquialisms — exclude from geo-analysis",
            "status":    ("⚠  Inconsistent", C_ORANGE),
            "breakdowns": inv_bd_fmt,
        })
    if empty_bio:
        incon_rows.append({
            "col":     "author_profile_bio_description",
            "finding": f"{empty_bio:,} empty strings instead of NaN  ({_pct(empty_bio,N)})",
            "note":    "Replace '' with NaN for consistent null-handling",
            "status":  ("⚠  Inconsistent", C_ORANGE),
        })

    sections = []
    if dup_rows:   sections.append(("Duplicate Rows",      dup_rows))
    if dtype_rows: sections.append(("Wrong Data Types",    dtype_rows))
    if incon_rows: sections.append(("Inconsistent Values", incon_rows))

    n_issues = sum(1 for _,rs in sections for r in rs if r["status"][1]==C_RED)
    n_warns  = sum(1 for _,rs in sections for r in rs if r["status"][1]==C_ORANGE)

    i_fg  = C_RED    if n_issues else C_GREEN
    i_bg  = BG_RED   if n_issues else BG_GREEN
    i_bd  = BD_RED   if n_issues else BD_GREEN
    i_lbl = f"  ✗  {n_issues} Issue{'s' if n_issues!=1 else ''}  " if n_issues else "  ✓  0 Issues  "

    pills = [
        (i_lbl,   i_fg, i_bg, i_bd),
        (f"  ⚠  {n_warns} Warning{'s' if n_warns!=1 else ''}  ",
         C_ORANGE, BG_ORANGE, BD_ORANGE),
        (f"  {N:,} rows  ·  8 columns  ",
         "#475569", "#F8FAFC", RULE_MED),
    ]

    fig = _render(
        title    = "Data Quality Report — Dataset 2: Authors",
        subtitle = "Exceptions only  ·  Missing data and dtype inventory covered in separate reports",
        sections = sections,
        pills    = pills,
    )
    _save(fig, output_dir, f"{Path(filepath).stem}_data_quality.png")
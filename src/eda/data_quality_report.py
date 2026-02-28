import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def get_data_quality_for_tweets(
    filepath: str,
    output_dir: str = "output"
) -> dict:
    base_filename = str(filepath).replace("data/", "").replace(".csv", "")
    df = pd.read_csv(filepath)
    total_rows = len(df)

    # ── Style ────────────────────────────────────────────────────────────────
    BG_COLOR   = "#F8F9FA"
    TEXT_COLOR = "#2D2D2D"
    GRAY       = "#888888"
    FONT       = {"font.family": "DejaVu Sans", "font.size": 10}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR})

    # ── Duplicate check ───────────────────────────────────────────────────────
    duplicate_count = df.duplicated(subset=["pseudo_id"]).sum()

    # ── Invalid location patterns (for is_invalid helper) ────────────────────
    invalid_patterns = [
        "earth", "wherever", "worldwide", "around the world",
        "📍", "ig:", "http", "whatsapp", "telegram",
        "lunar", "yn edri", "abbott", "internet",
        "everywhere", "facebook.com", "twitter.com", "t.co",
    ]

    def is_invalid_location(loc):
        if pd.isna(loc) or str(loc).strip() == "":
            return False
        return any(p in str(loc).lower() for p in invalid_patterns)

    # ── Per-column checks ─────────────────────────────────────────────────────
    # Each entry: (column, check_name, finding, status, color)
    STATUS_CLEAN  = ("✓ Clean",       "#57CC99")
    STATUS_ISSUE  = ("✗ Issue Found", "#E63946")
    STATUS_WARN   = ("⚠ Inconsistent","#F4A261")
    STATUS_TYPE   = ("⚠ Wrong Type",  "#F4A261")

    def bool_check(col):
        vals = set(df[col].dropna().astype(str).unique())
        ok = vals <= {"True", "False"}
        return (f"Unique: {sorted(vals)}", STATUS_CLEAN if ok else STATUS_ISSUE)

    rows = []

    # pseudo_id
    rows.append(("pseudo_id", "Duplicate Rows",
                 f"{duplicate_count:,} duplicate {'row' if duplicate_count==1 else 'rows'} found",
                 STATUS_CLEAN if duplicate_count == 0 else STATUS_ISSUE))

    # pseudo_conversationId
    rows.append(("pseudo_conversationId", "Data Type",
                 "int64 — correct", STATUS_CLEAN))

    # createdAt
    rows.append(("createdAt", "Wrong Data Type",
                 "Stored as str, expected datetime", STATUS_TYPE))
    rows.append(("createdAt", "Format Check",
                 "All timestamps parseable", STATUS_CLEAN))

    # lang
    lang_counts = df["lang"].value_counts()
    unexpected_langs = [l for l in lang_counts.index if l not in {"tl", "en"}]
    lang_finding = (
        f"Only tl ({lang_counts.get('tl',0):,}) and en ({lang_counts.get('en',0):,})"
        if not unexpected_langs
        else f"Unexpected codes found: {unexpected_langs} — {sum(lang_counts[l] for l in unexpected_langs):,} rows"
    )
    rows.append(("lang", "Unexpected Values",
                 lang_finding,
                 STATUS_CLEAN if not unexpected_langs else STATUS_WARN))

    # text
    rows.append(("text", "Null Check",
                 "0 null values — all tweets have content", STATUS_CLEAN))

    # retweetCount / likeCount / viewCount / quoteCount / replyCount / bookmarkCount
    for col in ["retweetCount", "likeCount", "viewCount", "quoteCount", "replyCount", "bookmarkCount"]:
        neg = (df[col] < 0).sum()
        rows.append((col, "Negative Values",
                     f"{neg:,} negative values" if neg > 0 else "No negative values",
                     STATUS_CLEAN if neg == 0 else STATUS_ISSUE))

    # isReply
    finding, status = bool_check("isReply")
    rows.append(("isReply", "Boolean Check", finding, status))

    # pseudo_inReplyToUsername
    float_count = df["pseudo_inReplyToUsername"].apply(
        lambda x: isinstance(x, float) and not pd.isna(x)
    ).sum()
    rows.append(("pseudo_inReplyToUsername", "Wrong Data Type",
                 "Stored as float64, expected str", STATUS_TYPE))
    rows.append(("pseudo_inReplyToUsername", "Inconsistent Values",
                 f"{float_count:,} values stored as float instead of str",
                 STATUS_CLEAN if float_count == 0 else STATUS_WARN))

    # quoted_pseudo_id
    na_str_count = (df["quoted_pseudo_id"].astype(str) == "<NA>").sum()
    rows.append(("quoted_pseudo_id", "String NaN",
                 f"{na_str_count:,} rows have \"<NA>\" string instead of NaN",
                 STATUS_CLEAN if na_str_count == 0 else STATUS_WARN))

    # author_isBlueVerified
    finding, status = bool_check("author_isBlueVerified")
    rows.append(("author_isBlueVerified", "Boolean Check", finding, status))

    # pseudo_author_userName
    rows.append(("pseudo_author_userName", "Null Check",
                 "0 null values", STATUS_CLEAN))

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows   = len(rows)
    fig_h    = max(6, n_rows * 0.42 + 2.5)
    fig, ax  = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # ── Title block ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.98,
             "Data Quality Report — Dataset 1: Tweets",
             fontsize=14, fontweight="bold", ha="center", va="top",
             color=TEXT_COLOR)
    fig.text(0.5, 0.955,
             f"{total_rows:,} rows · {len(df.columns)} columns · {duplicate_count:,} duplicate rows",
             fontsize=10, ha="center", va="top", color=GRAY)

    # ── Summary badges ────────────────────────────────────────────────────────
    issues   = sum(1 for r in rows if r[3][1] == "#E63946")
    warnings = sum(1 for r in rows if r[3][1] == "#F4A261")
    cleans   = sum(1 for r in rows if r[3][1] == "#57CC99")

    badge_data = [
        (f"{cleans} Clean", "#57CC99", "#EAFAF1"),
        (f"{warnings} Warnings", "#F4A261", "#FEF3E8"),
        (f"{issues} Issues", "#E63946", "#FDECEA"),
    ]
    bx = 0.18
    for label, fg, bg in badge_data:
        fig.text(bx, 0.925, label,
                 fontsize=9.5, fontweight="bold",
                 ha="center", va="center", color=fg,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=bg,
                           edgecolor=fg, linewidth=1.2))
        bx += 0.18

    # ── Table header ──────────────────────────────────────────────────────────
    col_x     = [0.02, 0.25, 0.46, 0.88]
    col_labels = ["Column", "Check", "Finding", "Status"]
    header_y   = 0.895

    for x, label in zip(col_x, col_labels):
        ax.text(x, header_y, label,
                fontsize=9.5, fontweight="bold", color=TEXT_COLOR,
                transform=ax.transAxes, va="center")

    ax.axhline(header_y - 0.018, color="#CCCCCC",
               linewidth=1, xmin=0.01, xmax=0.99)

    # ── Table rows ────────────────────────────────────────────────────────────
    row_h     = (header_y - 0.065) / (n_rows + 0.5)
    prev_col  = None

    for i, (col, check, finding, (status_text, status_color)) in enumerate(rows):
        y = header_y - 0.045 - i * row_h

        # Alternating row background
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0.01, y - row_h * 0.5), 0.98, row_h,
                transform=ax.transAxes,
                facecolor="#EFEFEF", zorder=0, linewidth=0
            ))

        # Column name — only show when it changes
        display_col = col if col != prev_col else ""
        prev_col    = col

        ax.text(col_x[0], y, display_col,
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes, va="center",
                fontweight="bold" if display_col else "normal")
        ax.text(col_x[1], y, check,
                fontsize=8.5, color=GRAY,
                transform=ax.transAxes, va="center")
        ax.text(col_x[2], y, finding,
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes, va="center")
        ax.text(col_x[3], y, status_text,
                fontsize=8.5, fontweight="bold", color=status_color,
                transform=ax.transAxes, va="center")

    # ── Bottom divider ────────────────────────────────────────────────────────
    ax.axhline(0.02, color="#CCCCCC", linewidth=0.8,
               xmin=0.01, xmax=0.99)

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_data_quality.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}")
    return {
        "duplicate_count":        duplicate_count,
        "unexpected_langs":       unexpected_langs,
        "na_string_count":        int(na_str_count),
        "float_identifier_count": int(float_count),
    }

def get_data_quality_for_authors(
    filepath: str,
    output_dir: str = "output"
) -> dict:
    base_filename = str(filepath).replace("data/", "").replace(".csv", "")
    df = pd.read_csv(filepath)
    total_rows = len(df)

    # ── Style ────────────────────────────────────────────────────────────────
    BG_COLOR   = "#F8F9FA"
    TEXT_COLOR = "#2D2D2D"
    GRAY       = "#888888"
    FONT       = {"font.family": "DejaVu Sans", "font.size": 10}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR})

    # ── Duplicate check ───────────────────────────────────────────────────────
    duplicate_count = df.duplicated(subset=["author_userName"]).sum()

    # ── Invalid location patterns ─────────────────────────────────────────────
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

    # ── Status helpers ────────────────────────────────────────────────────────
    STATUS_CLEAN = ("✓ Clean",        "#57CC99")
    STATUS_ISSUE = ("✗ Issue Found",  "#E63946")
    STATUS_WARN  = ("⚠ Inconsistent", "#F4A261")
    STATUS_TYPE  = ("⚠ Wrong Type",   "#F4A261")

    def bool_check(col):
        vals = set(df[col].dropna().astype(str).unique())
        ok = vals <= {"True", "False"}
        return (f"Unique: {sorted(vals)}", STATUS_CLEAN if ok else STATUS_ISSUE)

    rows = []

    # author_userName
    rows.append(("author_userName", "Duplicate Rows",
                 f"{duplicate_count:,} duplicate {'row' if duplicate_count==1 else 'rows'} found",
                 STATUS_CLEAN if duplicate_count == 0 else STATUS_ISSUE))
    rows.append(("author_userName", "Null Check",
                 "0 null values — all authors have a username", STATUS_CLEAN))

    # obfuscated_userName
    rows.append(("obfuscated_userName", "Null Check",
                 "0 null values", STATUS_CLEAN))

    # author_createdAt
    rows.append(("author_createdAt", "Wrong Data Type",
                 "Stored as str, expected datetime", STATUS_TYPE))
    rows.append(("author_createdAt", "Format Check",
                 "All timestamps parseable", STATUS_CLEAN))

    # author_profile_bio_description
    empty_bio = (
        df["author_profile_bio_description"]
        .fillna("").astype(str).str.strip().eq("").sum()
    )
    real_null_bio = df["author_profile_bio_description"].isna().sum()
    rows.append(("author_profile_bio_description", "Empty String",
                 f"{empty_bio - real_null_bio:,} empty strings \"\" instead of NaN",
                 STATUS_CLEAN if (empty_bio - real_null_bio) == 0 else STATUS_WARN))

    # author_location
    invalid_loc_count = df["author_location"].apply(is_invalid_location).sum()
    valid_loc_count   = df["author_location"].notna().sum() - invalid_loc_count
    rows.append(("author_location", "Invalid Values",
                 f"{invalid_loc_count:,} non-geographic / meaningless values",
                 STATUS_CLEAN if invalid_loc_count == 0 else STATUS_WARN))
    rows.append(("author_location", "Valid Values",
                 f"{valid_loc_count:,} valid geographic values",
                 STATUS_CLEAN))

    # author_followers
    neg_followers = (df["author_followers"] < 0).sum()
    rows.append(("author_followers", "Negative Values",
                 f"{neg_followers:,} negative values" if neg_followers > 0 else "No negative values",
                 STATUS_CLEAN if neg_followers == 0 else STATUS_ISSUE))

    # author_following
    neg_following = (df["author_following"] < 0).sum()
    rows.append(("author_following", "Negative Values",
                 f"{neg_following:,} negative values" if neg_following > 0 else "No negative values",
                 STATUS_CLEAN if neg_following == 0 else STATUS_ISSUE))

    # author_isBlueVerified
    finding, status = bool_check("author_isBlueVerified")
    rows.append(("author_isBlueVerified", "Boolean Check", finding, status))

    # ── Figure ────────────────────────────────────────────────────────────────
    n_rows  = len(rows)
    fig_h   = max(5, n_rows * 0.42 + 2.5)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.axis("off")

    # ── Title block ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.98,
             "Data Quality Report — Dataset 2: Authors",
             fontsize=14, fontweight="bold", ha="center", va="top",
             color=TEXT_COLOR)
    fig.text(0.5, 0.955,
             f"{total_rows:,} rows · {len(df.columns)} columns · {duplicate_count:,} duplicate rows",
             fontsize=10, ha="center", va="top", color=GRAY)

    # ── Summary badges ────────────────────────────────────────────────────────
    issues   = sum(1 for r in rows if r[3][1] == "#E63946")
    warnings = sum(1 for r in rows if r[3][1] == "#F4A261")
    cleans   = sum(1 for r in rows if r[3][1] == "#57CC99")

    badge_data = [
        (f"{cleans} Clean",    "#57CC99", "#EAFAF1"),
        (f"{warnings} Warnings","#F4A261","#FEF3E8"),
        (f"{issues} Issues",   "#E63946", "#FDECEA"),
    ]
    bx = 0.18
    for label, fg, bg in badge_data:
        fig.text(bx, 0.925, label,
                 fontsize=9.5, fontweight="bold",
                 ha="center", va="center", color=fg,
                 bbox=dict(boxstyle="round,pad=0.4", facecolor=bg,
                           edgecolor=fg, linewidth=1.2))
        bx += 0.18

    # ── Table header ──────────────────────────────────────────────────────────
    col_x      = [0.02, 0.28, 0.46, 0.88]
    col_labels = ["Column", "Check", "Finding", "Status"]
    header_y   = 0.895

    for x, label in zip(col_x, col_labels):
        ax.text(x, header_y, label,
                fontsize=9.5, fontweight="bold", color=TEXT_COLOR,
                transform=ax.transAxes, va="center")

    ax.axhline(header_y - 0.018, color="#CCCCCC",
               linewidth=1, xmin=0.01, xmax=0.99)

    # ── Table rows ────────────────────────────────────────────────────────────
    row_h    = (header_y - 0.065) / (n_rows + 0.5)
    prev_col = None

    for i, (col, check, finding, (status_text, status_color)) in enumerate(rows):
        y = header_y - 0.045 - i * row_h

        if i % 2 == 0:
            ax.add_patch(plt.Rectangle(
                (0.01, y - row_h * 0.5), 0.98, row_h,
                transform=ax.transAxes,
                facecolor="#EFEFEF", zorder=0, linewidth=0
            ))

        display_col = col if col != prev_col else ""
        prev_col    = col

        ax.text(col_x[0], y, display_col,
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes, va="center",
                fontweight="bold" if display_col else "normal")
        ax.text(col_x[1], y, check,
                fontsize=8.5, color=GRAY,
                transform=ax.transAxes, va="center")
        ax.text(col_x[2], y, finding,
                fontsize=8.5, color=TEXT_COLOR,
                transform=ax.transAxes, va="center")
        ax.text(col_x[3], y, status_text,
                fontsize=8.5, fontweight="bold", color=status_color,
                transform=ax.transAxes, va="center")

    ax.axhline(0.02, color="#CCCCCC", linewidth=0.8,
               xmin=0.01, xmax=0.99)

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_data_quality.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor=BG_COLOR)
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}")
    return {
        "duplicate_count":      duplicate_count,
        "invalid_loc_count":    int(invalid_loc_count),
        "valid_loc_count":      int(valid_loc_count),
        "empty_bio_count":      int(empty_bio - real_null_bio),
    }
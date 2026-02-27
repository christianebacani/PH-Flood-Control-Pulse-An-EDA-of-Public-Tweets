import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

def get_data_quality_for_authors(
        filepath: str,
        output_dir: str = "output"
) -> dict:
    base_filename = str(filepath).replace("data/", "").replace(".csv", "")
    df = pd.read_csv(filepath)

    # ── Style ────────────────────────────────────────────────────────────────
    BG_COLOR   = "#F8F9FA"
    TEXT_COLOR = "#2D2D2D"
    FONT       = {"font.family": "DejaVu Sans", "font.size": 11}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR})

    # ── Findings ─────────────────────────────────────────────────────────────

    # 1. Duplicate count
    duplicate_count = df.duplicated(subset=["author_userName"]).sum()

    # 2. author_location — classify valid vs invalid
    invalid_patterns = [
        "earth", "wherever", "worldwide", "around the world",
        "est.", "📍", "ig:", "http", "whatsapp", "telegram",
        "2020", "2021", "2019", "lunar", "yn edri", "abbott",
        "south girl", "vulcanizing", "menlo park", "internet",
        "everywhere", "ph 🇵🇭", "mnl |", "in your",
        "facebook.com", "twitter.com", "t.co",
    ]

    def is_invalid_location(loc):
        if pd.isna(loc) or str(loc).strip() == "":
            return False  # already counted as missing
        loc_lower = str(loc).lower()
        return any(pattern in loc_lower for pattern in invalid_patterns)

    location_counts      = df["author_location"].value_counts().head(20)
    invalid_location_count = df["author_location"].apply(is_invalid_location).sum()
    valid_location_count   = df["author_location"].notna().sum() - invalid_location_count

    # 3. author_profile_bio_description — empty strings
    empty_bio_count = (df["author_profile_bio_description"].astype(str).str.strip() == "").sum()
    empty_bio_count = (
        df["author_profile_bio_description"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )
    # 4. author_isBlueVerified unique values
    verified_values = df["author_isBlueVerified"].unique().tolist()

    # 5. author_createdAt format check
    try:
        pd.to_datetime(df["author_createdAt"])
        datetime_format_ok = True
        datetime_issues    = 0
    except Exception:
        datetime_format_ok = False
        datetime_issues    = df["author_createdAt"].apply(
            lambda x: pd.to_datetime(x, errors="coerce")
        ).isna().sum()

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(17, max(5, len(location_counts) * 0.45 + 2)),
        gridspec_kw={"width_ratios": [1.2, 1.4]}
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Data Quality Report", fontsize=15, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)

    # ── LEFT: Author location bar chart ───────────────────────────────────────
    ax_bar.set_facecolor(BG_COLOR)
    ax_bar.set_title("Top 20 Author Locations", fontsize=12,
                     fontweight="bold", pad=12, color=TEXT_COLOR)
    ax_bar.text(0.5, 1.00,
                f"{invalid_location_count} non-geographic values detected",
                transform=ax_bar.transAxes,
                ha="center", fontsize=10, color="#888888")

    bar_colors = ["#F07167" if is_invalid_location(loc) else "#57CC99"
                  for loc in location_counts.index]

    bars = ax_bar.barh(
        location_counts.index[::-1],
        location_counts.values[::-1],
        color=bar_colors[::-1],
        edgecolor="white", linewidth=1.2, zorder=3
    )

    ax_bar.xaxis.grid(True, color="#DDDDDD", linestyle="--",
                      linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right", "bottom"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color("#CCCCCC")
    ax_bar.tick_params(colors=TEXT_COLOR, length=0)
    ax_bar.set_xlabel("Author Count", fontsize=11, labelpad=8)

    for bar, count in zip(bars, location_counts.values[::-1]):
        ax_bar.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center", ha="left", fontsize=9.5,
            fontweight="bold", color=TEXT_COLOR
        )

    legend_items = [
        mpatches.Patch(color="#57CC99", label="Valid location"),
        mpatches.Patch(color="#F07167", label="Non-geographic / meaningless value"),
    ]
    ax_bar.legend(handles=legend_items, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08),
                  ncol=2, frameon=False, fontsize=10)

    # ── RIGHT: Data Quality Findings table ────────────────────────────────────
    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis("off")
    ax_table.set_title("Data Quality Findings", fontsize=12,
                       fontweight="bold", pad=12, color=TEXT_COLOR)

    def truncate(text, max_len=22):
        return text if len(text) <= max_len else text[:max_len - 1] + "…"

    findings = [
        {
            "check":   "Duplicate Rows",
            "column":  truncate("author_userName"),
            "finding": f"{duplicate_count:,} duplicate rows",
            "status":  "✓ Clean" if duplicate_count == 0 else "✗ Issue Found",
            "color":   "#57CC99" if duplicate_count == 0 else "#F07167",
        },
        {
            "check":   "Invalid Locations",
            "column":  truncate("author_location"),
            "finding": f"{invalid_location_count:,} non-geographic values",
            "status":  "✓ Clean" if invalid_location_count == 0 else "⚠ Inconsistent",
            "color":   "#57CC99" if invalid_location_count == 0 else "#F4A261",
        },
        {
            "check":   "Valid Locations",
            "column":  truncate("author_location"),
            "finding": f"{valid_location_count:,} valid geographic values",
            "status":  "✓ Clean",
            "color":   "#57CC99",
        },
        {
            "check":   "Empty String Bio",
            "column":  truncate("author_profile_bio_description"),
            "finding": f"{empty_bio_count} empty strings (should be NaN)",
            "status":  "✓ Clean" if empty_bio_count == 0 else "⚠ Inconsistent",
            "color":   "#57CC99" if empty_bio_count == 0 else "#F4A261",
        },
        {
            "check":   "Boolean Values",
            "column":  truncate("author_isBlueVerified"),
            "finding": f"Unique values: {verified_values}",
            "status":  "✓ Clean" if set(map(str, verified_values)) <= {"True", "False"} else "✗ Issue Found",
            "color":   "#57CC99" if set(map(str, verified_values)) <= {"True", "False"} else "#F07167",
        },
        {
            "check":   "DateTime Format",
            "column":  truncate("author_createdAt"),
            "finding": "All timestamps valid" if datetime_format_ok else f"{datetime_issues:,} malformed timestamps",
            "status":  "✓ Clean" if datetime_format_ok else "✗ Issue Found",
            "color":   "#57CC99" if datetime_format_ok else "#F07167",
        },
        {
            "check":   "Wrong Data Type",
            "column":  truncate("author_createdAt"),
            "finding": "Stored as str, should be datetime",
            "status":  "⚠ Wrong Type",
            "color":   "#F4A261",
        },
    ]

    n          = len(findings)
    row_height = 1.0 / (n + 1.5)

    headers = ["Check", "Column", "Finding", "Status"]
    x_pos   = [0.02, 0.18, 0.42, 0.82]
    y_header = 0.97

    for header, x in zip(headers, x_pos):
        ax_table.text(x, y_header, header,
                      fontsize=10, fontweight="bold", color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")

    ax_table.axhline(y=y_header - 0.03, color="#CCCCCC",
                     linewidth=1)

    for i, f in enumerate(findings):
        y = y_header - 0.06 - i * row_height

        if i % 2 == 0:
            ax_table.add_patch(plt.Rectangle(
                (0, y - row_height * 0.4), 1, row_height,
                transform=ax_table.transAxes,
                color="#EEEEEE", zorder=0
            ))

        ax_table.text(x_pos[0], y, f["check"],
                      fontsize=8.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")
        ax_table.text(x_pos[1], y, f["column"],
                      fontsize=8.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")
        ax_table.text(x_pos[2], y, f["finding"],
                      fontsize=8.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")
        ax_table.text(x_pos[3], y, f["status"],
                      fontsize=8.5, fontweight="bold", color=f["color"],
                      transform=ax_table.transAxes, va="center", ha="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{base_filename}_data_quality.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {output_path}")
    return {
        "duplicate_count":          duplicate_count,
        "invalid_location_count":   invalid_location_count,
        "valid_location_count":     valid_location_count,
        "empty_bio_count":          empty_bio_count,
        "datetime_format_ok":       datetime_format_ok,
    }

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
    FONT       = {"font.family": "DejaVu Sans", "font.size": 11}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR})

    # ── Findings ─────────────────────────────────────────────────────────────

    # 1. Duplicate count
    duplicate_count = df.duplicated(subset=["pseudo_id"]).sum()

    # 2. Lang unique values
    lang_counts = df["lang"].value_counts()

    # 3. quoted_pseudo_id — count "<NA>" strings vs real NaN
    quoted_na_string_count = (df["quoted_pseudo_id"].astype(str) == "<NA>").sum()
    quoted_real_nan_count  = df["quoted_pseudo_id"].isna().sum()

    # 4. pseudo_inReplyToUsername — check float dtype issue
    float_count = df["pseudo_inReplyToUsername"].apply(
        lambda x: isinstance(x, float) and not pd.isna(x)
    ).sum()

    # 5. isReply unique values
    isreply_values = df["isReply"].unique().tolist()

    # 6. author_isBlueVerified unique values
    verified_values = df["author_isBlueVerified"].unique().tolist()

    # 7. createdAt format check
    try:
        pd.to_datetime(df["createdAt"])
        datetime_format_ok = True
        datetime_issues    = 0
    except Exception:
        datetime_format_ok = False
        datetime_issues    = df["createdAt"].apply(
            lambda x: pd.to_datetime(x, errors="coerce")
        ).isna().sum()

    # ── Figure: two panels ────────────────────────────────────────────────────
    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(17, max(4, len(lang_counts) * 0.45 + 2)),
        gridspec_kw={"width_ratios": [1.2, 1.4]}
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Data Quality Report", fontsize=15, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)

    # ── LEFT: Lang distribution bar chart ─────────────────────────────────────
    ax_bar.set_facecolor(BG_COLOR)
    ax_bar.set_title("Language Distribution", fontsize=12,
                     fontweight="bold", pad=12, color=TEXT_COLOR)
    ax_bar.text(0.5, 1.00,
                f"{len(lang_counts)} unique language codes found",
                transform=ax_bar.transAxes,
                ha="center", fontsize=10, color="#888888")

    expected_langs = {"tl", "en"}

    langs_ordered  = lang_counts.index[::-1].tolist()
    values_ordered = lang_counts.values[::-1].tolist()
    max_val = max(values_ordered)
    min_display = max_val * 0.008  # minimum 0.8% of max for visibility
    values_display = [max(v, min_display) for v in values_ordered]

    colors_ordered = ["#57CC99" if lang in expected_langs else "#F07167"
                      for lang in langs_ordered]

    bars = ax_bar.barh(langs_ordered,
                       values_display,
                       color=colors_ordered,
                       edgecolor="white", linewidth=1.2, zorder=3)
    
    ax_bar.set_xlim(0, max(values_ordered) * 1.35)  # ← add here
    ax_bar.xaxis.grid(True, color="#DDDDDD", linestyle="--",
                      linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right", "bottom"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color("#CCCCCC")
    ax_bar.tick_params(colors=TEXT_COLOR, length=0)
    ax_bar.set_xlabel("Tweet Count", fontsize=11, labelpad=8)

    xlim_max = max(values_ordered) * 1.35
    for bar, count, lang in zip(bars, values_ordered, langs_ordered):
        pct = count / total_rows * 100
        label_text = f"{count:,}  ({pct:.1f}%)"
        bar_end = bar.get_width()

        if bar_end < xlim_max * 0.08:
            # tiny bar — place label to the right at fixed readable position
            ax_bar.text(
                xlim_max * 0.08,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                va="center", ha="left", fontsize=9.5, fontweight="bold",
                color=TEXT_COLOR
                )
        else:
            # normal bar — place label just after bar end
            ax_bar.text(
                bar_end + xlim_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label_text,
                va="center", ha="left", fontsize=9.5, fontweight="bold",
                color=TEXT_COLOR
                )

    legend_items = [
        mpatches.Patch(color="#57CC99", label="Expected  (tl, en)"),
        mpatches.Patch(color="#F07167", label="Unexpected language code"),
    ]
    ax_bar.legend(handles=legend_items, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12),
                  ncol=2, frameon=False, fontsize=10)

    # ── RIGHT: Data Quality Findings table ────────────────────────────────────
    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis("off")
    ax_table.set_title("Data Quality Findings", fontsize=12,
                       fontweight="bold", pad=12, color=TEXT_COLOR)

    def truncate(text, max_len=22):
        return text if len(text) <= max_len else text[:max_len - 1] + "…"

    findings = [
        {
            "check":   "Duplicate Rows",
            "column":  truncate("pseudo_id"),
            "finding": f"{duplicate_count:,} duplicate rows",
            "status":  "✓ Clean" if duplicate_count == 0 else "✗ Issue Found",
            "color":   "#57CC99" if duplicate_count == 0 else "#F07167",
        },
        {
            "check":   "String NaN",
            "column":  truncate("quoted_pseudo_id"),
            "finding": f"{quoted_na_string_count:,} rows have \"<NA>\" string",
            "status":  "✓ Clean" if quoted_na_string_count == 0 else "⚠ Inconsistent",
            "color":   "#57CC99" if quoted_na_string_count == 0 else "#F4A261",
        },
        {
            "check":   "Float Identifiers",
            "column":  truncate("pseudo_inReplyToUsername"),
            "finding": f"{float_count:,} values stored as float",
            "status":  "✓ Clean" if float_count == 0 else "⚠ Inconsistent",
            "color":   "#57CC99" if float_count == 0 else "#F4A261",
        },
        {
            "check":   "Boolean Values",
            "column":  truncate("isReply"),
            "finding": f"Unique values: {isreply_values}",
            "status":  "✓ Clean" if set(map(str, isreply_values)) <= {"True", "False"} else "✗ Issue Found",
            "color":   "#57CC99" if set(map(str, isreply_values)) <= {"True", "False"} else "#F07167",
        },
        {
            "check":   "Boolean Values",
            "column":  truncate("author_isBlueVerified"),
            "finding": f"Unique values: {verified_values}",
            "status":  "✓ Clean" if set(map(str, verified_values)) <= {"True", "False"} else "✗ Issue Found",
            "color":   "#57CC99" if set(map(str, verified_values)) <= {"True", "False"} else "#F07167",
        },
        {
            "check":   "DateTime Format",
            "column":  truncate("createdAt"),
            "finding": "All timestamps valid" if datetime_format_ok else f"{datetime_issues:,} malformed timestamps",
            "status":  "✓ Clean" if datetime_format_ok else "✗ Issue Found",
            "color":   "#57CC99" if datetime_format_ok else "#F07167",
        },
        {
            "check":   "Wrong Data Type",
            "column":  truncate("createdAt"),
            "finding": "Stored as str, should be datetime",
            "status":  "⚠ Wrong Type",
            "color":   "#F4A261",
        },
        {
            "check":   "Wrong Data Type",
            "column":  truncate("pseudo_inReplyToUsername"),
            "finding": "Stored as float64, should be str",
            "status":  "⚠ Wrong Type",
            "color":   "#F4A261",
        },
    ]

    n          = len(findings)
    row_height = 1.0 / (n + 1.5)

    # Header
    headers = ["Check", "Column", "Finding", "Status"]
    x_pos   = [0.02, 0.18, 0.42, 0.82]
    y_header = 0.97

    for header, x in zip(headers, x_pos):
        ax_table.text(x, y_header, header,
                      fontsize=10, fontweight="bold", color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")

    ax_table.axhline(y=y_header - 0.03, color="#CCCCCC",
                     linewidth=1)

    # Rows
    for i, f in enumerate(findings):
        y = y_header - 0.06 - i * row_height

        if i % 2 == 0:
            ax_table.add_patch(plt.Rectangle(
                (0, y - row_height * 0.4), 1, row_height,
                transform=ax_table.transAxes,
                color="#EEEEEE", zorder=0
            ))

        ax_table.text(x_pos[0], y, f["check"],
                      fontsize=8.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")
        ax_table.text(x_pos[1], y, f["column"],
                      fontsize=8.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")
        ax_table.text(x_pos[2], y, f["finding"],
                      fontsize=8.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center", ha="left")
        ax_table.text(x_pos[3], y, f["status"],
                      fontsize=8.5, fontweight="bold", color=f["color"],
                      transform=ax_table.transAxes, va="center", ha="left")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{base_filename}_data_quality.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {output_path}")
    return {
        "duplicate_count":         duplicate_count,
        "lang_counts":             lang_counts.to_dict(),
        "quoted_na_string_count":  quoted_na_string_count,
        "float_identifier_count":  float_count,
        "datetime_format_ok":      datetime_format_ok,
    }
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from zipfile import ZipFile
from pathlib import Path

def extract_files_from_zipfile(filepath: str) -> None:
    # Extract all the files inside the zipfile of data/ directory
    with ZipFile("data/archive.zip", 'r') as zip_file:
        zip_file.extractall("data") # File destination

def count_rows_and_columns(filepath: str, output_dir: str = "output") -> tuple[int, int]:
    base_filename = str(filepath).replace("data/", "")
    base_filename = base_filename.replace(".csv", "")

    df = pd.read_csv(filepath)
    rows, cols = df.shape

    # ── Style ────────────────────────────────────────────────────────────────
    PALETTE    = {"Rows": "#4C9BE8", "Columns": "#F07167"}
    BG_COLOR   = "#F8F9FA"
    TEXT_COLOR = "#2D2D2D"
    FONT       = {"font.family": "DejaVu Sans", "font.size": 11}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR})

    # ── Figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    labels = ["Rows", "Columns"]
    values = [rows, cols]
    colors = [PALETTE[l] for l in labels]

    bars = ax.bar(labels, values, color=colors, width=0.45, zorder=3,
                  edgecolor="white", linewidth=1.2)

    # ── Gridlines (subtle, behind bars) ──────────────────────────────────────
    ax.yaxis.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── Spine cleanup ─────────────────────────────────────────────────────────
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR, length=0)

    # ── Value labels ─────────────────────────────────────────────────────────
    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + max_val * 0.015,
            f"{val:,}",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
            color=bar.get_facecolor(),
        )

    # ── Labels & title ───────────────────────────────────────────────────────
    ax.set_ylabel("Count", fontsize=12, labelpad=10)
    ax.set_title("Dataset Shape Overview", fontsize=15, fontweight="bold",
                 pad=30, color=TEXT_COLOR)
    ax.set_ylim(0, max_val * 1.15)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── Subtitle with ratio ───────────────────────────────────────────────────
    ax.text(0.5, 1.02, f"{rows:,} rows × {cols:,} columns",
            transform=ax.transAxes,
            ha="center", fontsize=10, color="#888888")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_dataset_shape.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}  ({rows:,} rows, {cols:,} cols)")
    return rows, cols

def get_column_names_and_dtypes(filepath: str, output_dir: str = "output") -> pd.Series:
    base_filename = str(filepath).replace("data/", "")
    base_filename = base_filename.replace(".csv", "")

    df = pd.read_csv(filepath)
    column_names_and_dtypes = df.dtypes

    # ── Style ────────────────────────────────────────────────────────────────
    BG_COLOR   = "#F8F9FA"
    TEXT_COLOR = "#2D2D2D"
    DTYPE_COLORS = {
        "object":  "#4C9BE8",
        "str":     "#4C9BE8",   # ← add this, same color as object
        "int64":   "#57CC99",
        "float64": "#F4A261",
        "bool":    "#C77DFF",
        "datetime64[ns]": "#F07167",
    }
    DEFAULT_COLOR = "#AAAAAA"
    FONT = {"font.family": "DejaVu Sans", "font.size": 11}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR})

    # ── Figure: two panels side by side ──────────────────────────────────────
    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(13, max(4, len(df.columns) * 0.45)),
        gridspec_kw={"width_ratios": [1, 1.6]}
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Column Names & Data Types", fontsize=15, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)

    # ── LEFT: dtype distribution bar chart ───────────────────────────────────
    dtype_counts = column_names_and_dtypes.astype(str).value_counts()
    bar_colors   = [DTYPE_COLORS.get(dt, DEFAULT_COLOR) for dt in dtype_counts.index]

    bars = ax_bar.barh(dtype_counts.index, dtype_counts.values,
                       color=bar_colors, edgecolor="white", linewidth=1.2, zorder=3)

    ax_bar.set_facecolor(BG_COLOR)
    ax_bar.xaxis.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.8, zorder=0)
    ax_bar.set_axisbelow(True)
    for spine in ["top", "right", "bottom"]:
        ax_bar.spines[spine].set_visible(False)
    ax_bar.spines["left"].set_color("#CCCCCC")
    ax_bar.tick_params(colors=TEXT_COLOR, length=0)
    ax_bar.set_xlabel("Number of Columns", fontsize=11, labelpad=8)
    ax_bar.set_title("dtype Distribution", fontsize=12, fontweight="bold",
                     pad=12, color=TEXT_COLOR)
    ax_bar.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # Value labels on bars
    for bar, val in zip(bars, dtype_counts.values):
        ax_bar.text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            str(val), va="center", ha="left",
            fontsize=11, fontweight="bold",
            color=bar.get_facecolor(),
        )

    # ── RIGHT: styled column/dtype table ─────────────────────────────────────
    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis("off")
    ax_table.set_title("Column Detail", fontsize=12, fontweight="bold",
                       pad=12, color=TEXT_COLOR)

    col_names = column_names_and_dtypes.index.tolist()
    col_dtypes = column_names_and_dtypes.astype(str).tolist()
    n = len(col_names)
    row_height = 1 / (n + 1)

    # Header
    ax_table.text(0.05, 1 - row_height * 0.5, "Column",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center")
    ax_table.text(0.65, 1 - row_height * 0.5, "dtype",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center")

    # Divider line under header
    ax_table.axhline(y=1 - row_height, color="#CCCCCC", linewidth=1)

    # Rows
    for i, (col, dtype) in enumerate(zip(col_names, col_dtypes)):
        y = 1 - row_height * (i + 1.5)

        # Alternating row background
        if i % 2 == 0:
            ax_table.add_patch(plt.Rectangle(
                (0, y - row_height * 0.5), 1, row_height,
                transform=ax_table.transAxes,
                color="#EEEEEE", zorder=0
            ))

        ax_table.text(0.05, y, col, fontsize=10, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center")

        dtype_color = DTYPE_COLORS.get(dtype, DEFAULT_COLOR)
        ax_table.text(0.65, y, dtype, fontsize=10, fontweight="bold",
                      color=dtype_color,
                      transform=ax_table.transAxes, va="center")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_column_names_and_dtypes.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}")
    return column_names_and_dtypes

def display_first_few_rows(filepath: str, output_dir: str = "output", n_rows: int = 3) -> pd.DataFrame:
    base_filename = str(filepath).replace("data/", "")
    base_filename = base_filename.replace(".csv", "")

    df = pd.read_csv(filepath)
    first_few_rows = df.head(n_rows)

    # ── Style ────────────────────────────────────────────────────────────────
    BG_COLOR    = "#F8F9FA"
    TEXT_COLOR  = "#2D2D2D"
    CARD_COLOR  = "#FFFFFF"
    HEADER_COLOR = "#4C9BE8"
    ALT_ROW     = "#F0F4FF"
    FONT        = {"font.family": "DejaVu Sans", "font.size": 10}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR})

    n_cols  = len(df.columns)
    n_cards = len(first_few_rows)

    # ── Figure: one card per row, stacked vertically ─────────────────────────
    card_height = n_cols * 0.32 + 0.8
    fig_height  = card_height * n_cards + 1.0

    fig, axes = plt.subplots(n_cards, 1, figsize=(11, fig_height))
    fig.patch.set_facecolor(BG_COLOR)

    if n_cards == 1:
        axes = [axes]

    fig.suptitle("First Few Rows Preview", fontsize=15, fontweight="bold",
             color=TEXT_COLOR, y=1.0)

    for card_idx, (ax, (_, row)) in enumerate(zip(axes, first_few_rows.iterrows())):
        ax.set_facecolor(BG_COLOR)
        ax.axis("off")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # ── Card background ───────────────────────────────────────────────
        card_patch = mpatches.FancyBboxPatch(
            (0.01, 0.02), 0.98, 0.96,
            boxstyle="round,pad=0.01",
            linewidth=1.2, edgecolor="#DDDDDD",
            facecolor=CARD_COLOR,
            transform=ax.transAxes, zorder=0
        )
        ax.add_patch(card_patch)

        # ── Card header ───────────────────────────────────────────────────
        HEADER_H = 0.06  # fixed header height regardless of column count

        header_patch = mpatches.FancyBboxPatch(
            (0.01, 0.92), 0.98, HEADER_H,
            boxstyle="round,pad=0.005",
            linewidth=0, facecolor=HEADER_COLOR,
            transform=ax.transAxes, zorder=1
        )
        ax.add_patch(header_patch)
        ax.text(0.5, 0.95,
            f"Row {card_idx + 1}",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=2)

        # ── Rows per card ─────────────────────────────────────────────────
        row_h = 0.90 / n_cols  # remaining 90% of axes split across all columns

        for i, (col, val) in enumerate(row.items()):
            y_top = 0.92 - i * row_h
            y_mid = y_top - row_h / 2

            # Alternating background
            if i % 2 == 0:
                ax.add_patch(plt.Rectangle(
                    (0.01, y_top - row_h), 0.98, row_h,
                    transform=ax.transAxes,
                    facecolor=ALT_ROW, zorder=0
                ))

            # Column name
            ax.text(0.04, y_mid, str(col),
                    transform=ax.transAxes,
                    ha="left", va="center",
                    fontsize=9.5, fontweight="bold", color="#555555", zorder=2)

            # Divider
            ax.axvline(x=0.42, ymin=y_top - row_h, ymax=y_top,
                       color="#DDDDDD", linewidth=0.8, zorder=1)

            # Value — strip newlines and truncate long strings
            str_val = str(val).replace("\n", " ").replace("\r", " ")
            if len(str_val) > 55:
                str_val = str_val[:52] + "..."

            ax.text(0.44, y_mid, str_val,
                    transform=ax.transAxes,
                    ha="left", va="center",
                    fontsize=9.5, color=TEXT_COLOR, zorder=2)

    plt.tight_layout(h_pad=1.5)
    fig.subplots_adjust(top=0.97)

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_first_few_rows.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}")
    return first_few_rows

def get_null_count_per_column(filepath: str, output_dir: str = "output") -> dict:
    base_filename = str(filepath).replace("data/", "")
    base_filename = base_filename.replace(".csv", "")

    df = pd.read_csv(filepath)
    columns = list(df.columns)
    total_rows = len(df)

    # ── Compute null counts ───────────────────────────────────────────────────
    null_count_per_column = {}
    for column in columns:
        total_nan_count = df[column].isna().sum()
        total_empty_string_count = (df[column] == '').sum()
        null_count_per_column[column] = int(total_nan_count + total_empty_string_count)

    # ── Filter only columns with nulls ───────────────────────────────────────
    null_series = {k: v for k, v in null_count_per_column.items() if v > 0}

    # ── Style ────────────────────────────────────────────────────────────────
    BG_COLOR   = "#F8F9FA"
    TEXT_COLOR = "#2D2D2D"
    FONT       = {"font.family": "DejaVu Sans", "font.size": 11}
    plt.rcParams.update({**FONT, "text.color": TEXT_COLOR, "axes.labelcolor": TEXT_COLOR})

    def severity_color(pct):
        if pct == 0:  return "#AAAAAA"   # grey  — clean
        if pct < 5:   return "#57CC99"   # green — low
        if pct < 20:  return "#F4A261"   # orange — moderate
        return "#F07167"                 # red   — critical

    def severity_label(pct):
        if pct == 0:  return "Clean"
        if pct < 5:   return "Low"
        if pct < 20:  return "Moderate"
        return "Critical"

    # ── Figure: two panels side by side ──────────────────────────────────────
    fig, (ax_bar, ax_table) = plt.subplots(
        1, 2, figsize=(17, max(4, len(df.columns) * 0.45)),
        gridspec_kw={"width_ratios": [1.2, 1.4]}
    )
    fig.patch.set_facecolor(BG_COLOR)
    fig.suptitle("Missing Data Analysis", fontsize=15, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)

    # ── LEFT: horizontal bar chart (only columns with nulls) ─────────────────
    ax_bar.set_facecolor(BG_COLOR)

    if not null_series:
        ax_bar.text(0.5, 0.5, "✓ No missing values found!",
                    transform=ax_bar.transAxes,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#57CC99")
        ax_bar.axis("off")
        ax_bar.set_title("Columns with Missing Values", fontsize=12,
                         fontweight="bold", pad=12, color=TEXT_COLOR)
    else:
        col_names   = list(null_series.keys())
        null_counts = list(null_series.values())
        pcts        = [v / total_rows * 100 for v in null_counts]
        colors      = [severity_color(p) for p in pcts]

        # Sort ascending so highest % is at top
        sorted_data = sorted(zip(pcts, null_counts, col_names, colors), reverse=False)
        pcts, null_counts, col_names, colors = zip(*sorted_data)

        bars = ax_bar.barh(col_names, pcts, color=colors,
                           edgecolor="white", linewidth=1.2, zorder=3)

        ax_bar.xaxis.grid(True, color="#DDDDDD", linestyle="--", linewidth=0.8, zorder=0)
        ax_bar.set_axisbelow(True)
        for spine in ["top", "right", "bottom"]:
            ax_bar.spines[spine].set_visible(False)
        ax_bar.spines["left"].set_color("#CCCCCC")
        ax_bar.tick_params(colors=TEXT_COLOR, length=0)
        ax_bar.set_xlabel("% Missing", fontsize=11, labelpad=8)
        ax_bar.set_xlim(0, min(max(pcts) * 1.45, 100))
        ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

        # Value labels
        for bar, count, pct, color in zip(bars, null_counts, pcts, colors):
            ax_bar.text(
                pct + 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,}  ({pct:.1f}%)",
                va="center", ha="left",
                fontsize=10, fontweight="bold",
                color=color
            )

        ax_bar.set_title("Columns with Missing Values", fontsize=12,
                         fontweight="bold", pad=12, color=TEXT_COLOR)
        ax_bar.text(0.5, 1.00,
            f"{len(null_series)} of {len(columns)} columns have missing values",
            transform=ax_bar.transAxes,
            ha="center", fontsize=10, color="#888888")

        # Legend
        legend_items = [
            mpatches.Patch(color="#57CC99", label="Low  (<5%)"),
            mpatches.Patch(color="#F4A261", label="Moderate  (5–20%)"),
            mpatches.Patch(color="#F07167", label="Critical  (>20%)"),
        ]
        ax_bar.legend(handles=legend_items, loc="upper center",
                      bbox_to_anchor=(0.5, -0.15),
                      ncol=3, frameon=False, fontsize=10)

    # ── RIGHT: full column audit table ───────────────────────────────────────
    ax_table.set_facecolor(BG_COLOR)
    ax_table.axis("off")
    ax_table.set_title("Column Null Audit", fontsize=12, fontweight="bold",
                        pad=12, color=TEXT_COLOR)

    n = len(columns)
    row_height = 1 / (n + 1)

    # Header
    ax_table.text(0.05, 1 - row_height * 0.5, "Column",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center", ha='left')
    ax_table.text(0.55, 1 - row_height * 0.5, "Nulls",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center", ha='left')
    ax_table.text(0.72, 1 - row_height * 0.5, "% Missing",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center", ha='left')
    ax_table.text(0.88, 1 - row_height * 0.5, "Status",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center", ha='left')

    # Divider under header
    ax_table.axhline(y=1 - row_height, color="#CCCCCC", linewidth=1)

    # Rows
    for i, col in enumerate(columns):
        y = 1 - row_height * (i + 1.5)
        null_count = null_count_per_column[col]
        pct = null_count / total_rows * 100
        color = severity_color(pct)

        # Alternating background
        if i % 2 == 0:
            ax_table.add_patch(plt.Rectangle(
                (0, y - row_height * 0.5), 1, row_height,
                transform=ax_table.transAxes,
                color="#EEEEEE", zorder=0
            ))

        ax_table.text(0.05, y, col, fontsize=9.5, color=TEXT_COLOR,
                      transform=ax_table.transAxes, va="center")
        ax_table.text(0.55, y, f"{null_count:,}", fontsize=9.5,
                      fontweight="bold", color=color,
                      transform=ax_table.transAxes, va="center")
        ax_table.text(0.72, y, f"{pct:.1f}%", fontsize=9.5,
                      fontweight="bold", color=color,
                      transform=ax_table.transAxes, va="center")
        ax_table.text(0.88, y, severity_label(pct), fontsize=9.5,
                      fontweight="bold", color=color,
                      transform=ax_table.transAxes, va="center")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_missing_data.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}")
    return null_count_per_column

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

    for bar, count in zip(bars, values_ordered):
        pct = count / total_rows * 100
        ax_bar.text(
            max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}  ({pct:.1f}%)",
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
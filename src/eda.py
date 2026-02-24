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
        1, 2, figsize=(15, max(4, len(df.columns) * 0.45)),
        gridspec_kw={"width_ratios": [1.4, 1]}
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
        ax_bar.text(0.5, 1.02,
                    f"{len(null_series)} of {len(columns)} columns have missing values",
                    transform=ax_bar.transAxes,
                    ha="center", fontsize=10, color="#888888")

        # Legend
        legend_items = [
            mpatches.Patch(color="#57CC99", label="Low  (<5%)"),
            mpatches.Patch(color="#F4A261", label="Moderate  (5–20%)"),
            mpatches.Patch(color="#F07167", label="Critical  (>20%)"),
        ]
        ax_bar.legend(handles=legend_items, loc="lower right",
                      frameon=False, fontsize=10)

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
                  transform=ax_table.transAxes, va="center")
    ax_table.text(0.55, 1 - row_height * 0.5, "Nulls",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center")
    ax_table.text(0.75, 1 - row_height * 0.5, "% Missing",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center")
    ax_table.text(0.92, 1 - row_height * 0.5, "Status",
                  fontsize=11, fontweight="bold", color=TEXT_COLOR,
                  transform=ax_table.transAxes, va="center")

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
        ax_table.text(0.75, y, f"{pct:.1f}%", fontsize=9.5,
                      fontweight="bold", color=color,
                      transform=ax_table.transAxes, va="center")
        ax_table.text(0.92, y, severity_label(pct), fontsize=9.5,
                      fontweight="bold", color=color,
                      transform=ax_table.transAxes, va="center")

    # ── Save ─────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(output_dir) / f"{base_filename}_missing_data.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Chart saved → {out_path}")
    return null_count_per_column
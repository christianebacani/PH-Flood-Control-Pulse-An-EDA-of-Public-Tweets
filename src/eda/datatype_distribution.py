import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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
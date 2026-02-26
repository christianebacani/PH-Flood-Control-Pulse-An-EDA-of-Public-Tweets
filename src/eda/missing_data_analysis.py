import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

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
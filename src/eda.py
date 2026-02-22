import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
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

def get_column_names_and_dtypes(filepath: str) -> pd.Series:
    df = pd.read_csv(filepath)
    # Get the column names and datatype of the dataset
    column_names_and_dtypes = df.dtypes

    return column_names_and_dtypes

def display_first_few_rows(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    # Get the first 3 rows of the dataset
    first_few_rows = df.head(3)

    return first_few_rows
import re
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from pathlib import Path
from matplotlib.ticker import FuncFormatter

# ──────────────────────────────────────────────────────────────────
# 🎨  Design tokens  (matches the rest of the EDA suite)
# ──────────────────────────────────────────────────────────────────

BG      = "#F8FAFC"
CARD_BG = "#FFFFFF"
TXT     = "#0F172A"
TXT_MED = "#475569"
TXT_LT  = "#94A3B8"
RULE    = "#E2E8F0"
GRID    = "#CBD5E1"

C_KEYWORD = "#3B82F6"   # blue
C_HASHTAG = "#8B5CF6"   # violet
C_BIGRAM  = "#F59E0B"   # amber — distinct from keywords and hashtags

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
})

# ──────────────────────────────────────────────────────────────────
# 🔤  Stopwords  (English + Filipino)
# ──────────────────────────────────────────────────────────────────

ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "yet", "both", "either", "neither", "this", "that", "these",
    "those", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "its", "they", "them", "their", "what", "which", "who", "whom",
    "how", "when", "where", "why", "all", "any", "each", "few", "more",
    "most", "other", "some", "such", "than", "then", "too", "very", "just",
    "also", "about", "as", "if", "up", "out", "into", "over", "after",
    "before", "between", "through", "during", "still", "now", "even",
    "back", "only", "here", "there", "while", "because", "since", "until",
    "although", "though", "however", "therefore", "thus", "hence", "well",
    "re", "s", "t", "don", "doesn", "didn", "won", "hasn", "hadn",
    "isn", "aren", "wasn", "weren", "ain", "via", "amp", "get", "got",
    "let", "say", "said", "like", "know", "make", "made", "go", "going",
    "one", "two", "three", "four", "five", "use", "used", "need", "want",
}

FILIPINO_STOPWORDS = {
    "ako", "ikaw", "ka", "siya", "kami", "tayo", "kayo", "sila",
    "ko", "mo", "niya", "namin", "natin", "ninyo", "nila",
    "akin", "iyo", "kanya", "atin", "inyo", "kanila",
    "ang", "ng", "na", "sa", "ni", "kay", "nang", "kung", "dahil",
    "para", "pero", "at", "o", "kasi", "kaya", "din", "rin", "pa",
    "po", "nga", "naman", "lang", "lamang", "man", "raw", "daw",
    "ba", "yata", "pala", "talaga", "halos", "mula", "hanggang",
    "dito", "doon", "diyan", "ito", "iyon", "iyan", "yon", "yun",
    "saan", "kailan", "sino", "ano", "paano", "bakit",
    "may", "mayroon", "wala", "walang", "hindi", "huwag",
    "oo", "opo", "ayaw", "gusto", "ibig",
    "yan", "yung", "yong", "nung", "noong", "noon",
    "lahat", "bawat", "ilang", "ibang", "iba",
    "kanyang", "nito", "niyon",
    "mga", "eh", "ah", "oh", "ha", "ay", "si", "ni",
    "sya", "niya", "nito", "diba", "talaga", "dapat",
    "kung", "kapag", "habang", "gayun", "ganon", "ganyan",
    "mismo", "sana", "siguro", "pwede", "puwede",
}

STOPWORDS = ENGLISH_STOPWORDS | FILIPINO_STOPWORDS

# Domain noise: ultra-high frequency words with no analytical signal.
# Includes DPWH full-name fragments, generic political words, and
# Filipino function words that slipped through the stopword filter.
DOMAIN_NOISE = {
    # Core dataset terms (appear in nearly every tweet)
    "flood", "control", "dpwh", "flooding", "project", "projects",
    "rt", "http", "https", "co", "t",
    # DPWH full-name fragments → suppress garbage bigrams
    "department", "public", "works", "highways", "blue", "ribbon",
    # Generic gov/political words with no specificity in this corpus
    "senate", "committee", "budget", "president", "official",
    "office", "congress", "house", "government", "administration",
    # Vague/generic high-frequency words
    "people", "time", "year", "years", "money",
    "new", "big", "good", "bad", "great", "many", "much",
    "city", "district", "secretary", "commission", "independent",
    # Short abbreviations and fragments
    "sen", "his", "rep", "gov", "pnp", "nyo", "mag",
    # Stems/variants already represented by cleaner forms
    "corrupt",   # redundant — "corruption" already in keywords
    # Filipino function/filler words not caught by stopword list
    "kaban", "bayan", "ibalik", "lng", "tlga", "naman", "kasi", "kahit",
    "asawang", "1st", "mayor", "vico",
    # More Filipino verb prefixes / infixes with no standalone meaning
    "nya", "nag", "pag", "mga", "sec", "pangulong", "nepo",
    # Duplicate stems (cleaner form already present)
    "contractor",   # "contractors" is the cleaner plural form
    # Single-name fragments already captured in bigrams
    "martin", "dizon", "hearing",
    # Remaining noise from latest output
    "dating", "read", "ngayon", "umano", "edsa", "infrastructure",
    "senator", "witness", "state",
}

# ──────────────────────────────────────────────────────────────────
# 🔧  Regex patterns
# ──────────────────────────────────────────────────────────────────

_URL_RE     = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")
_CLEAN_RE   = re.compile(r"[^a-zA-Z0-9\s]")
_NUMERIC_RE = re.compile(r"^\d+$")


# ──────────────────────────────────────────────────────────────────
# 🔧  Extraction helpers
# ──────────────────────────────────────────────────────────────────

def _clean_tokens(text: str) -> list:
    """Shared cleaning pipeline for keywords and bigrams."""
    t = _URL_RE.sub(" ", str(text))
    t = _MENTION_RE.sub(" ", t)
    t = _HASHTAG_RE.sub(" ", t)
    t = _CLEAN_RE.sub(" ", t).lower()
    return [
        tok for tok in t.split()
        if tok not in STOPWORDS
        and tok not in DOMAIN_NOISE
        and len(tok) > 2
        and not _NUMERIC_RE.match(tok)
    ]


def _extract_keywords(texts: pd.Series) -> Counter:
    counter = Counter()
    for text in texts.dropna():
        counter.update(_clean_tokens(text))
    return counter


def _extract_hashtags(texts: pd.Series) -> Counter:
    counter = Counter()
    for text in texts.dropna():
        tags = _HASHTAG_RE.findall(str(text))
        counter.update(t.lower() for t in tags if len(t) > 1)
    return counter


def _extract_bigrams(texts: pd.Series) -> Counter:
    """
    Two-word phrases after the same cleaning pipeline as keywords.
    Bigrams surface concepts that single tokens miss —
    e.g. 'senate investigation', 'ghost project', 'billion pesos'.
    """
    counter = Counter()
    for text in texts.dropna():
        tokens = _clean_tokens(text)
        for w1, w2 in zip(tokens, tokens[1:]):
            counter[(w1, w2)] += 1
    return counter


def _count_obfuscated_mentions(texts: pd.Series) -> int:
    total = 0
    for text in texts.dropna():
        total += sum(
            1 for m in _MENTION_RE.findall(str(text))
            if _NUMERIC_RE.match(m)
        )
    return total


# ──────────────────────────────────────────────────────────────────
# 🔧  Shared axis styler
# ──────────────────────────────────────────────────────────────────

def _style_ax(ax, grid_axis="x"):
    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom", "left"))
        if loc in ("bottom", "left"):
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    ax.tick_params(which="major", length=4, width=0.8, labelsize=9)
    if grid_axis == "x":
        ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.yaxis.grid(False)
    else:
        ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def _safe_show(fig, save_path=None):
    if save_path:
        save_path = os.path.normpath(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Chart saved → {save_path}")
    is_gui = matplotlib.get_backend().lower() not in (
        "agg", "cairo", "pdf", "ps", "svg", "template"
    )
    if is_gui:
        plt.show()
    else:
        if not save_path:
            fig.savefig("text_analysis_output.png", dpi=300, bbox_inches="tight")
            print("[viz] Non-interactive — saved → text_analysis_output.png")
        plt.close(fig)


# ──────────────────────────────────────────────────────────────────
# 📊  Shared bar panel renderer
# ──────────────────────────────────────────────────────────────────

def _bar_panel(ax, items: list, color: str, title: str, xlabel: str,
               bar_height: float = 0.60, label_fontsize: float = 9,
               value_fontsize: float = 8.5, accent_top: int = 3):
    """
    Horizontal bar chart with ranked items.
    Top `accent_top` bars are rendered at full opacity; rest slightly muted.
    items: list of (label_str, count) in descending order.
    """
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=TXT_MED)
        ax.set_title(title, fontsize=11, fontweight="bold", color=TXT, pad=10)
        return

    labels = [label for label, _ in items][::-1]
    values = [count for _, count in items][::-1]
    n      = len(values)

    # Opacity: top entries full, rest slightly dimmed
    alphas = [0.88 if (n - 1 - i) < accent_top else 0.60
              for i in range(n)]

    for i, (lbl, val, alpha) in enumerate(zip(labels, values, alphas)):
        ax.barh(lbl, val, color=color, height=bar_height,
                alpha=alpha, zorder=2)

    max_val = max(values)
    for i, (lbl, val) in enumerate(zip(labels, values)):
        ax.text(
            val + max_val * 0.012,
            i,
            f"{val:,}",
            va="center", fontsize=value_fontsize,
            color=TXT_MED, fontweight="normal",
        )

    # Rank numbers on the y-axis labels
    ranked_labels = [f"#{n-i}  {lbl}" for i, lbl in enumerate(labels)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(ranked_labels, fontsize=label_fontsize)

    ax.set_title(title, fontsize=11, fontweight="bold", color=TXT, pad=10)
    ax.set_xlabel(xlabel, fontsize=8.5, color=TXT_MED, labelpad=6)
    ax.set_xlim(0, max_val * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="y", labelsize=label_fontsize, length=0)
    ax.tick_params(axis="x", labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_clip_on(False)

    # Clean spines
    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom",))
        if loc == "bottom":
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def _section_header(fig, ax, label: str, color: str, description: str):
    """Draw a colored left-rule section header above an axes."""
    bbox  = ax.get_position()
    x_fig = bbox.x0
    y_fig = bbox.y1 + 0.012

    # Colored pill label
    fig.text(x_fig, y_fig, f"  {label}  ",
             fontsize=8, fontweight="bold", color="white",
             va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.30", facecolor=color,
                       edgecolor="none", linewidth=0))
    # Description beside it
    fig.text(x_fig + 0.068, y_fig, description,
             fontsize=8, color=TXT_MED, va="bottom", ha="left",
             style="italic")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_text_analysis(data_source, save_path=None, top_n: int = 15):
    """
    Text Analysis — Keywords, Hashtags & Bigrams from the `text` column.

    Layout (executive-ready):
      Row 1 (full width) — Top Keywords: primary analytical signal
      Row 2 (split 50/50) — Top Hashtags | Top Phrases (Bigrams)

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path   : output PNG path
    top_n       : entries per panel (default 15)
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str)          else data_source.copy()

    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")

    N     = len(df)
    texts = df["text"]

    # ── Extract ───────────────────────────────────────────────────────────────
    keyword_counts = _extract_keywords(texts)
    hashtag_counts = _extract_hashtags(texts)
    bigram_counts  = _extract_bigrams(texts)
    n_obfuscated   = _count_obfuscated_mentions(texts)

    n_hashtags  = sum(hashtag_counts.values())
    unique_tags = len(hashtag_counts)

    top_n_sub = min(10, top_n)   # bottom panels capped at 10 for readability

    kw_items = keyword_counts.most_common(top_n)
    ht_items = [(f"#{tag}", cnt) for tag, cnt
                in hashtag_counts.most_common(top_n_sub)]
    bi_items = [(f"{w1} {w2}", cnt) for (w1, w2), cnt
                in bigram_counts.most_common(top_n_sub)]

    # ── Figure ────────────────────────────────────────────────────────────────
    # Use a 4-row GridSpec:
    #   Row 0 — header title bar
    #   Row 1 — KPI stats strip
    #   Row 2 — keywords (full width)
    #   Row 3 — hashtags | bigrams (split)
    fig = plt.figure(figsize=(18, 22), facecolor=BG)

    gs = gridspec.GridSpec(
        4, 2, figure=fig,
        left=0.13, right=0.97,
        top=0.985, bottom=0.048,
        hspace=0.0,
        wspace=0.40,
        height_ratios=[0.045, 0.038, 0.55, 0.48],
    )

    # ── Header axes ───────────────────────────────────────────────────────────
    ax_header = fig.add_subplot(gs[0, :])
    ax_header.set_facecolor(TXT)
    ax_header.set_xlim(0, 1)
    ax_header.set_ylim(0, 1)
    ax_header.axis("off")
    ax_header.text(0.5, 0.5,
                   "Text Analysis — Keywords, Hashtags & Top Phrases",
                   ha="center", va="center",
                   fontsize=16, fontweight="bold", color="white",
                   transform=ax_header.transAxes)

    # ── KPI strip axes ────────────────────────────────────────────────────────
    ax_kpi = fig.add_subplot(gs[1, :])
    ax_kpi.set_facecolor("#1E293B")
    ax_kpi.set_xlim(0, 1)
    ax_kpi.set_ylim(0, 1)
    ax_kpi.axis("off")

    kpis = [
        (f"{N:,}",           "tweets analysed"),
        (f"{n_hashtags:,}",  "hashtag uses"),
        (f"{unique_tags:,}", "unique hashtags"),
        (f"{n_obfuscated:,}","obfuscated @IDs excluded"),
    ]
    for i, (val, lbl) in enumerate(kpis):
        x = 0.12 + i * 0.24
        ax_kpi.text(x, 0.68, val,
                    ha="left", va="center",
                    fontsize=11, fontweight="bold", color="#60A5FA",
                    transform=ax_kpi.transAxes)
        ax_kpi.text(x, 0.18, lbl,
                    ha="left", va="center",
                    fontsize=7.5, color=TXT_LT,
                    transform=ax_kpi.transAxes)

    # ── Chart axes ────────────────────────────────────────────────────────────
    ax_kw = fig.add_subplot(gs[2, :])
    ax_ht = fig.add_subplot(gs[3, 0])
    ax_bi = fig.add_subplot(gs[3, 1])

    # Add breathing room between header rows and chart rows
    gs.update(hspace=0.44)

    # ── Draw chart panels ─────────────────────────────────────────────────────
    _bar_panel(ax_kw, kw_items, C_KEYWORD,
               f"Top {top_n} Keywords — Most Frequent Terms in Discourse",
               "Number of Occurrences",
               bar_height=0.52, label_fontsize=9, value_fontsize=8.5)

    _bar_panel(ax_ht, ht_items, C_HASHTAG,
               f"Top {top_n_sub} Hashtags",
               "Number of Uses",
               bar_height=0.58, label_fontsize=9, value_fontsize=8.5)

    _bar_panel(ax_bi, bi_items, C_BIGRAM,
               f"Top {top_n_sub} Phrases (Bigrams)",
               "Co-occurrences",
               bar_height=0.58, label_fontsize=9, value_fontsize=8.5)

    # ── Section pill + description + left rule above each chart panel ─────────
    for ax, pill_lbl, color, desc, full_title in [
        (ax_kw, "KEYWORDS", C_KEYWORD,
         "Individual words driving the conversation",
         f"Top {top_n} Keywords — Most Frequent Terms in Discourse"),
        (ax_ht, "HASHTAGS", C_HASHTAG,
         "Organised campaigns & topics",
         f"Top {top_n_sub} Hashtags"),
        (ax_bi, "BIGRAMS",  C_BIGRAM,
         "Two-word phrases — people, places & concepts",
         f"Top {top_n_sub} Phrases (Bigrams)"),
    ]:
        # Left color rule
        ax.plot([0, 0], [0, 1], color=color, linewidth=3,
                transform=ax.transAxes, clip_on=False, zorder=10)
        # Pill badge — sits cleanly above chart title space
        ax.text(0.003, 1.045, f" {pill_lbl} ",
                transform=ax.transAxes,
                fontsize=7.5, fontweight="bold", color="white",
                va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.28", facecolor=color,
                          edgecolor="none"))
        # Description text beside pill
        pill_width = len(pill_lbl) * 0.009 + 0.055
        ax.text(pill_width, 1.045, desc,
                transform=ax.transAxes,
                fontsize=8, color=TXT_MED,
                va="bottom", ha="left", style="italic")
        # Chart title
        ax.set_title(full_title,
                     fontsize=10.5, fontweight="bold",
                     color=TXT, loc="left", pad=6)

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.022,
        "Bigrams = most frequent two-word combinations after removing "
        "stopwords and domain-specific noise.  "
        f"All @mentions were obfuscated numeric IDs ({n_obfuscated:,} excluded).",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
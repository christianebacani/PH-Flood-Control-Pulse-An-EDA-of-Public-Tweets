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
# Also includes fragments of "Dept of Public Works and Highways Blue Ribbon
# Committee" which produce garbage bigrams like "public works", "works highways",
# "department public", "blue ribbon" after stopword removal.
DOMAIN_NOISE = {
    # Core dataset terms (appear in nearly every tweet)
    "flood", "control", "dpwh", "flooding", "project", "projects",
    "rt", "http", "https", "co", "t",
    # DPWH full-name fragments → suppress garbage bigrams
    "department", "public", "works", "highways", "blue", "ribbon",
    # Generic gov/political words with no specificity in this corpus
    "senate", "committee", "budget", "president", "official",
    "office", "congress", "house", "government", "administration",
    # Vague high-frequency words
    "people", "time", "year", "years", "money",
    "new", "big", "good", "bad", "great", "many", "much",
    # Abbreviations / fragments that slipped keyword filter
    "sen", "his", "pnp", "gov",
    # Filipino function words not caught by stopword list
    "kaban", "lng", "tlga", "naman", "kasi", "kahit",
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
# 📊  Single horizontal bar panel
# ──────────────────────────────────────────────────────────────────

def _bar_panel(ax, items: list, color: str, title: str, xlabel: str):
    """items: list of (label_str, count) in descending order."""
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=TXT_MED)
        ax.set_title(title)
        return

    labels = [label for label, _ in items][::-1]
    values = [count for _, count in items][::-1]

    bars = ax.barh(labels, values, color=color, height=0.62,
                   alpha=0.85, zorder=2)

    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(
            val + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center", fontsize=8.5, color=TXT_MED,
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel(xlabel, fontsize=9, color=TXT_MED, labelpad=6)
    ax.set_xlim(0, max_val * 1.28)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="y", labelsize=9)
    for lbl in ax.get_yticklabels():
        lbl.set_clip_on(False)
    _style_ax(ax, grid_axis="x")


# ──────────────────────────────────────────────────────────────────
# 📊  PUBLIC FUNCTION
# ──────────────────────────────────────────────────────────────────

def get_text_analysis(data_source, save_path=None, top_n: int = 20):
    """
    Text Analysis — top keywords, hashtags, and bigrams from the `text` column.

    Note: all @mentions in this dataset are obfuscated numeric IDs (anonymised).
    The third panel therefore shows top bigrams (two-word phrases) instead,
    which provide richer analytical signal about the topics being discussed.

    Parameters
    ----------
    data_source : str or pd.DataFrame
    save_path   : output file path (PNG)
    top_n       : how many top entries to show per panel (default 20)
    """
    df = pd.read_csv(data_source) if isinstance(data_source, str) \
         else data_source.copy()

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
    unique_bi   = len(bigram_counts)

    kw_items = keyword_counts.most_common(top_n)
    ht_items = [(f"#{tag}", cnt) for tag, cnt in hashtag_counts.most_common(top_n)]
    bi_items = [(f"{w1} {w2}", cnt) for (w1, w2), cnt in bigram_counts.most_common(top_n)]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        "Text Analysis \u2014 Keywords, Hashtags & Top Phrases",
        fontsize=17, fontweight="bold", color=TXT, y=0.99,
    )

    fig.text(
        0.5, 0.965,
        f"{N:,} tweets  ·  {n_hashtags:,} hashtag uses ({unique_tags:,} unique)  "
        f"·  {unique_bi:,} unique bigrams  "
        f"·  {n_obfuscated:,} obfuscated @IDs excluded from mentions",
        ha="center", fontsize=9, color=TXT_MED,
    )

    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.09, right=0.97,
        top=0.92, bottom=0.06,
        wspace=0.55,
    )

    ax_kw = fig.add_subplot(gs[0, 0])
    ax_ht = fig.add_subplot(gs[0, 1])
    ax_bi = fig.add_subplot(gs[0, 2])

    _bar_panel(ax_kw, kw_items, C_KEYWORD,
               f"Top {top_n} Keywords",            "Occurrences")
    _bar_panel(ax_ht, ht_items, C_HASHTAG,
               f"Top {top_n} Hashtags",             "Uses")
    _bar_panel(ax_bi, bi_items, C_BIGRAM,
               f"Top {top_n} Phrases (Bigrams)",    "Co-occurrences")

    fig.text(
        0.5, 0.01,
        "* Phrases panel shows the most frequent two-word combinations after removing stopwords and domain noise.  "
        f"{n_obfuscated:,} numeric @IDs (anonymised accounts) were excluded from mention analysis.",
        ha="center", fontsize=8, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
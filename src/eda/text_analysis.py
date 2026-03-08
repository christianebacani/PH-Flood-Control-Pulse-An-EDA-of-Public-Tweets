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
# 🎨  Design tokens
# ──────────────────────────────────────────────────────────────────

BG      = "#F8FAFC"
CARD_BG = "#FFFFFF"
TXT     = "#0F172A"
TXT_MED = "#475569"
TXT_LT  = "#94A3B8"
RULE    = "#E2E8F0"
GRID    = "#CBD5E1"

C_KEYWORD = "#3B82F6"
C_HASHTAG = "#8B5CF6"
C_BIGRAM  = "#F59E0B"

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
# 🔤  Stopwords
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

DOMAIN_NOISE = {
    "flood", "control", "dpwh", "flooding", "project", "projects",
    "rt", "http", "https", "co", "t",
    "department", "public", "works", "highways", "blue", "ribbon",
    "senate", "committee", "budget", "president", "official",
    "office", "congress", "house", "government", "administration",
    "people", "time", "year", "years", "money",
    "new", "big", "good", "bad", "great", "many", "much",
    "city", "district", "secretary", "commission", "independent",
    "sen", "his", "rep", "gov", "pnp", "nyo", "mag",
    "corrupt",
    "kaban", "bayan", "ibalik", "lng", "tlga", "naman", "kasi", "kahit",
    "asawang", "1st", "mayor", "vico",
    "nya", "nag", "pag", "mga", "sec", "pangulong", "nepo",
    "contractor",
    "martin", "dizon", "hearing",
    "dating", "read", "ngayon", "umano", "edsa", "infrastructure",
    "senator", "witness", "state",
    "ping", "officials", "official", "speaker",
    "former", "against", "panfilo", "bongbong",
    "ici", "sarah", "pro", "tempore",
    "isang", "luneta", "party", "list", "bong", "revilla",
    "nasa", "engineer", "vince", "philippines",
    "alam", "senador", "dds", "mas",
}

_URL_RE     = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")
_CLEAN_RE   = re.compile(r"[^a-zA-Z0-9\s]")
_NUMERIC_RE = re.compile(r"^\d+$")


def _clean_tokens(text: str) -> list:
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


def _bar_panel(ax, items: list, color: str, title: str, xlabel: str,
               bar_height: float = 0.60, label_fontsize: float = 9,
               value_fontsize: float = 8.5, accent_top: int = 3):
    if not items:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color=TXT_MED)
        # Title drawn 28 pt above axes top (below pill at 48 pt)
        ax.annotate(title, xy=(0.5, 1), xycoords="axes fraction",
                    xytext=(0, 28), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=TXT,
                    ha="center", va="bottom", annotation_clip=False)
        return

    labels = [label for label, _ in items][::-1]
    values = [count for _, count in items][::-1]
    n      = len(values)

    alphas = [0.88 if (n - 1 - i) < accent_top else 0.60 for i in range(n)]

    for i, (lbl, val, alpha) in enumerate(zip(labels, values, alphas)):
        ax.barh(lbl, val, color=color, height=bar_height, alpha=alpha, zorder=2)

    max_val = max(values)
    for i, (lbl, val) in enumerate(zip(labels, values)):
        ax.text(
            val + max_val * 0.012, i,
            f"{val:,}",
            va="center", fontsize=value_fontsize,
            color=TXT_MED, fontweight="normal",
        )

    ranked_labels = [f"#{n-i}  {lbl}" for i, lbl in enumerate(labels)]
    ax.set_yticks(range(n))
    ax.set_yticklabels(ranked_labels, fontsize=label_fontsize)

    # Title drawn at 28 pt above axes top — sits between pill row (48 pt) and axes
    ax.annotate(title, xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, 28), textcoords="offset points",
                fontsize=11, fontweight="bold", color=TXT,
                ha="center", va="bottom", annotation_clip=False)

    ax.set_xlabel(xlabel, fontsize=8.5, color=TXT_MED, labelpad=6)
    ax.set_xlim(0, max_val * 1.22)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="y", labelsize=label_fontsize, length=0)
    ax.tick_params(axis="x", labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_clip_on(False)

    for loc, spine in ax.spines.items():
        spine.set_visible(loc in ("bottom",))
        if loc == "bottom":
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7, zorder=0)
    ax.yaxis.grid(False)
    ax.set_axisbelow(True)


def get_text_analysis(data_source, save_path=None, top_n: int = 15):
    df = pd.read_csv(data_source) if isinstance(data_source, str) else data_source.copy()

    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")

    N     = len(df)
    texts = df["text"]

    keyword_counts = _extract_keywords(texts)
    hashtag_counts = _extract_hashtags(texts)
    bigram_counts  = _extract_bigrams(texts)
    n_obfuscated   = _count_obfuscated_mentions(texts)

    n_hashtags  = sum(hashtag_counts.values())
    unique_tags = len(hashtag_counts)

    top_n_sub = min(10, top_n)

    kw_items = keyword_counts.most_common(top_n)
    ht_items = [(f"#{tag}", cnt) for tag, cnt in hashtag_counts.most_common(top_n_sub)]
    bi_items = [(f"{w1} {w2}", cnt) for (w1, w2), cnt in bigram_counts.most_common(top_n_sub)]

    # ── Figure ───────────────────────────────────────────────────────────────
    # Extra vertical space (22 → was 21) gives the pill rows room to breathe
    fig = plt.figure(figsize=(18, 22), facecolor=BG)

    import matplotlib.patches as mpatches

    # Header band
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.954), 1, 0.046, boxstyle="square,pad=0",
        facecolor=TXT, edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))
    fig.text(0.5, 0.977,
             "Text Analysis — Keywords, Hashtags & Top Phrases",
             ha="center", va="center",
             fontsize=16, fontweight="bold", color="white", zorder=3)

    # KPI strip
    fig.add_artist(mpatches.FancyBboxPatch(
        (0, 0.916), 1, 0.038, boxstyle="square,pad=0",
        facecolor="#1E293B", edgecolor="none",
        transform=fig.transFigure, clip_on=False, zorder=2))
    kpis = [
        (f"{N:,}",           "tweets analysed"),
        (f"{n_hashtags:,}",  "hashtag uses"),
        (f"{unique_tags:,}", "unique hashtags"),
        (f"{n_obfuscated:,}","obfuscated @IDs excluded"),
    ]
    for i, (val, lbl) in enumerate(kpis):
        xf = 0.07 + i * 0.235
        fig.text(xf, 0.941, val, ha="left", va="center",
                 fontsize=11, fontweight="bold", color="#60A5FA", zorder=3)
        fig.text(xf, 0.924, lbl, ha="left", va="center",
                 fontsize=7.5, color=TXT_LT, zorder=3)
        # Thin vertical divider before each KPI except the first
        if i > 0:
            fig.add_artist(mpatches.FancyBboxPatch(
                (xf - 0.018, 0.920), 0.0015, 0.026,
                boxstyle="square,pad=0",
                facecolor="#334155", edgecolor="none",
                transform=fig.transFigure, clip_on=False, zorder=3))

    # ── GridSpec: more top padding so pill row clears the KPI strip ──────────
    gs = gridspec.GridSpec(
        2, 2, figure=fig,
        left=0.13, right=0.97,
        top=0.865,          # leave room above for pill + title of top panel
        bottom=0.050,
        hspace=0.28,
        wspace=0.42,
        height_ratios=[1.1, 1.0],
    )
    ax_kw = fig.add_subplot(gs[0, :])
    ax_ht = fig.add_subplot(gs[1, 0])
    ax_bi = fig.add_subplot(gs[1, 1])

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

    # ── Pill badges drawn with axes.annotate (axes-fraction coords) ──────────
    # This keeps pill + description anchored to the axes regardless of figure
    # size, avoiding the overlap caused by figure-fraction drift.
    panel_meta = [
        (ax_kw, "KEYWORDS", C_KEYWORD, "Individual words driving the conversation"),
        (ax_ht, "HASHTAGS", C_HASHTAG, "Organised campaigns & topics"),
        (ax_bi, "BIGRAMS",  C_BIGRAM,  "Two-word phrases — people, places & concepts"),
    ]

    for ax, pill, color, desc in panel_meta:
        # Left color rule
        ax.plot([0, 0], [0, 1], color=color, linewidth=3,
                transform=ax.transAxes, clip_on=False, zorder=10)

        # Pill badge — placed just above the axes using annotate so it scales
        # with the axes and never drifts into the title
        # Pill badge at 48 pt above axes top — well clear of title at 28 pt
        ax.annotate(
            f"  {pill}  ",
            xy=(0, 1), xycoords="axes fraction",
            xytext=(0, 48), textcoords="offset points",
            fontsize=7.5, fontweight="bold", color="white",
            va="bottom", ha="left", annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.28",
                      facecolor=color, edgecolor="none"),
        )

        # Description beside the pill
        ax.annotate(
            desc,
            xy=(0, 1), xycoords="axes fraction",
            xytext=(65, 48), textcoords="offset points",
            fontsize=8, color=TXT_MED, style="italic",
            va="bottom", ha="left", annotation_clip=False,
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.020,
        "Bigrams = most frequent two-word combinations after removing "
        "stopwords and domain-specific noise.  "
        f"All @mentions were obfuscated numeric IDs ({n_obfuscated:,} excluded).",
        ha="center", fontsize=7.5, color=TXT_LT, style="italic",
    )

    _safe_show(fig, save_path)
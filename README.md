# PH Flood Control Pulse: An Exploratory Data Analysis of Public Tweets

When Typhoon season hits the Philippines, one of the loudest conversations on Twitter isn't about the weather — it's about the government. This EDA explores **195,744 public tweets** from well-known Twitter accounts discussing the DPWH (Department of Public Works and Highways) and its flood control projects, asking a simple first question before any analysis begins: **can we trust this data?**

> 📦 **Source:** [DPWH Flood Control Projects 2025 — Kaggle](https://www.kaggle.com/datasets/bwandowando/tweets-on-dpwh-and-flood-control-projects-2025)

This report covers two linked datasets:

| | Dataset | Rows | Columns | What it contains |
|---|---|---|---|---|
| 1 | `for_export_dpwh_floodcontrol` | 195,744 | 16 | The tweets — text, engagement counts, timestamps, language |
| 2 | `well_known_authors_dpwh_floodcontrol` | 227 | 8 | The authors — profiles, follower counts, verification status |

Each dataset is examined across four steps: **shape → schema → missing data → data quality.**

---

## Dataset 1: Tweets

### 1.1 How Big Is the Dataset?

![Dataset Shape](output/for_export_dpwh_floodcontrol_dataset_shape.png)

Nearly **200,000 tweets** collected from a focused set of well-known accounts. That's large enough to find meaningful patterns in engagement, language use, and temporal trends — but small enough to process without heavy infrastructure.

At **16 columns**, each tweet carries rich metadata beyond just the text: who wrote it, when, how many people saw it, liked it, retweeted it, replied to it, bookmarked it, and whether it was a reply or a quote tweet.

---

### 1.2 What Does Each Column Contain?

![Column Names and Data Types](output/for_export_dpwh_floodcontrol_column_names_and_dtypes.png)

Most columns are numbers — specifically the engagement metrics. Here's a plain-English breakdown of the full schema:

| Column | Type | What it means |
|---|---|---|
| `pseudo_id` | `int64` | Unique identifier for each tweet |
| `text` | `str` | The tweet content |
| `retweetCount` | `int64` | How many times it was retweeted |
| `replyCount` | `int64` | How many replies it received |
| `likeCount` | `int64` | How many likes it received |
| `quoteCount` | `int64` | How many times it was quote-tweeted |
| `viewCount` | `int64` | How many times it was seen |
| `bookmarkCount` | `int64` | How many times it was bookmarked |
| `createdAt` | `str` ⚠️ | When the tweet was posted — stored as text, should be a date |
| `lang` | `str` | Language of the tweet (`tl` = Filipino, `en` = English) |
| `isReply` | `bool` ⚠️ | Whether this tweet is a reply to another tweet |
| `pseudo_conversationId` | `int64` | Groups tweets in the same thread |
| `pseudo_inReplyToUsername` | `float64` ⚠️ | The user being replied to — stored incorrectly as a number |
| `pseudo_author_userName` | `int64` | The tweet author (as an obfuscated ID) |
| `quoted_pseudo_id` | `float64` | The ID of the tweet being quoted (empty if not a quote tweet) |
| `author_isBlueVerified` | `bool` ⚠️ | Whether the author has Twitter Blue verification |

> ⚠️ = columns with a type issue that needs fixing before analysis. Covered in Section 1.4.

**dtype summary:** 9 `int64` · 3 `str` · 2 `float64` · 2 `bool`

---

### 1.3 Is Any Data Missing?

![Missing Data](output/for_export_dpwh_floodcontrol_missing_data.png)

**Good news: only 2 of 16 columns have missing values — and both are missing for a perfectly good reason.**

This is important to understand before jumping to "clean the data." Not all missing values are mistakes. In this dataset, the two missing columns are empty *by design*, not by accident.

#### `quoted_pseudo_id` — 170,346 missing (87%)

A "quoted tweet ID" only exists if a tweet is quoting another tweet. If someone just posts an original thought, there's nothing to quote — so this field is empty. **87% of tweets are original posts, not quote tweets.** That's not a data problem; that's just how Twitter works.

> **Decision: Keep as-is.** The empty value *is* the information — it tells us the tweet is original. Filling it would be wrong. Removing those rows would delete 87% of the dataset.

#### `pseudo_inReplyToUsername` — 123,213 missing (62.9%)

Similarly, a "reply-to username" only exists if the tweet is a reply. When `isReply = False`, this column is always empty. When `isReply = True`, it is always filled. The two columns move perfectly in sync.

> **Decision: Keep as-is.** The missing values are fully explained by `isReply`. No imputation needed. Removing these rows would eliminate every original tweet in the dataset.

**The other 14 columns have zero missing values.** No rows need to be removed.

---

### 1.4 Is the Data Trustworthy?

![Data Quality Report](output/for_export_dpwh_floodcontrol_data_quality.png)

The data is largely clean and usable, but there are **1 issue** and **6 warnings** that need to be addressed before analysis. None of them are showstoppers — they are all standard preprocessing tasks.

#### 🔴 Duplicate Rows — 1 row

One tweet appears twice in the dataset (`pseudo_id` has 1 duplicate). This is almost certainly a scraping overlap — the same tweet was captured twice during data collection. It affects just 0.0005% of the data, so the impact is negligible, but the duplicate should be dropped before any counting or aggregation.

```python
df = df.drop_duplicates(subset=["pseudo_id"])
```

#### 🟡 Wrong Data Types — 4 columns

Four columns are stored in the wrong format, which means they can't be used correctly until they're fixed. Think of it like storing a date as "November 8 2025" instead of `2025-11-08` — a computer can read the text, but it can't do date math on it.

| Column | Stored as | Should be | Why it matters |
|---|---|---|---|
| `createdAt` | text (`str`) | date (`datetime64`) | Can't sort by time, group by month, or plot trends without this fix |
| `isReply` | mixed text + bool | `bool` | Inconsistent — some rows say `True`, others say `"True"` as a string. Comparisons will silently fail |
| `pseudo_inReplyToUsername` | number (`float64`) | text (`str`) | IDs became decimals (e.g. `82000787115977.0`) when pandas added NaNs. Loses precision |
| `author_isBlueVerified` | mixed text + bool | `bool` | Same mixed serialisation issue as `isReply` |

```python
df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
df["isReply"] = df["isReply"].astype(bool)
df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)
df["pseudo_inReplyToUsername"] = df["pseudo_inReplyToUsername"].astype("Int64").astype(str)
```

#### 🟡 Inconsistent Values — 2 columns

**`lang` — 410 rows with unexpected language code `'und'`**

Twitter assigns `'und'` (undetermined) when it can't detect the language. Only `'tl'` (Filipino) and `'en'` (English) were expected. The breakdown is:

| Language | Count | % |
|---|---|---|
| `tl` (Filipino) | 126,090 | 64.4% |
| `en` (English) | 69,244 | 35.4% |
| `und` (Undetermined) | 410 | 0.2% |

The `'und'` rows are a small fraction. They should be excluded from language-based analysis or grouped under an `'other'` category.

**`pseudo_inReplyToUsername` — 72,531 values stored as float**

This is a direct consequence of the dtype issue above. Once the column is properly cast to `str`, the float-formatted IDs (`82000787115977.0`) will be fixed.

---

## Dataset 2: Authors

### 2.1 How Big Is the Dataset?

![Dataset Shape](output/well_known_authors_dpwh_floodcontrol_dataset_shape.png)

Just **227 authors** drove nearly 200,000 tweets. That's a remarkably small and focused group — an average of roughly 860 tweets per author. This is not a random sample of Twitter; it's a curated set of accounts that Twitter's algorithm or the researcher identified as influential voices on this topic.

---

### 2.2 What Does Each Column Contain?

![Column Names and Data Types](output/well_known_authors_dpwh_floodcontrol_column_names_and_dtypes.png)

The author dataset is simpler — 8 columns covering identity, reach, and a bit of context about who these accounts are.

| Column | Type | What it means |
|---|---|---|
| `author_userName` | `str` | The Twitter handle (e.g. `@ABSCBNNews`) |
| `author_createdAt` | `str` ⚠️ | When the account was created — stored as text, should be a date |
| `obfuscated_userName` | `str` | An anonymised version of the username |
| `author_profile_bio_description` | `str` | The author's Twitter bio |
| `author_location` | `str` | Self-reported location — unreliable (see Section 2.3) |
| `author_followers` | `int64` | Number of followers |
| `author_following` | `int64` | Number of accounts they follow |
| `author_isBlueVerified` | `bool` ⚠️ | Whether the author has Twitter Blue verification |

> ⚠️ = columns with a type issue. Covered in Section 2.4.

**dtype summary:** 5 `str` · 2 `int64` · 1 `bool`

---

### 2.3 Is Any Data Missing?

![Missing Data](output/well_known_authors_dpwh_floodcontrol_missing_data.png)

**2 of 8 columns have missing values — and in both cases, the user chose not to fill them in.**

This is different from the tweets dataset. Here, the missing values aren't structural — the information *could* exist, but the author decided not to share it. In data science, this is called **Missing Not At Random (MNAR)**: the fact that it's missing tells you something about the person (they value privacy, or they didn't bother).

#### `author_location` — 40 missing (17.6%)

Twitter's location field is self-reported and completely optional. 40 authors (17.6%) left it blank. And among those who did fill it in, some entries are not real locations — we'll cover that in Section 2.4.

> **Decision: Retain; fill blank values with `"Unknown"`.** Filling with a placeholder keeps the row in the dataset for other analysis (engagement, follower counts, etc.) while clearly marking that location data is unavailable. We can't meaningfully guess someone's location.

#### `author_profile_bio_description` — 6 missing (2.6%)

6 authors left their bio blank. This is a minor gap and only affects text-based analysis of author profiles.

> **Decision: Retain; fill blank values with `"No bio provided"`.** Only 6 rows are affected. The bio is descriptive and non-critical — a placeholder is appropriate.

**The other 6 columns are fully complete.** No rows need to be removed.

---

### 2.4 Is the Data Trustworthy?

![Data Quality Report](output/well_known_authors_dpwh_floodcontrol_data_quality.png)

The author dataset is in good shape — **0 issues, 3 warnings.** The warnings are minor and easy to fix.

#### 🟡 Wrong Data Types — 2 columns

| Column | Stored as | Should be | Why it matters |
|---|---|---|---|
| `author_createdAt` | text (`str`) | date (`datetime64`) | Can't calculate account age or group authors by when they joined without this fix |
| `author_isBlueVerified` | mixed text + bool | `bool` | Same mixed serialisation issue as in the tweets dataset |

```python
authors["author_createdAt"] = pd.to_datetime(authors["author_createdAt"], utc=True)
authors["author_isBlueVerified"] = authors["author_isBlueVerified"].astype(bool)
```

#### 🟡 Inconsistent Values — 1 column

**`author_location` — 16 entries that aren't real locations (7.05%)**

Twitter's location field is a free-text box. Authors can write anything. Among the 227 authors, 16 wrote something that isn't a geographic place:

| Entry | Count |
|---|---|
| "Earth" | 2 |
| "Around The World" | 1 |
| "facebook.com/aidelacruzonline" | 1 |
| "Abbott Elementary" | 1 |
| "WhatsApp & Telegram" | 1 |

These aren't data collection errors — the authors really did write these things. But they can't be used in geographic analysis. They should be recoded as `"Unknown"` alongside the actual missing values.

After cleaning, **170 authors (74.9%)** have a usable geographic location.

---

## Summary: What Needs to Be Fixed Before Analysis

Before moving into univariate distributions, temporal trends, or any deeper analysis, these preprocessing steps should be applied:

### Tweets Dataset

```python
# 1. Drop the single duplicate
df = df.drop_duplicates(subset=["pseudo_id"])

# 2. Fix data types
df["createdAt"] = pd.to_datetime(df["createdAt"], utc=True)
df["isReply"] = df["isReply"].astype(bool)
df["author_isBlueVerified"] = df["author_isBlueVerified"].astype(bool)
df["pseudo_inReplyToUsername"] = df["pseudo_inReplyToUsername"].astype("Int64").astype(str)

# 3. Handle 'und' language codes
df["lang"] = df["lang"].replace("und", "other")
```

### Authors Dataset

```python
# 1. Fix data types
authors["author_createdAt"] = pd.to_datetime(authors["author_createdAt"], utc=True)
authors["author_isBlueVerified"] = authors["author_isBlueVerified"].astype(bool)

# 2. Standardise missing and invalid locations
invalid_locations = ["Earth", "Around The World", "facebook.com/aidelacruzonline",
                     "Abbott Elementary", "WhatsApp & Telegram"]
authors["author_location"] = authors["author_location"].replace(invalid_locations, "Unknown")
authors["author_location"] = authors["author_location"].fillna("Unknown")

# 3. Fill missing bios
authors["author_profile_bio_description"] = (
    authors["author_profile_bio_description"].fillna("No bio provided")
)
```

After these steps, both datasets are clean and ready for analysis.
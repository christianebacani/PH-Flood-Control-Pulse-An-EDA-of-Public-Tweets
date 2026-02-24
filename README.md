# PH Flood Control Pulse: An EDA of Public Tweets

This project provides Exploratory Data Analysis for [public tweets of well-known Twitter authors regarding about the PH Flood Control issue of the DPWH](https://www.kaggle.com/datasets/bwandowando/tweets-on-dpwh-and-flood-control-projects-2025) (Department of Public Works and Highways).

---

## Dataset 1: For Export (DPWH Flood Control Tweets)

### 1.1 Dataset Shape

![Dataset Shape](output/for_export_dpwh_floodcontrol_dataset_shape.png)

> The dataset contains **195,744 rows** and **16 columns**, indicating a large
> volume of tweet data collected for analysis.

### 1.2 Column Names & Data Types

![Column Names and Data Types](output/for_export_dpwh_floodcontrol_column_names_and_dtypes.png)

> The dataset consists mostly of `int64` columns (9), followed by `str` (3),
> `float64` (2), and `bool` (2). Engagement metrics such as `retweetCount`,
> `likeCount`, and `viewCount` are all numeric, while `text` and `lang` are string columns.

### 1.3 First Few Rows

![First Few Rows](output/for_export_dpwh_floodcontrol_first_few_rows.png)

> A preview of the first 3 rows shows tweets in both English (`en`) and Tagalog (`tl`),
> with most having low engagement counts. The `text` column contains the raw tweet content.

### 1.4 Missing Data Analysis

![Missing Data](output/for_export_dpwh_floodcontrol_missing_data.png)

> **2 of 16 columns** have missing values. Both are structurally valid cases of
> **MCAR (Missing Completely At Random)** — the missingness is by design, not data error.
>
> | Column | Missing | % | Type | Decision |
> |---|---|---|---|---|
> | `pseudo_inReplyToUsername` | 123,213 | 62.9% | MCAR | **Keep** — `NaN` means the tweet is not a reply |
> | `quoted_pseudo_id` | 170,346 | 87.0% | MCAR | **Keep** — `NaN` means the tweet is not a quote tweet |
>
> No rows need to be removed. The remaining **14 columns are clean** with 0% missing values.

---

## Dataset 2: Well Known Authors (DPWH Flood Control)

### 2.1 Dataset Shape

![Dataset Shape](output/well_known_authors_dpwh_floodcontrol_dataset_shape.png)

> This dataset contains **227 rows** and **8 columns**, representing a curated
> list of well-known Twitter authors who tweeted about the DPWH flood control issue.

### 2.2 Column Names & Data Types

![Column Names and Data Types](output/well_known_authors_dpwh_floodcontrol_column_names_and_dtypes.png)

> The dataset is predominantly `str` columns (5), with `int64` (2) for follower/following
> counts and `bool` (1) for blue verification status.

### 2.3 First Few Rows

![First Few Rows](output/well_known_authors_dpwh_floodcontrol_first_few_rows.png)

> The first 3 rows reveal high-profile accounts with large follower counts (632K, 254K, 1.7M).
> Notable accounts include a radio host, a lawyer, and ABCWorldNews — suggesting
> the dataset captures both local and international voices.

### 2.4 Missing Data Analysis

![Missing Data](output/well_known_authors_dpwh_floodcontrol_missing_data.png)

> **2 of 8 columns** have missing values. Both are user-generated fields where
> missingness reflects the author simply not filling in their profile — classified
> as **MCAR (Missing Completely At Random)**.
>
> | Column | Missing | % | Type | Decision |
> |---|---|---|---|---|
> | `author_profile_bio_description` | 6 | 2.6% | MCAR | **Keep** — fill with `"No bio provided"` for display purposes |
> | `author_location` | 40 | 17.6% | MCAR | **Keep** — fill with `"Unknown"` for display purposes |
>
> No rows need to be removed. The remaining **6 columns are clean** with 0% missing values.

---
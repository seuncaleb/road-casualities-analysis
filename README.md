# Road Collision Analysis — Great Britain (2024)

An end-to-end data analysis and machine learning project using the UK Department for Transport's 2024 Road Safety dataset. The project explores collision patterns through exploratory data analysis, unsupervised clustering, and supervised classification.

---

## Overview

Road casualties in Great Britain have been recorded by police since 1979. The 2024 dataset represents the lowest recorded fatality count outside of COVID-affected years. Despite this long-run decline, understanding *why* collisions happen and *who* is most at risk remains important for designing targeted safety interventions.

This project takes a data-driven approach to that question. Using vehicle-level collision records, it:

- Identifies patterns across driver demographics, vehicle types, and road manoeuvres
- Groups collisions into behavioural clusters using unsupervised learning
- Builds a classifier to predict the first point of vehicle impact

---

## Dataset

**Source:** [Department for Transport — Road Safety Open Data](https://www.gov.uk/government/statistical-data-sets/road-safety-open-data)

**File used:** `dft-road-casualty-statistics-vehicle-2024.csv`

The raw dataset contains vehicle-level records from police-reported collisions across Great Britain in 2024. A companion codebook (`2024_code_list.csv`) was used to decode numerical category codes into readable labels.

After cleaning and filtering for informative records, the working dataset contained approximately **1,727 usable records** across **15 features**.

---

## Methods

### 1. Data Cleaning and Preprocessing

- Loaded raw data and used the codebook to decode numeric category codes into human-readable labels
- Dropped 17 irrelevant or highly sparse columns
- Replaced out-of-range driver age values with the column mean
- Filtered records where key variables contained uninformative entries (e.g. "Data missing or out of range", "Not known")

### 2. Exploratory Data Analysis

Four visualisations were produced to surface initial patterns:

| Figure | Description |
|--------|-------------|
| 2.1 | Proportional manoeuvres by junction location |
| 2.2 | Manoeuvre count by driver sex |
| 2.3 | Distribution of driver age by vehicle type |
| 2.4 | Proportional first point of impact by driver age band |

### 3. Unsupervised Learning — K-Means Clustering

- One-hot encoded 13 categorical variables (15 features expanded to 109 after encoding)
- Normalised with `StandardScaler`
- Applied K-Means with `k=4` clusters and `n_init=10`
- Profiled each cluster to identify distinct collision behaviour groups

### 4. Supervised Learning — Decision Tree Classifier

- **Target variable:** First point of impact (front, nearside, offside, back, did not impact)
- 80/20 stratified train/test split — 1,381 training samples, 346 test samples
- `DecisionTreeClassifier` with `max_depth=10` to reduce overfitting

**Results:**

| Metric | Value |
|--------|-------|
| Overall accuracy | 67.3% |
| Nearside precision | 0.74 |
| Nearside recall | 0.91 |

**Top predictive features:** vehicle type, driver sex, direction of travel, reversing manoeuvre, wall or fence contact

The model performed well for the dominant nearside class but struggled with rarer impact types — a known class imbalance limitation discussed in the report.

---

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, cleaning, transformation |
| `numpy` | Numerical operations, missing value handling |
| `matplotlib` | Exploratory visualisations |
| `scikit-learn` | StandardScaler, KMeans, DecisionTreeClassifier, train_test_split, classification metrics |
| `re` | Feature name cleaning for display |

---


```
road-casualities-analysis/
├── Road_Casualities.ipynb    # Full analysis notebook
├── road_casualities.pdf      # Written report with methodology and findings
└── README.md                 # This file
```

> The raw data files (`dft-road-casualty-statistics-vehicle-2024.csv` and `2024_code_list.csv`) are not included due to file size. Download them from the DfT Road Safety Open Data portal linked above.

---

## Key Findings

- Nearside impacts are the most common first point of contact across all driver demographics
- Younger drivers (16–25) show a higher proportion of front-impact collisions at junctions
- Male drivers appear in significantly more collision records across all manoeuvre types
- Vehicle type is the single strongest predictor of impact point in the classifier
- Reversing manoeuvres and wall/fence contact objects are strongly associated with specific cluster profiles



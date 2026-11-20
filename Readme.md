# Pasture Biomass Prediction — CSIRO Biomass Competition

Multi-target computer vision regression pipeline for estimating pasture biomass (in grams) from field imagery, evaluated using the **R² Score** metric. Accurate biomass estimation enables farmers to optimize livestock feed allocation and prevent pasture degradation.

## Key Highlights

- **Comprehensive Exploratory Data Analysis:** Analyzed target distributions, inter-target correlation structure, physical target additivity, and field metadata (`State`, `Species`, `Pre_GSHH_NDVI`, `Height_Ave_cm`) across 357 unique pasture images.
- **Backbone & Architecture:** `BiomassEVA02TargetQueryModel` built on **EVA-02** (`eva02_tiny_patch14_224`) with target-specific query heads for simultaneous 5-target regression.
- **Leak-Free Validation:** **5-Fold Group Cross-Validation** (`GroupKFold`) grouped by original image ID (`group_id`) to prevent data leakage across augmented variants.
- **Strong Multi-Target Performance:** Achieves **0.9923 Overall Out-of-Fold (OOF) Mean R²** across all five dry matter and green biomass targets.

## Dataset & Exploratory Data Analysis

### Dataset Overview
- **Source:** CSIRO Pasture Biomass Dataset (`data/csiro-biomass/`)
- **Raw Structure (`train.csv` & `test.csv`):** `train.csv` contains **1,785 rows** in long format (**357 unique images** × 5 target variables), pivoted into wide format (1 row per image) for multi-target modeling.
- **Field Metadata (`meta_df`):** Includes Australian geographic state (`State`), pasture species composition (`Species`), Normalized Difference Vegetation Index (`Pre_GSHH_NDVI`), and average sward height (`Height_Ave_cm`).

### Target Variables & Summary Statistics (`n = 357` unique images)

All targets exhibit right-skewed distributions—most notably `Dry_Clover_g`, where over 25% of samples contain `0.00 g`:

| Target Column | Description | Mean ± Std (g) | Median (g) | Min – Max (g) |
| :--- | :--- | :---: | :---: | :---: |
| `Dry_Clover_g` | Dry clover biomass | 6.65 ± 12.12 | 1.42 | 0.00 – 71.79 |
| `Dry_Dead_g` | Dry dead plant matter biomass | 12.04 ± 12.40 | 7.98 | 0.00 – 83.84 |
| `Dry_Green_g` | Dry green vegetation biomass | 26.62 ± 25.40 | 20.80 | 0.00 – 157.98 |
| `Dry_Total_g` | Total dry pasture biomass | 45.32 ± 27.98 | 40.30 | 1.04 – 185.70 |
| `GDM_g` | Green Dry Matter biomass | 33.27 ± 24.94 | 27.11 | 1.04

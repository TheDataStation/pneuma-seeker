# Source

The datasets in this directory are derived from the KramaBench benchmark:

https://github.com/mitdbg/Kramabench

We copied the datasets from the KramaBench repository and applied additional preprocessing steps, described below.

If you use these datasets, please also cite the KramaBench repository.

# Dataset Download

Some datasets are too large to include in this repository. You can download the CSV files from the following link:

https://drive.google.com/drive/folders/1jL2T8N8X8ncFyERdZ1lg0DcKZDwFqKaG?usp=sharing

# Directory Layout

Each domain (`archeology`, `astronomy`, `biomedical`, `environment`, `legal`, `wildfire`) follows the same structure:

- `<domain>_tabular.json`: the questions, each with an `id`, the `query`, the reference `answer`, and the `data_sources` needed to answer it.
- `dataset/`: the CSV tables for the domain.
- `metadata.csv`: a short description of each table (if any).

# Preprocessing

The following preprocessing steps were applied to the datasets relative to the original KramaBench version:

- Extract sheets in XLSX files into CSV files, including metadata when available (for example, README sheets within XLSX files).
- Remove non-CSV files and add missing column names to certain tables (for example, the `C-SE-phospho` sheet in `1-s2.0-S0092867420301070-mmc3.xlsx`).
- Some CSV files contain multiple logical tables. These were split into individual tables. For example, `2024_CSN_Fraud, Identity Theft, and Other Reports by Military Consumers.csv` contains separate breakdowns by military status, branch, and rank.

# Evaluation Metrics

**Answer Quality** is the average score across questions. Each question is run 3 times and keeps its best score (best-of-3, which extends pass@3 to fractional scores).

A run scores 1 if its answer is correct and 0 otherwise. For questions whose answers are sets or lists, a run is scored with F1 against the reference list. Two questions use a question-specific score:

- `biomedical-hard-8`: the reference answer lists genes (`CDK12`, `SMARCA4`), so site-level answers such as `SMARCA4-S1484` are matched by their gene. This question is scored by recall: the fraction of reference genes found.
- `wildfire-hard-18`: the answer has two claims (whether suppression shortens fire duration and whether it reduces buildings affected), and each claim counts for half.

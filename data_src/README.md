# Source

The datasets in this directory are derived from the KramaBench benchmark:

https://github.com/mitdbg/Kramabench

We copied the datasets from the KramaBench repository and applied additional preprocessing steps, described below.

If you use these datasets, please also cite the KramaBench repository.

# Dataset Download

Some datasets are too large to include in this repository. You can download the CSV files from the following link:

https://drive.google.com/drive/folders/1jL2T8N8X8ncFyERdZ1lg0DcKZDwFqKaG?usp=sharing

# Preprocessing

The following preprocessing steps were applied to the datasets relative to the original KramaBench version:

- Extract sheets in XLSX files into CSV files, including metadata when available (for example, README sheets within XLSX files).
- Remove non-CSV files and add missing column names to certain tables (for example, the `C-SE-phospho` sheet in `1-s2.0-S0092867420301070-mmc3.xlsx`).
- Some CSV files contain multiple logical tables. These were split into individual tables. For example, `2024_CSN_Fraud, Identity Theft, and Other Reports by Military Consumers.csv` contains separate breakdowns by military status, branch, and rank.

# Evaluation Metrics

- Answer Quality: best-of-3 (max score of 3 runs per question, averaged across questions; extends pass@3 to fractional scores)

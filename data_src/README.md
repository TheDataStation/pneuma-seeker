# Source

The datasets in this directory are derived from two benchmarks (please cite them if you use these datasets):

- KramaBench (`archeology`, `astronomy`, `biomedical`, `environment`, `legal`, `wildfire`): https://github.com/mitdbg/Kramabench
- DataAgentBench (`bookreview`, `crmarenapro`, `deps_dev_v1`, `github_repos`, `googlelocal`, `imdb`, `music_brainz_20k`, `pancancer_atlas`, `patents`, `stockindex`, `stockmarket`): https://github.com/ucbepic/DataAgentBench

We copied the datasets from these repositories and applied additional preprocessing steps, described below.

# Dataset Download

To download the tables that are not in the repository (the larger KramaBench datasets, Astronomy and Wildfire, and the DataAgentBench database files) and start the DataAgentBench Postgres databases (loaded from the `.sql` files on first start), run:

```bash
data_src/download.sh
docker compose -f data_src/docker-compose.yml up -d
```

To register the datasets into Pneuma-Seeker, with their metadata, run:

```bash
docker compose exec core python -m pneuma_seeker.services.db.datasets.benchmarks
```

# Directory Layout

Each KramaBench domain follows the same structure:

- `<domain>_tabular.json`: the questions, each with an `id`, the `query`, the reference `answer`, and the `data_sources` needed to answer it.
- `dataset/`: the CSV tables for the domain.
- `metadata.csv`: a short description of each table (if any).

Each DataAgentBench benchmark follows the same structure:

- `benchmark.json`: the questions, each with an `id`, the `question`, and the reference `answer`.
- `dataset/`: the database files (SQLite, DuckDB, or a Postgres `.sql` dump).
- `databases.json`: each database's type (`sqlite`, `duckdb` or `postgres`), its file in `dataset/`, and for Postgres its connection.
- `metadata.csv`: the metadata of each database, table and column (`database`, `table_name`, `column_name`, `description`; an empty `table_name` describes the database, an empty `column_name` the table).

# Preprocessing

The following preprocessing steps were applied to the datasets relative to the original KramaBench version:

- Extract sheets in XLSX files into CSV files, including metadata when available (for example, README sheets within XLSX files).
- Remove non-CSV files and add missing column names to certain tables (for example, the `C-SE-phospho` sheet in `1-s2.0-S0092867420301070-mmc3.xlsx`).
- Some CSV files contain multiple logical tables. These were split into individual tables. For example, `2024_CSN_Fraud, Identity Theft, and Other Reports by Military Consumers.csv` contains separate breakdowns by military status, branch, and rank.

Relative to the original DataAgentBench version:

- The datasets that use MongoDB (`agnews`, `civic_unstructured`, `cve`, `krama`, `usaspending`, `yelp`) are excluded, leaving 11 datasets and 53 questions.
- Each question's `query.json` and `ground_truth.csv` are combined into `benchmark.json`, with the question and answer unchanged.
- Each dataset's `db_description.txt` is converted into `metadata.csv`. The hints in `db_description_withhint.txt` are not included.

# Evaluation Metrics

**Answer Quality** is the average score across questions. Each question is run 3 times and keeps its best score (best-of-3, which extends pass@3 to fractional scores).

A run scores 1 if its answer is correct and 0 otherwise. For questions whose answers are sets or lists, a run is scored with F1 against the reference list. Two questions use a question-specific score:

- `biomedical-hard-8`: the reference answer lists genes (`CDK12`, `SMARCA4`), so site-level answers such as `SMARCA4-S1484` are matched by their gene. This question is scored by recall: the fraction of reference genes found.
- `wildfire-hard-18`: the answer has two claims (whether suppression shortens fire duration and whether it reduces buildings affected), and each claim counts for half.

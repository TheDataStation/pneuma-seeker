# Pneuma-Seeker

[![arXiv](https://img.shields.io/badge/arXiv-2603.10747-b31b1b?logo=arxiv)](https://arxiv.org/abs/2603.10747)
[![Demo](https://img.shields.io/badge/YouTube-Demo%20Video-red?logo=youtube)](https://youtu.be/ZNAhTE6n0yI)

`Pneuma-Seeker` is an LLM-powered system for answering questions over tabular data. Given a question, it automatically finds relevant tables, combines them, and runs the necessary computations to produce an answer.

# Installation

To install and run `Pneuma-Seeker`, you need to set up both the backend and frontend.

First, copy the environment configuration:
```bash
cp .env.example .env
```
Then update values as needed. See [the configuration file](./src/pneuma_seeker/shared/config.py) for all available options.

Next, clone the frontend repository:
```bash
git clone https://github.com/TheDataStation/pneuma-seeker-ui.git
```

To spin up the entire system using Docker, run:
```bash
docker compose up -d
cd pneuma-seeker-ui
docker compose up -d
```

# Development

## Frontend Development

For frontend development guidelines, refer to the [pneuma-seeker-ui README](./pneuma-seeker-ui/README.md).

## Backend Development

We recommend using `Miniconda` (see [installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)) to manage dependencies. Create and activate a new environment using:
```bash
conda create --name pneuma_seeker python=3.12.12 -y
conda activate pneuma_seeker
pip install -r requirements.txt
```

Authentication and authorization depend on Postgres. You can spin up a local instance using Docker:
```bash
docker compose up postgres -d
```

> Note: If you have a local Postgres instance running, make sure to update the `POSTGRES_HOST` variable in your `.env` file accordingly (e.g., to `localhost`).

Start the backend server:
```bash
cd src/pneuma_seeker
```
On macOS:
```bash
fastapi dev main.py > main.out 2>&1
```
Linux/other:
```bash
nohup fastapi dev main.py --host 0.0.0.0 --port 8000 >> main.out &
```

To run unit tests:
```bash
# Run from the project root
python -m pytest tests/
```

### Running the UI locally
Navigate to the cloned UI repository and start the development server:
```bash
cd pneuma-seeker-ui
npm run dev
```

# Next Steps
Before asking questions on a dataset, you need to index it using the `/index` endpoint in the backend (see [indexing.py](src/pneuma_seeker/routers/indexing.py)).

For a complete walkthrough of the system, including dataset indexing and query execution, refer to [quick_start.ipynb](./quick_start.ipynb), which demonstrates how to configure Pneuma-Seeker on a sample dataset.

For a deeper understanding of the system architecture, check out the [architecture overview](/docs/architecture.md) in the `/docs` directory.

# Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute, report issues, and submit pull requests.

# Contact

For questions or support regarding `Pneuma-Seeker`, please email: pneuma-team@googlegroups.com

# Citation

## [Pneuma-Seeker paper](https://arxiv.org/abs/2603.10747)
```
@misc{PneumaSeeker2026,
      title={Pneuma-Seeker: A Relational Reification Mechanism to Align AI Agents with Human Work over Relational Data}, 
      author={Muhammad Imam Luthfi Balaka and John Hillesland and Kemal Badur and Raul Castro Fernandez},
      year={2026},
      eprint={2603.10747},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2603.10747}, 
}
```

## [Pneuma-Seeker demo paper](https://dl.acm.org/doi/10.1145/3786335.3813215)
```
@inbook{PneumaSeekerDemo2026,
      author = {Balaka, Muhammad Imam Luthfi and Castro Fernandez, Raul},
      title = {Demonstration of Pneuma-Seeker: An Agentic System for Reifying and Fulfilling Information Needs on Tabular Data},
      year = {2026},
      isbn = {9798400724152},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3786335.3813215},
      booktitle = {Proceedings of the ACM Conference on AI and Agentic Systems},
      pages = {1199–1203},
      numpages = {5}
}
```

## [The Pneuma project paper](https://www.cidrdb.org/cidr2026/papers/p31-balaka.pdf)
```
@inproceedings{PneumaProjectCIDR2026,
      author    = {Muhammad Imam Luthfi Balaka and Raul Castro Fernandez},
      title     = {The Pneuma Project: Reifying Information Needs as Relational Schemas to Automate Discovery, Guide Preparation, and Align Data with Intent},
      booktitle = {Proceedings of the 16th Annual Conference on Innovative Data Systems Research (CIDR '26)},
      year      = {2026},
}
```

## [Pneuma (data discovery system) paper](https://dl.acm.org/doi/10.1145/3725337)
```
@article{PneumaDataDiscovery2025,
      author = {Balaka, Muhammad Imam Luthfi and Alexander, David and Wang, Qiming and Gong, Yue and Krisnadhi, Adila and Castro Fernandez, Raul},
      title = {Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System},
      year = {2025},
      issue_date = {June 2025},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      volume = {3},
      number = {3},
      url = {https://doi.org/10.1145/3725337},
      doi = {10.1145/3725337},
      journal = {Proc. ACM Manag. Data},
      month = jun,
      articleno = {200},
      numpages = {28},
      keywords = {data discovery, large language models, natural-language questions}
}
```

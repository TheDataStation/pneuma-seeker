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

To install and run the backend, you can simply run:
```bash
docker compose up -d
```

However, for development purposes, we **recommend** installing `Miniconda` (see [installation guide](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)). Then, create a new environment using:
```bash
conda create --name pneuma_seeker python=3.12.12 -y
conda activate pneuma_seeker
pip install -r requirements.txt
```

Start the backend server:
```bash
cd src/pneuma_seeker
```
On macOS:
```bash
fastapi dev main.py > main.out 2>&1
```
Otherwise:
```bash
nohup fastapi dev main.py --host 0.0.0.0 --port 8000 >> main.out &
```

Clone and install the UI:
```bash
git clone https://github.com/luthfibalaka/pneuma-seeker-ui.git
cd pneuma-seeker-ui
git checkout stable-0.6.22
pip install .
cd ..
```

Start the UI (OpenWebUI):
```bash
cd pneuma-seeker-ui
nohup open-webui serve >> output.out &
```

In the OpenWebUI interface, create and log into an admin account. Then, import all functions (`.json` files) in `/openwebui_functions` so that the UI can communicate with the Pneuma-Seeker backend. Enable all imported functions and select `PneumaSeeker` as the model for chat.

![Import functions to OpenWebUI](docs/figures/openwebui_import.png)
![Enable functions in OpenWebUI](docs/figures/openwebui_function_list.png)
![Select Pneuma in OpenWebUI](docs/figures/openwebui_chat_interface.png)

# Next Steps
Before asking questions on a dataset, you need to index it using the `/index` endpoint in the backend (see [main.py](src/pneuma_seeker/main.py)).

For a complete walkthrough of the system, including dataset indexing and query execution, refer to [quick_start.ipynb](./quick_start.ipynb), which demonstrates how to configure Pneuma-Seeker on a sample dataset.

For a deeper understanding of the system, refer to the documentation in `/docs`, including the [architecture overview](/docs/architecture.md) of `Pneuma-Seeker`.

# Contributing

We welcome contributions. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute, report issues, and submit pull requests.

# Contact

For questions about `Pneuma-Seeker`, please email: pneuma-team@googlegroups.com

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

## [Pneuma-Seeker demo paper](https://arxiv.org/abs/2604.14422)
```
@misc{balaka2026demonstrationpneumaseekeragenticreifying,
      title={Demonstration of Pneuma-Seeker: Agentic System for Reifying and Fulfilling Information Needs on Tabular Data}, 
      author={Muhammad Imam Luthfi Balaka and Raul Castro Fernandez},
      year={2026},
      eprint={2604.14422},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2604.14422}, 
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

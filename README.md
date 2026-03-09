![The Architecture of Pneuma-Seeker](pneuma_seeker.png)

# Pneuma-Seeker

A system that reifies an _active_ information needs on tabular data as a relational data model $(\mathcal{T},S)$, where $\mathcal{T}$ is a set of views derived from the underlying dataset (table collection), and $S$ is a Python script over $\mathcal{T}$, and fulfills it by executing $S$.

## How to Run the System

Prepare the environment as follows:
```bash
conda create --name pneuma_seeker python=3.12.9
pip install -r requirements.txt
```

You may change any configuration through `.env` (refer to `src/pneuma_seeker/shared/config.py` to know all available configs). Then, run the system using:

```bash
cd src/pneuma_seeker
nohup fastapi dev main.py --host 0.0.0.0 --port 8000 >> main.out &
```

On MacOS, FastAPI does not work with nohup, so we use:
```bash
fastapi dev main.py > main.out 2>&1
```

To run the frontend, do the following:
```bash
cd ..
git clone https://github.com/luthfibalaka/pneuma-seeker-ui.git
cd pneuma-seeker-ui
git checkout stable-0.6.22
pip install .
nohup open-webui serve >> output.out &
```
Then, import all functions in the `openwebui_functions` directory to the OpenWebUI interface so that it can call the backend (`Pneuma-Seeker`)

## How to Run Unit Tests

```bash
cd ./tests/pneuma_seeker
python -m unittest discover
```

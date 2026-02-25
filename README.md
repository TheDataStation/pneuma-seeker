![The Architecture of Pneuma-Seeker](pneuma_seeker.png)

# Pneuma-Seeker

A system that helps users identify and fulfill their latent information needs.

## How to Run the System

```bash
conda create --name pneuma_seeker python=3.12.9
pip install -r requirements.txt
cd src/pneuma_seeker
nohup fastapi dev main.py --host 0.0.0.0 --port 8000 >> main.out &
```

Alternatively on MacOS,
```bash
fastapi dev main.py > main.out 2>&1
```

## Extra: How to Run the ([UI](https://github.com/luthfibalaka/pneuma-seeker-ui/tree/stable-0.6.22))

```bash
nohup open-webui serve >> output.out &
```

## How to Test the System

```bash
cd ./tests/pneuma_seeker
python -m unittest discover
```

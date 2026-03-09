# Setup
```bash
conda create --name smolagents python=3.12
pip install -r requirements.txt
```

# Run
Adjust configurations in `e2e.py` (e.g., which model to use or which benchmark to use), run
```bash
nohup python e2e.py >> e2e.out &
```

# Installation Instructions

## Option A: Native Python (recommended for development)

### 1. Prerequisites
- Python 3.12+ (`python3 --version`)
- `pip` and `venv` (bundled with Python)
- ~2 GB free disk space

### 2. Clone and set up a virtual environment
```bash
git clone <this-repo>
cd Healthcare-Readmission-DevSecMLOps
python3 -m venv venv
source venv/bin/activate      # Windows (PowerShell): venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. (Optional) Install git hooks
```bash
pre-commit install
```

### 5. Verify the install
```bash
export PYTHONPATH=.            # Windows (PowerShell): $env:PYTHONPATH="."
python -c "import fastapi, sklearn, xgboost, lightgbm, catboost, mlflow, evidently; print('OK')"
```

### 6. Run the pipeline and start the API
See `docs/DEVELOPER_GUIDE.md` for the full stage-by-stage sequence, or the
condensed version in the top-level `README.md` Quick Start.

## Option B: Docker (recommended for deployment / no local Python needed)

### 1. Prerequisites
- Docker Engine 24+
- Docker Compose v2 (`docker compose version`)

### 2. Build and run
```bash
git clone <this-repo>
cd Healthcare-Readmission-DevSecMLOps

# One-time: run the pipeline image to produce trained model artifacts
docker build -f docker/Dockerfile.pipeline -t healthcare-readmission-pipeline .
docker run --rm -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" healthcare-readmission-pipeline

# Bring up the full stack: API + Prometheus + Grafana
docker compose up --build
```

### 3. Verify
```bash
curl http://localhost:8000/health
```
Open http://localhost:8000/docs for the interactive API explorer,
http://localhost:9090 for Prometheus, and http://localhost:3000
(`admin`/`admin`) for Grafana.

## Option C: Cloud (AWS ECS Fargate via Terraform)

Requires Terraform 1.7+ and an AWS account/credentials. See
`docs/DEPLOYMENT_GUIDE.md` → "Option 2" for the full `terraform init / plan
/ apply` sequence and required variables (`terraform/terraform.tfvars.example`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Set `PYTHONPATH=.` (or `$env:PYTHONPATH="."` on Windows) before running any script |
| `/health` returns `"status": "degraded"` | The model hasn't been trained yet — run `python src/training/train.py` first (needs `preprocess.py` to have run before it) |
| `xgboost`/`lightgbm`/`catboost` import errors on Linux | Install `libgomp1` (`apt-get install libgomp1`) — already handled in the provided Dockerfiles |
| Safety scan fails with an auth error | The modern Safety CLI requires a free API key from safetycli.com — set `SAFETY_API_KEY`, or skip it locally (CI skips gracefully if unset) |

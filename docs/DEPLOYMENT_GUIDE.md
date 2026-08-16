# Deployment Guide

## Option 1: Docker Compose (local / single-host)

Prerequisite: a trained model. Either train locally first (see
`docs/DEVELOPER_GUIDE.md`) so `models/*.joblib` exist before building the
image, or run the pipeline image once to populate `models/`.

```bash
# 1. Run the full offline pipeline once to produce models/*.joblib
docker build -f docker/Dockerfile.pipeline -t healthcare-readmission-pipeline .
docker run --rm -v "$(pwd)/models:/app/models" -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/reports:/app/reports" healthcare-readmission-pipeline

# 2. Bring up the API + Prometheus + Grafana
docker compose up --build
```

Services:
- API: http://localhost:8000 (`/docs` for Swagger UI)
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (default `admin` / `admin`, change on
  first login)

Stop with `docker compose down`; add `-v` to also drop the Prometheus/Grafana
volumes.

## Option 2: AWS ECS Fargate via Terraform

The `terraform/` directory provisions: an ECR repository, an ECS Fargate
cluster/service/task, an Application Load Balancer, CloudWatch logging, and
the IAM execution role — using the account's default VPC to stay
demo-friendly.

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # edit values as needed
terraform init
terraform plan -var-file="terraform.tfvars"
terraform apply -var-file="terraform.tfvars"

# Push your built image to the ECR repo Terraform created:
aws ecr get-login-password --region <region> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
docker tag healthcare-readmission-api:latest <ecr_repository_url>:latest
docker push <ecr_repository_url>:latest

# Then force a new deployment to pick up the pushed image:
aws ecs update-service --cluster <ecs_cluster_name> \
  --service <ecs_service_name> --force-new-deployment
```

`terraform output api_url` gives you the public ALB URL once the service is
healthy. Default sizing (`task_cpu=512`, `task_memory=1024`,
`desired_count=1`) is intentionally modest — bump these in
`terraform.tfvars` for real traffic, and consider adding an
`aws_appautoscaling_target`/`policy` pair for CPU-based autoscaling in a
production fork.

Destroy with `terraform destroy -var-file="terraform.tfvars"` when done, to
avoid ongoing ALB/Fargate charges.

## Option 3: CI/CD-Driven Deployment

`.github/workflows/ci-cd.yml` runs lint → security scan → tests → Docker
build/Trivy scan/smoke-test → push to GHCR → deploy → post-deploy smoke
tests on every push to `main`. The `deploy` job is a placeholder step ready
to be wired to `terraform apply`, an ECS `update-service` call, or another
target — see the inline comments in that job.

Required repo secrets for the full pipeline:
| Secret | Used by | Required? |
|---|---|---|
| `GITHUB_TOKEN` | GHCR push (auto-provided by GitHub Actions) | Yes (automatic) |
| `SAFETY_API_KEY` | Safety CLI cloud vulnerability DB | Optional — scan is skipped with a warning if absent |
| `DEPLOY_URL` | Post-deploy smoke tests | Optional — smoke test is skipped with a warning if absent |

## Configuration

All runtime configuration lives in `configs/config.yaml` (paths,
hyperparameter search space, drift thresholds, API host/port, etc.) — no
values are hardcoded in application code. Override the config file location
with the `HEALTHCARE_MLOPS_CONFIG` environment variable if you need
environment-specific configs (e.g. `configs/config.prod.yaml`).

## Health Checks

Every deployment path (Docker `HEALTHCHECK`, Compose `healthcheck`, ECS
task-definition `healthCheck`, ALB target-group health check) points at
`GET /health`, which returns `503`-equivalent `"status": "degraded"` when
model artifacts aren't loaded — use this to gate traffic until training has
completed at least once.

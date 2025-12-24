# Verda Agent 🌱

AI-powered plant diagnosis and care recommendation service.

## Components

- Image classification (MobileNetV2)
- Weather-aware reasoning
- Care recommendation engine

## Folder Structure

- `app/` – agent logic
- `models/` – model checkpoints (mounted at runtime)
- `data/` – care datasets (Excel)

## Running Locally

```bash
export MODEL_PATH=models/fine_tuned_mobilenet.pth
export CARE_DATA_PATH=data/care_details.xlsx

python -m app.verda_agent

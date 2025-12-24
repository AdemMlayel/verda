# Verda Docker 🐳

Infrastructure repository for running Verda services locally or in staging.

## Services

- **verda-agent**  
  Python ML agent for plant diagnosis and care recommendations.

## Prerequisites

- Docker
- Docker Compose
- `verda-agent` repo cloned locally
- Model checkpoint placed in `verda-agent/models`

## Setup

```bash
cp .env.example .env

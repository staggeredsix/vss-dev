# AI Workbench Deployment

This folder contains a simplified deployment scheme for running the Video Search and Summarization (VSS) demo in NVIDIA AI Workbench. Each service runs in its own container and all containers share a common Docker network.

The frontend is built as a standalone container so it can be launched directly from the AI Workbench UI. It is not included in the compose file.

## Services

- **telemetry** – collects basic runtime metrics.
- **ollama** – serves models used by the pipeline.
- **pipeline** – runs the VSS processing pipeline.

## Usage

1. Ensure Docker is running and the `aiw-net` network exists:
   ```bash
   docker network create aiw-net || true
   ```
2. Start the backend services:
   ```bash
   docker compose -f deploy/ai-workbench/docker-compose.yaml up -d
   ```
3. Build the frontend container from the project root:
   ```bash
   docker build -f Dockerfile.frontend -t vss-frontend .
   ```
   Launch the container from the AI Workbench UI to access the Gradio interface.

All services request access to GPU resources using the `NVIDIA_VISIBLE_DEVICES` environment variable and compose `deploy.resources` settings.

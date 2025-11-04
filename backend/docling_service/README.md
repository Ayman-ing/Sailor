# Docling Processing Microservice

A standalone service to handle heavy PDF processing using Docling.

## Setup on Powerful Machine

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Run the service
uv run python main.py
```

The service will be available at `http://<machine-ip>:8001`

## API Endpoints

- `POST /process` - Upload PDF and get chunks
- `GET /health` - Health check

## Example Usage

```bash
curl -X POST "http://<machine-ip>:8001/process" \
  -F "file=@document.pdf"
```

# VLLM Translation Integration

This document describes the VLLM sidecar container integration for translation support in ComfyStream.

## Architecture

The translation feature uses a sidecar container pattern:

```
┌─────────────────┐    HTTP API    ┌──────────────────┐
│   ComfyStream   │ ──────────────► │   VLLM Container │
│   Container     │                 │  (Translation)   │
│                 │ ◄────────────── │                  │
└─────────────────┘   Responses    └──────────────────┘
```

## Components

### 1. VLLM Container (`docker/Dockerfile.vllm`)
- Based on `vllm/vllm-openai:latest`
- Runs translation models via OpenAI-compatible API
- Configurable model selection
- Health check endpoints

### 2. Translation Client (`src/comfystream/translation/client.py`)
- Async HTTP client for VLLM communication
- Error handling and fallback support
- Batch translation capabilities
- Health monitoring

### 3. API Endpoints (in `server/app.py`)
- `POST /translate` - Single text translation
- `POST /translate/batch` - Batch translation
- `GET /translate/health` - Service health check

### 4. Docker Compose (`docker/docker-compose.yml`)
- Orchestrates both containers
- Manages dependencies and networking
- Shared volumes for model storage

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_ENDPOINT` | `http://localhost:8000` | VLLM service endpoint |
| `VLLM_MODEL` | `microsoft/DialoGPT-medium` | Translation model |
| `VLLM_HOST` | `0.0.0.0` | VLLM service host |
| `VLLM_PORT` | `8000` | VLLM service port |

### Model Selection

You can use different models by setting the `VLLM_MODEL` environment variable:

```bash
# For better translation quality (requires more GPU memory)
export VLLM_MODEL="Helsinki-NLP/opus-mt-en-de"

# For multilingual support
export VLLM_MODEL="facebook/mbart-large-50-many-to-many-mmt"
```

## Usage

### Starting the Services

```bash
# Start both containers
docker-compose -f docker/docker-compose.yml up -d

# Check service status
curl http://localhost:8889/translate/health
```

### API Usage

```bash
# Single translation
curl -X POST http://localhost:8889/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "source_lang": "en",
    "target_lang": "es"
  }'

# Batch translation
curl -X POST http://localhost:8889/translate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello", "Goodbye"],
    "source_lang": "en", 
    "target_lang": "fr"
  }'
```

### Response Format

```json
{
  "translated_text": "Hola mundo",
  "source_language": "en",
  "target_language": "es",
  "model_info": "translation-model",
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 2,
    "total_tokens": 6
  }
}
```

## Integration with ComfyUI Workflows

The translation API can be integrated into ComfyUI workflows through custom nodes that make HTTP requests to the translation endpoints. This allows real-time translation of text elements in video streams.

## Monitoring and Debugging

### Health Checks
- ComfyStream health: `GET http://localhost:8889/health`
- Translation health: `GET http://localhost:8889/translate/health`
- VLLM direct health: `GET http://localhost:8000/health`

### Logs
```bash
# View ComfyStream logs
docker-compose logs comfystream

# View VLLM logs
docker-compose logs vllm

# Follow logs in real-time
docker-compose logs -f
```

## Troubleshooting

### Common Issues

1. **Translation service unavailable**
   - Check if VLLM container is running: `docker-compose ps`
   - Verify network connectivity between containers
   - Check VLLM health endpoint

2. **Out of memory errors**
   - Use smaller models
   - Increase GPU memory allocation
   - Reduce batch sizes

3. **Slow translation**
   - Model loading time on first request
   - Use model caching
   - Consider using faster models

### Performance Optimization

- **Model Caching**: Mount persistent volumes for model storage
- **Batch Processing**: Use batch endpoints for multiple translations
- **Model Selection**: Choose appropriate model size for your GPU
- **Connection Pooling**: The client reuses HTTP connections automatically

## Security Considerations

- The VLLM container runs as non-root user
- No external network access required for basic operation
- Model downloads happen during container build
- API endpoints have input validation

## Extending the Integration

The translation client can be extended to support:
- Custom prompt templates
- Different translation models per language pair
- Translation caching
- Custom preprocessing/postprocessing
- Integration with other language services
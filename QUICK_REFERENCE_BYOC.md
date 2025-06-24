# ComfyStream BYOC - Quick Reference Guide

## 🚀 Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the BYOC server
python example_byoc_server.py --workspace /path/to/comfyui/workspace

# 3. Test it works
curl http://localhost:5000/health

# 4. Test BYOC functionality
curl -X POST http://localhost:5000/process/request/text-reversal \
  -H "Content-Type: application/json" \
  -H "Livepeer: eyJyZXF1ZXN0IjogIntcInJ1blwiOiBcImVjaG9cIn0iLCAiY2FwYWJpbGl0eSI6ICJ0ZXh0LXJldmVyc2FsIiwgInRpbWVvdXRfc2Vjb25kcyI6IDMwfQ==" \
  -d '{"text":"Hello World!"}'
```

## 📁 Key Files

| File | Purpose | Priority |
|------|---------|----------|
| `src/comfystream/server/byoc_server.py` | Main BYOC server implementation | 🔴 Critical |
| `example_byoc_server.py` | Standalone server runner | 🔴 Critical |
| `src/comfystream/server/trickle/` | Trickle streaming integration | 🟠 Important |
| `test_byoc_example.py` | Test suite | 🟡 Testing |
| `docker/livepeer.dockerfile` | Docker container | 🟡 Deployment |
| `README_BYOC.md` | User documentation | 🟢 Reference |
| `HANDOVER_BYOC.md` | Complete handover doc | 🟢 Reference |

## 🔧 Core Components

### BYOC Server (`byoc_server.py`)
- **Class**: `ComfyStreamBYOCServer` - Main server
- **Method**: `process_capability_request()` - Handles BYOC requests
- **Method**: `start_stream()` / `stop_stream()` - Stream management

### Capabilities Supported
1. `text-reversal` - Text reversal (BYOC example)
2. `comfystream-video` - Video processing with streaming
3. `comfystream-image` - Single-shot image processing

### API Endpoints
- `POST /process/request/{capability}` - BYOC processing
- `POST /stream/start` - Start stream
- `DELETE /stream/{manifest_id}` - Stop stream  
- `GET /streams` - List streams
- `GET /health` - Health check

## 🐳 Docker

```bash
# Build
docker build -f docker/livepeer.dockerfile -t comfystream-byoc .

# Run
docker run -p 5000:5000 -v /workspace:/workspace comfystream-byoc
```

## 🧪 Testing

```bash
# Automated tests
python test_byoc_example.py

# Manual health check
curl http://localhost:5000/health

# Stream test
curl -X POST http://localhost:5000/stream/start \
  -H "Content-Type: application/json" \
  -d '{"prompts": [{"prompt": "test"}], "stream_url": "http://test"}'
```

## 🚨 Common Issues

| Problem | Solution |
|---------|----------|
| Port 5000 in use | Use `--port 5001` |
| Workspace not found | Check path with `--workspace` |
| Import errors | Run `pip install -e .` |
| Docker build fails | Check dependencies in Dockerfile |

## 📊 Architecture

```
Livepeer Orchestrator → BYOC Server → ComfyUI Pipeline → Trickle Stream
                        (Port 5000)   (GPU Required)    (Video Output)
```

## 🔍 Monitoring

- **Health**: `curl http://localhost:5000/health`
- **Logs**: Check console output or use `--log-level DEBUG`
- **Streams**: `curl http://localhost:5000/streams`

## 📚 Next Steps

1. **Production**: Review `HANDOVER_BYOC.md` security section
2. **Scaling**: Consider load balancing for multiple instances  
3. **Monitoring**: Set up Prometheus metrics
4. **Integration**: Connect to actual Livepeer orchestrator

## 🆘 Need Help?

1. Check `README_BYOC.md` for detailed usage
2. Review `HANDOVER_BYOC.md` for complete documentation
3. Run `python test_byoc_example.py` to verify setup
4. Enable debug logging: `--log-level DEBUG`

---
*This is a quick reference. For complete information, see HANDOVER_BYOC.md* 
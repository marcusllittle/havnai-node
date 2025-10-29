# havnai-node

Stage 5A introduces Creator Mode, public job submissions, and tokenized rewards for heavy workloads.

## Quick Start

```bash
pip install -r requirements.txt
python server/app.py
```

Run a standard worker node:

```bash
python node/client.py
```

Run a creator node capable of image generation workloads:

```bash
CREATOR_MODE=true python node/client.py
```

Submit a public job (example image generation request):

```bash
curl -X POST http://localhost:5001/submit-job \
  -H "Content-Type: application/json" \
  -d '{
        "wallet": "0x7110347e2bcd02F5F3485Dc6bEc5e0b5f9Eb9262",
        "model": "sdxl",
        "task_type": "image_gen",
        "prompt": "cyberpunk skyline"
      }'
```

Visit `http://localhost:5001/dashboard` to monitor creator nodes, queued jobs, and $HAI rewards.

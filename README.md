# HavnAI Node Runtime

Stage 7 introduces a public onboarding flow so external GPU operators can join the HavnAI grid with a single command. The coordinator manages job queues, wallet-linked rewards, creator-model discovery, and a live dashboard / leaderboard.

## Join the Grid (One-Time Setup)

1. **Prepare a wallet** — use any EVM-compatible address that can receive $HAI on Sepolia.
2. **Visit the coordinator join page** — `http://<server-ip>:5001/join` for prerequisites and the one-liner.
3. **Run the installer** (replace the host and optional join token):
   ```bash
   curl -fsSL http://<server-ip>:5001/static/install-node.sh | bash -s -- --server http://<server-ip>:5001 [--token YOUR_JOIN_TOKEN]
   ```
4. **Edit `~/.havnai/.env`** to add your wallet (and enable creator mode if desired).
5. **Start the service**:
   - Linux/systemd: `systemctl --user daemon-reload && systemctl --user enable --now havnai-node`
   - macOS/launchd: `launchctl load -w ~/Library/LaunchAgents/com.havnai.node.plist`
6. **Monitor activity**:
   - Dashboard: `http://<server-ip>:5001/dashboard`
   - Leaderboard: `http://<server-ip>:5001/network/leaderboard`
   - Logs: `journalctl --user -u havnai-node -f` (Linux) or `log stream --predicate 'process == "havnai-node"'` (macOS)

## Environment Variables (`~/.havnai/.env`)

| Key | Description |
| --- | --- |
| `SERVER_URL` | Coordinator base URL (e.g. `http://192.168.1.50:5001`) |
| `WALLET` | EVM wallet used for rewards |
| `CREATOR_MODE` | `true` to enable heavy image generation tasks; otherwise `false` |
| `NODE_NAME` | Friendly name shown on the dashboard/leaderboard |
| `JOIN_TOKEN` | Optional join token required by the coordinator |

Creator Mode nodes can drop checkpoints into `~/.havnai/models/creator/`; the client auto-registers hashes/metadata with the server on startup.

## Running from Source (manual)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server/app.py

# In another shell
python node/client.py  # set CREATOR_MODE=true for heavy workloads
```

## Docker (optional)

A lightweight client image is provided:

```bash
docker-compose -f docker-compose.sample.yml up --build -d
```

Mount `~/.havnai` into the container and populate `.env` exactly as the installer does.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `401 unauthorized` on `/register` | Ensure `JOIN_TOKEN` in `~/.havnai/.env` matches the server's `SERVER_JOIN_TOKEN` |
| Node offline in dashboard | Confirm `havnai-node` service is running and `SERVER_URL` is reachable |
| Rewards not credited | Wallet missing or malformed — edit `.env`, rerun client, and watch `/network/leaderboard` |
| Creator models ignored | Files must reside in `~/.havnai/models/creator` with supported extensions (`.onnx`, `.safetensors`, `.ckpt`) |
| Installer missing Python | Supply `--server` and rerun; script attempts apt/dnf/brew installs when possible |

## Project Layout

```
server/   Flask coordinator (APIs, dashboard, installer script)
client/   Python worker agent (heartbeats, model registration, job execution)
node/     Legacy wrapper + launch artifacts
server/static/installer-node.sh  One-command installer hosted by the coordinator
Dockerfile.node & docker-compose.sample.yml  Optional container path
```

## Stage 8 Preview

- Hardened TLS ingress (Cloudflare Tunnel / HTTPS termination)
- Public node directory + automatic reputation scoring
- Token payout batching & Sepolia settlement flows

Happy grid-building! Open an issue with the exact command/output if onboarding fails.

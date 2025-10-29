from __future__ import annotations

import json
import logging
from logging.handlers import RotatingFileHandler
import os
import random
import re
import sqlite3
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from onnx import TensorProto, helper

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS_DIR = STATIC_DIR / "models"
CREATOR_MODEL_DIR = Path(__file__).resolve().parent / "models" / "creator"
LOGS_DIR = Path(__file__).resolve().parent / "logs"
REGISTRY_FILE = BASE_DIR / "nodes.json"
DB_PATH = BASE_DIR / "db" / "ledger.db"
CLIENT_PATH = BASE_DIR / "client" / "client.py"
CLIENT_REQUIREMENTS = BASE_DIR / "client" / "requirements-node.txt"
VERSION_FILE = BASE_DIR / "VERSION"

SUPPORTED_MODEL_EXTS = {".onnx", ".safetensors", ".ckpt"}

# Reward weights bootstrap – enriched at runtime via registration
MODEL_WEIGHTS: Dict[str, float] = {
    "mlp-classifier": 1.0,
    "conv-demo": 1.5,
    "sdxl": 10.0,
    "nsfw-sdxl": 12.0,
}

MODEL_STATS: Dict[str, Dict[str, float]] = {}
REGISTERED_MODELS: Dict[str, Dict[str, Any]] = {}
EVENT_LOGS: deque = deque(maxlen=200)
NODES: Dict[str, Dict[str, Any]] = {}
TASKS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()
DB_CONN: Optional[sqlite3.Connection] = None
RATE_LIMIT_BUCKETS: Dict[str, deque] = defaultdict(deque)

ONLINE_THRESHOLD = 120  # seconds before a node is considered offline
WALLET_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")

SERVER_JOIN_TOKEN = os.getenv("SERVER_JOIN_TOKEN", "").strip()
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "").strip()

# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------


def resolve_version() -> str:
    if VERSION_FILE.exists():
        return VERSION_FILE.read_text().strip()
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR)
            .decode()
            .strip()
        )
    except Exception:
        return "dev"


APP_VERSION = resolve_version()

# ---------------------------------------------------------------------------
# Flask application & CORS
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
if CORS_ORIGINS and CORS_ORIGINS != "*":
    origins = [origin.strip() for origin in CORS_ORIGINS.split(",") if origin.strip()]
    CORS(app, resources={r"/*": {"origins": origins}})
else:
    CORS(app)

app.config["APP_VERSION"] = APP_VERSION

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "time": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload)


def setup_logging() -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("havnai")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(LOGS_DIR / "havnai.log", maxBytes=5 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(JSONFormatter())
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


LOGGER = setup_logging()


def log_event(message: str, level: str = "info", **extra: Any) -> None:
    LOGGER.log(getattr(logging, level.upper(), logging.INFO), message, extra=extra if extra else None)
    EVENT_LOGS.append({"timestamp": datetime.utcnow().isoformat() + "Z", "level": level, "message": message})


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def unix_now() -> float:
    return time.time()


def parse_timestamp(value: Any) -> float:
    if value is None:
        return unix_now()
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "")).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return unix_now()
    return unix_now()


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def resolve_weight(model_name: str, default: float = 1.0) -> float:
    return float(MODEL_WEIGHTS.get(model_name, default))


def check_join_token() -> bool:
    if not SERVER_JOIN_TOKEN:
        return True
    header_token = request.headers.get("X-Join-Token", "")
    query_token = request.args.get("token", "")
    provided = header_token or query_token
    if provided != SERVER_JOIN_TOKEN:
        return False
    return True


def rate_limit(key: str, limit: int, per_seconds: int = 60) -> bool:
    now = unix_now()
    window_start = now - per_seconds
    bucket = RATE_LIMIT_BUCKETS.setdefault(key, deque())
    while bucket and bucket[0] < window_start:
        bucket.popleft()
    if len(bucket) >= limit:
        return False
    bucket.append(now)
    return True


# ---------------------------------------------------------------------------
# Directory bootstrap
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    for directory in (STATIC_DIR, MODELS_DIR, CREATOR_MODEL_DIR, LOGS_DIR, BASE_DIR / "db"):
        directory.mkdir(parents=True, exist_ok=True)


ensure_directories()

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def get_db() -> sqlite3.Connection:
    global DB_CONN
    if DB_CONN is None:
        DB_CONN = sqlite3.connect(DB_PATH, check_same_thread=False)
        DB_CONN.row_factory = sqlite3.Row
    return DB_CONN


def init_db() -> None:
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            model TEXT NOT NULL,
            data TEXT,
            task_type TEXT NOT NULL,
            weight REAL NOT NULL,
            status TEXT NOT NULL,
            node_id TEXT,
            timestamp REAL NOT NULL,
            assigned_at REAL,
            completed_at REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            wallet TEXT NOT NULL,
            task_id TEXT NOT NULL UNIQUE,
            reward_hai REAL NOT NULL,
            timestamp REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS node_models (
            node_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            filename TEXT,
            hash TEXT,
            size INTEGER,
            tags TEXT,
            weight REAL,
            task_type TEXT,
            updated_at REAL,
            PRIMARY KEY (node_id, model_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS node_wallets (
            node_id TEXT PRIMARY KEY,
            wallet TEXT NOT NULL,
            node_name TEXT,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("UPDATE jobs SET status='queued', node_id=NULL WHERE status='running'")
    conn.commit()


init_db()

# ---------------------------------------------------------------------------
# Model catalog bootstrap
# ---------------------------------------------------------------------------


def build_mlp_model(path: Path, input_dim: int, hidden_dim: int, output_dim: int) -> None:
    if path.exists():
        return
    weights1 = np.random.randn(input_dim, hidden_dim).astype(np.float32) * 0.1
    bias1 = np.random.randn(hidden_dim).astype(np.float32) * 0.1
    weights2 = np.random.randn(hidden_dim, output_dim).astype(np.float32) * 0.1
    bias2 = np.random.randn(output_dim).astype(np.float32) * 0.1

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, input_dim])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, output_dim])

    W1 = helper.make_tensor("W1", TensorProto.FLOAT, weights1.shape, weights1.flatten())
    B1 = helper.make_tensor("B1", TensorProto.FLOAT, bias1.shape, bias1)
    W2 = helper.make_tensor("W2", TensorProto.FLOAT, weights2.shape, weights2.flatten())
    B2 = helper.make_tensor("B2", TensorProto.FLOAT, bias2.shape, bias2)

    node1 = helper.make_node("Gemm", ["input", "W1", "B1"], ["h1"], alpha=1.0, beta=1.0)
    relu = helper.make_node("Relu", ["h1"], ["relu1"])
    node2 = helper.make_node("Gemm", ["relu1", "W2", "B2"], ["output"], alpha=1.0, beta=1.0)

    graph = helper.make_graph([node1, relu, node2], "mlp_graph", [input_tensor], [output_tensor], [W1, B1, W2, B2])
    model = helper.make_model(graph, producer_name="havnai")
    onnx.save(model, path)


def build_conv_model(path: Path) -> None:
    if path.exists():
        return
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 16, 16])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 14, 14])

    weights = np.random.randn(8, 1, 3, 3).astype(np.float32) * 0.05
    bias = np.random.randn(8).astype(np.float32) * 0.05

    W = helper.make_tensor("conv_W", TensorProto.FLOAT, weights.shape, weights.flatten())
    B = helper.make_tensor("conv_B", TensorProto.FLOAT, bias.shape, bias)
    conv = helper.make_node("Conv", ["input", "conv_W", "conv_B"], ["conv_out"], pads=[0, 0, 0, 0])
    relu = helper.make_node("Relu", ["conv_out"], ["relu_out"])
    graph = helper.make_graph([conv, relu], "conv_graph", [input_tensor], [output_tensor], [W, B])
    model = helper.make_model(graph, producer_name="havnai")
    onnx.save(model, path)


MODEL_CATALOG: List[Dict[str, Any]] = [
    {
        "name": "mlp-classifier",
        "filename": "mlp-classifier.onnx",
        "input_shape": [1, 64],
        "reward_weight": MODEL_WEIGHTS.get("mlp-classifier", 1.0),
        "task_type": "ai",
        "builder": lambda path: build_mlp_model(path, 64, 32, 10),
        "source": "builtin",
    },
    {
        "name": "conv-demo",
        "filename": "conv-demo.onnx",
        "input_shape": [1, 1, 16, 16],
        "reward_weight": MODEL_WEIGHTS.get("conv-demo", 1.5),
        "task_type": "ai",
        "builder": build_conv_model,
        "source": "builtin",
    },
]

STATIC_CREATOR_MODELS: Dict[str, Dict[str, Any]] = {}


def ensure_model_catalog() -> None:
    for cfg in MODEL_CATALOG:
        path = MODELS_DIR / cfg["filename"]
        builder = cfg.get("builder")
        if callable(builder):
            builder(path)
        cfg["path"] = path
        cfg["url"] = f"/static/models/{path.name}"
        MODEL_STATS.setdefault(cfg["name"], {"count": 0.0, "total_time": 0.0})

    global STATIC_CREATOR_MODELS
    STATIC_CREATOR_MODELS = {}
    if CREATOR_MODEL_DIR.exists():
        for path in CREATOR_MODEL_DIR.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_MODEL_EXTS:
                continue
            name = path.stem.lower().replace(" ", "-")
            weight = MODEL_WEIGHTS.get(name)
            if weight is None:
                weight = 10.0 if path.suffix.lower() in {".safetensors", ".ckpt"} else 2.0
                MODEL_WEIGHTS[name] = weight
            task_type = "image_gen" if path.suffix.lower() in {".safetensors", ".ckpt"} else "ai"
            STATIC_CREATOR_MODELS[name] = {
                "name": name,
                "filename": path.name,
                "path": path,
                "url": f"/models/download/{path.name}",
                "reward_weight": weight,
                "task_type": task_type,
                "size": path.stat().st_size,
                "source": "server",
                "tags": [path.suffix.lstrip('.')],
            }
            MODEL_STATS.setdefault(name, {"count": 0.0, "total_time": 0.0})


ensure_model_catalog()


def refresh_registered_models() -> None:
    global REGISTERED_MODELS
    conn = get_db()
    rows = conn.execute(
        "SELECT node_id, model_name, filename, hash, size, tags, weight, task_type FROM node_models"
    ).fetchall()
    catalog: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row["model_name"].lower()
        entry = catalog.setdefault(
            name,
            {
                "hash": row["hash"],
                "size": row["size"],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "weight": float(row["weight"] or MODEL_WEIGHTS.get(name, 5.0)),
                "task_type": row["task_type"] or ("image_gen" if name.startswith("sd") else "ai"),
                "nodes": set(),
                "source": "registered",
            },
        )
        entry["name"] = name
        entry["nodes"].add(row["node_id"])
        if row["hash"]:
            entry["hash"] = row["hash"]
        if row["size"]:
            entry["size"] = row["size"]
        if row["weight"]:
            entry["weight"] = float(row["weight"])
        if row["task_type"]:
            entry["task_type"] = row["task_type"]
    for entry in catalog.values():
        entry["nodes"] = sorted(entry["nodes"])
        MODEL_WEIGHTS[entry["name"]] = entry["weight"]
    REGISTERED_MODELS = catalog


refresh_registered_models()

# ---------------------------------------------------------------------------
# Node persistence (nodes.json + wallet bindings)
# ---------------------------------------------------------------------------


def load_nodes() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return {}
    with REGISTRY_FILE.open() as f:
        data = json.load(f)
    now = unix_now()
    for node in data.values():
        node.setdefault("os", "unknown")
        node.setdefault("gpu", {})
        node.setdefault("rewards", 0.0)
        node.setdefault("utilization", 0.0)
        node.setdefault("avg_utilization", node.get("utilization", 0.0))
        node.setdefault("tasks_completed", 0)
        node.setdefault("current_task", None)
        node.setdefault("last_result", {})
        node.setdefault("reward_history", [])
        node.setdefault("last_reward", 0.0)
        node.setdefault("start_time", now)
        node.setdefault("last_seen", iso_now())
        node.setdefault("role", node.get("role", "worker"))
        node.setdefault("node_name", node.get("node_name") or node.get("node_id"))
        node.setdefault("wallet", node.get("wallet"))
        node["last_seen_unix"] = parse_timestamp(node.get("last_seen"))
    return data


def save_nodes() -> None:
    payload = {}
    for node_id, info in NODES.items():
        serial = dict(info)
        serial.pop("last_seen_unix", None)
        payload[node_id] = serial
    with REGISTRY_FILE.open("w") as f:
        json.dump(payload, f, indent=2)


NODES = load_nodes()


def load_node_wallets() -> None:
    conn = get_db()
    rows = conn.execute("SELECT node_id, wallet, node_name FROM node_wallets").fetchall()
    for row in rows:
        node = NODES.setdefault(row["node_id"], {
            "os": "unknown",
            "gpu": {},
            "rewards": 0.0,
            "utilization": 0.0,
            "avg_utilization": 0.0,
            "tasks_completed": 0,
            "current_task": None,
            "last_result": {},
            "reward_history": [],
            "last_reward": 0.0,
            "start_time": unix_now(),
            "role": "worker",
            "last_seen": iso_now(),
        })
        node["wallet"] = row["wallet"]
        if row["node_name"]:
            node["node_name"] = row["node_name"]


load_node_wallets()
log_event(f"Telemetry online with {len(NODES)} cached node(s).", version=APP_VERSION)

# ---------------------------------------------------------------------------
# Job + reward helpers
# ---------------------------------------------------------------------------


def enqueue_job(wallet: str, model: str, task_type: str, data: str, weight: float) -> str:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    conn = get_db()
    conn.execute(
        """
        INSERT INTO jobs (id, wallet, model, data, task_type, weight, status, node_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)
        """,
        (job_id, wallet, model, data, task_type, float(weight), unix_now()),
    )
    conn.commit()
    return job_id


def fetch_next_job_for_node(node_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY timestamp ASC").fetchall()
    node = NODES.get(node_id, {})
    role = node.get("role", "worker")
    for row in rows:
        model_name = row["model"].lower()
        cfg = get_model_config(model_name)
        if not cfg:
            continue
        task_type = row["task_type"] or cfg.get("task_type", "ai")
        if task_type == "image_gen" and role != "creator":
            continue
        if cfg.get("source") == "registered":
            allowed_nodes = set(REGISTERED_MODELS.get(model_name, {}).get("nodes", []))
            if node_id not in allowed_nodes:
                continue
        return dict(row)
    return None


def assign_job_to_node(job_id: str, node_id: str) -> None:
    conn = get_db()
    conn.execute("UPDATE jobs SET status='running', node_id=?, assigned_at=? WHERE id=?", (node_id, unix_now(), job_id))
    conn.commit()


def complete_job(job_id: str, status: str) -> None:
    conn = get_db()
    conn.execute("UPDATE jobs SET status=?, completed_at=? WHERE id=?", (status, unix_now(), job_id))
    conn.commit()


def record_reward(wallet: Optional[str], task_id: str, reward: float) -> None:
    if not wallet:
        return
    conn = get_db()
    conn.execute(
        """
        INSERT INTO rewards (wallet, task_id, reward_hai, timestamp)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET
            wallet=excluded.wallet,
            reward_hai=excluded.reward_hai,
            timestamp=excluded.timestamp
        """,
        (wallet, task_id, reward, unix_now()),
    )
    conn.commit()


def get_job_summary(limit: int = 12) -> Dict[str, Any]:
    conn = get_db()
    queued = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
    total_distributed = conn.execute("SELECT COALESCE(SUM(reward_hai),0) FROM rewards").fetchone()[0]
    rows = conn.execute(
        """
        SELECT jobs.id, jobs.wallet, jobs.model, jobs.task_type, jobs.status, jobs.weight,
               jobs.completed_at, rewards.reward_hai
        FROM jobs
        LEFT JOIN rewards ON rewards.task_id = jobs.id
        ORDER BY jobs.timestamp DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    feed = []
    for row in rows:
        completed_at = row["completed_at"]
        completed_iso = (
            datetime.utcfromtimestamp(completed_at).isoformat() + "Z"
            if completed_at
            else None
        )
        feed.append(
            {
                "job_id": row["id"],
                "wallet": row["wallet"],
                "model": row["model"],
                "task_type": row["task_type"],
                "status": row["status"],
                "weight": float(row["weight"] or MODEL_WEIGHTS.get(row["model"], 1.0)),
                "reward": round(row["reward_hai"] or 0.0, 6),
                "completed_at": completed_iso,
            }
        )
    return {
        "queued_jobs": queued,
        "active_jobs": active,
        "total_distributed": round(total_distributed or 0.0, 6),
        "feed": feed,
    }


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Model registry helpers
# ---------------------------------------------------------------------------


def get_model_config(model_name: str) -> Optional[Dict[str, Any]]:
    model_name = (model_name or "").lower()
    for cfg in MODEL_CATALOG:
        if cfg["name"] == model_name:
            return cfg
    if model_name in STATIC_CREATOR_MODELS:
        return STATIC_CREATOR_MODELS[model_name]
    if model_name in REGISTERED_MODELS:
        entry = REGISTERED_MODELS[model_name]
        return {
            "name": model_name,
            "reward_weight": entry.get("weight", resolve_weight(model_name, 5.0)),
            "task_type": entry.get("task_type", "ai"),
            "url": entry.get("url", ""),
            "source": entry.get("source", "registered"),
            "nodes": entry.get("nodes", []),
        }
    return None


def build_models_catalog() -> List[Dict[str, Any]]:
    catalog: List[Dict[str, Any]] = []
    for cfg in MODEL_CATALOG:
        path = MODELS_DIR / cfg["filename"]
        catalog.append(
            {
                "model": cfg["name"],
                "weight": resolve_weight(cfg["name"], cfg.get("reward_weight", 1.0)),
                "source": cfg.get("source", "builtin"),
                "nodes": "server",
                "tags": ["onnx"],
                "size": path.stat().st_size if path.exists() else 0,
            }
        )
    for name, meta in STATIC_CREATOR_MODELS.items():
        catalog.append(
            {
                "model": name,
                "weight": resolve_weight(name, meta.get("reward_weight", 5.0)),
                "source": "server",
                "nodes": "server",
                "tags": meta.get("tags", []),
                "size": meta.get("size", 0),
            }
        )
    for name, meta in REGISTERED_MODELS.items():
        catalog.append(
            {
                "model": name,
                "weight": resolve_weight(name, meta.get("weight", 5.0)),
                "source": meta.get("source", "registered"),
                "nodes": meta.get("nodes", []),
                "tags": meta.get("tags", []),
                "size": meta.get("size", 0),
            }
        )
    return catalog


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------


def issue_internal_task(node_id: str) -> Dict[str, Any]:
    cfg = random.choice(MODEL_CATALOG)
    task_id = f"ai-{uuid.uuid4().hex[:10]}"
    return {
        "task_id": task_id,
        "task_type": cfg.get("task_type", "ai"),
        "model_name": cfg["name"],
        "model_url": cfg["url"],
        "input_shape": cfg["input_shape"],
        "reward_weight": cfg.get("reward_weight", 1.0),
        "assigned_to": node_id,
        "status": "pending",
        "job_id": None,
        "wallet": None,
    }


def pending_tasks_for_node(node_id: str) -> List[Dict[str, Any]]:
    return [task for task in TASKS.values() if task["assigned_to"] == node_id and task["status"] in {"pending", "assigned"}]


# ---------------------------------------------------------------------------
# Leaderboard helpers
# ---------------------------------------------------------------------------


def leaderboard_rows(limit: int = 25) -> List[Dict[str, Any]]:
    conn = get_db()
    all_time_rows = conn.execute(
        "SELECT wallet, SUM(reward_hai) AS total, COUNT(*) AS jobs FROM rewards GROUP BY wallet"
    ).fetchall()
    totals = {row["wallet"]: float(row["total"] or 0) for row in all_time_rows}
    job_counts = {row["wallet"]: row["jobs"] for row in all_time_rows}

    cutoff = unix_now() - 86400
    last24_rows = conn.execute(
        "SELECT wallet, SUM(reward_hai) AS total FROM rewards WHERE timestamp >= ? GROUP BY wallet",
        (cutoff,),
    ).fetchall()
    last24 = {row["wallet"]: float(row["total"] or 0) for row in last24_rows}

    wallet_nodes: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rows = conn.execute("SELECT wallet, node_id, node_name FROM node_wallets").fetchall()
    for row in rows:
        node_id = row["node_id"]
        node = NODES.get(node_id, {})
        wallet_nodes[row["wallet"]].append(
            {
                "node_id": node_id,
                "node_name": row["node_name"] or node.get("node_name") or node_id,
                "role": node.get("role", "worker"),
            }
        )

    leaderboard = []
    for wallet, total in totals.items():
        nodes = wallet_nodes.get(wallet, [])
        creator = any(node.get("role") == "creator" for node in nodes)
        leaderboard.append(
            {
                "wallet": wallet,
                "wallet_short": wallet[:6] + "…" + wallet[-4:] if wallet else "—",
                "all_time": round(total, 6),
                "jobs": job_counts.get(wallet, 0),
                "rewards_24h": round(last24.get(wallet, 0.0), 6),
                "nodes": nodes,
                "creator": creator,
            }
        )
    leaderboard.sort(key=lambda row: row["all_time"], reverse=True)
    for idx, row in enumerate(leaderboard, start=1):
        row["rank"] = idx
        if idx > limit:
            break
    return leaderboard[:limit]


# ---------------------------------------------------------------------------
# Routes – public pages
# ---------------------------------------------------------------------------


@app.route("/health")
def health() -> Any:
    job_summary = get_job_summary()
    return jsonify(
        {
            "status": "ok",
            "nodes": len(NODES),
            "queue_depth": job_summary["queued_jobs"],
            "version": APP_VERSION,
        }
    )


@app.route("/join")
def join_page() -> Any:
    host = request.host_url.rstrip("/")
    token_hint = f" --token {SERVER_JOIN_TOKEN}" if SERVER_JOIN_TOKEN else ""
    install_cmd = f"curl -fsSL {host}/static/install-node.sh | bash -s -- --server {host}{token_hint}"
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>Join HavnAI Grid</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#03121b; color:#e6f6ff; padding:2.5rem; }}
        code {{ background:#07202d; padding:0.25rem 0.4rem; border-radius:4px; }}
        pre {{ background:#07202d; padding:1rem; border-radius:8px; overflow:auto; }}
        a {{ color:#5fe4ff; }}
        h1 {{ color:#5fe4ff; }}
        ul {{ line-height:1.6; }}
      </style>
    </head>
    <body>
      <h1>Join the HavnAI GPU Grid</h1>
      <p>Run the installer on your GPU machine:</p>
      <pre><code>{install_cmd}</code></pre>
      <h2>Prerequisites</h2>
      <ul>
        <li>64-bit Linux (Ubuntu/Debian/RHEL) or macOS (12+)</li>
        <li>Python 3.10 or newer, curl, and a GPU driver/runtime</li>
        <li>$HAI wallet address (EVM compatible)</li>
      </ul>
      <h2>What happens next?</h2>
      <ol>
        <li>Installer prepares <code>~/.havnai</code>, Python venv, and the node binary</li>
        <li>Configure your wallet inside <code>~/.havnai/.env</code></li>
        <li>Enable the service (systemd or launchd)</li>
        <li>Monitor progress via <a href=\"/dashboard\">dashboard</a> and <a href=\"/network/leaderboard\">leaderboard</a></li>
      </ol>
      <p>Need the join token or help? Contact the grid operator.</p>
    </body>
    </html>
    """
    return html


@app.route("/network/leaderboard")
def leaderboard() -> Any:
    data = leaderboard_rows()
    if request.args.get("format") == "json":
        return jsonify({"leaderboard": data})

    rows_html = "".join(
        f"<tr><td>{row['rank']}</td><td>{', '.join(node['node_name'] for node in row['nodes']) or '—'}</td>"
        f"<td>{row['wallet_short']}</td><td>{row['jobs']}</td><td>{row['rewards_24h']:.6f}</td>"
        f"<td>{row['all_time']:.6f}</td><td>{'✅' if row['creator'] else ''}</td></tr>"
        for row in data
    )
    html = f"""
    <!DOCTYPE html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <title>HavnAI Leaderboard</title>
      <style>
        body {{ font-family: Arial, sans-serif; background:#03121b; color:#e6f6ff; padding:2rem; }}
        table {{ width:100%; border-collapse:collapse; margin-top:1.5rem; }}
        th, td {{ padding:0.75rem 1rem; border-bottom:1px solid rgba(95,228,255,0.15); }}
        th {{ text-transform:uppercase; font-size:0.75rem; letter-spacing:0.08em; color:#5fe4ff; }}
        a {{ color:#5fe4ff; }}
      </style>
    </head>
    <body>
      <h1>HavnAI Network Leaderboard</h1>
      <p><a href=\"/dashboard\">Back to dashboard</a> · <a href=\"/join\">Join the grid</a></p>
      <table>
        <thead><tr><th>Rank</th><th>Node Name(s)</th><th>Wallet</th><th>Jobs</th><th>24h Rewards</th><th>All-Time</th><th>Creator</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </body>
    </html>
    """
    return html


# ---------------------------------------------------------------------------
# Client asset helpers
# ---------------------------------------------------------------------------


@app.route("/client/download")
def client_download() -> Any:
    return send_file(CLIENT_PATH, as_attachment=True, download_name="havnai_client.py")


@app.route("/client/requirements")
def client_requirements() -> Any:
    return send_file(CLIENT_REQUIREMENTS, as_attachment=True, download_name="requirements-node.txt")


@app.route("/client/version")
def client_version() -> Any:
    return APP_VERSION, 200, {"Content-Type": "text/plain"}


# ---------------------------------------------------------------------------
# API routes – models catalog
# ---------------------------------------------------------------------------


@app.route("/models/list")
def list_models() -> Any:
    return jsonify({"models": build_models_catalog()})


@app.route("/models/download/<path:filename>")
def download_model(filename: str) -> Any:
    return send_from_directory(CREATOR_MODEL_DIR, filename, as_attachment=True)


# ---------------------------------------------------------------------------
# API routes – jobs & nodes
# ---------------------------------------------------------------------------


@app.route("/submit-job", methods=["POST"])
def submit_job() -> Any:
    if not rate_limit(f"submit-job:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    payload = request.get_json() or {}
    wallet = str(payload.get("wallet", "")).strip()
    model_name = str(payload.get("model", "")).lower()
    task_type = (payload.get("task_type") or "ai").lower()
    job_data = payload.get("data") or payload.get("prompt", "")
    weight = payload.get("weight")

    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400

    cfg = get_model_config(model_name)
    if not cfg:
        return jsonify({"error": "unknown model"}), 400

    if weight is None:
        weight = cfg.get("reward_weight", resolve_weight(model_name, 5.0))
    task_type = cfg.get("task_type", task_type)

    with LOCK:
        job_id = enqueue_job(wallet, cfg.get("name", model_name), task_type, job_data, float(weight))
    log_event("Public job queued", wallet=wallet, model=model_name, job_id=job_id)
    return jsonify({"status": "queued", "job_id": job_id}), 200


@app.route("/register", methods=["POST"])
def register() -> Any:
    if not rate_limit(f"register:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403

    data = request.get_json() or {}
    node_id = data.get("node_id")
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    with LOCK:
        node = NODES.get(node_id)
        if not node:
            node = {
                "os": data.get("os", "unknown"),
                "gpu": data.get("gpu", {}),
                "rewards": 0.0,
                "avg_utilization": 0.0,
                "utilization": data.get("utilization", 0.0),
                "tasks_completed": 0,
                "current_task": None,
                "last_result": {},
                "reward_history": [],
                "last_reward": 0.0,
                "start_time": data.get("start_time", unix_now()),
                "role": data.get("role", "worker"),
                "node_name": data.get("node_name") or node_id,
            }
            NODES[node_id] = node
            log_event("Node registered", node_id=node_id, role=node["role"], version=data.get("version"))

        node["os"] = data.get("os", node.get("os", "unknown"))
        node["gpu"] = data.get("gpu", node.get("gpu", {}))
        node["role"] = data.get("role", node.get("role", "worker"))
        node["node_name"] = data.get("node_name") or node.get("node_name", node_id)
        node["version"] = data.get("version", "dev")
        util = data.get("gpu", {}).get("utilization") if isinstance(data.get("gpu"), dict) else data.get("utilization")
        util = float(util or node.get("utilization", 0.0))
        node["utilization"] = util
        samples = node.setdefault("util_samples", [])
        samples.append(util)
        if len(samples) > 60:
            samples.pop(0)
        node["avg_utilization"] = round(sum(samples) / len(samples), 2) if samples else util
        node["last_seen"] = iso_now()
        node["last_seen_unix"] = unix_now()
        save_nodes()

    return jsonify({"status": "ok", "node": node_id}), 200


@app.route("/register/models", methods=["POST"])
def register_models() -> Any:
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403
    data = request.get_json() or {}
    node_id = data.get("node_id")
    models = data.get("models", [])
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    with LOCK:
        conn = get_db()
        existing = {
            row["model_name"]
            for row in conn.execute("SELECT model_name FROM node_models WHERE node_id=?", (node_id,))
        }
        incoming = set()
        now_ts = unix_now()
        for item in models:
            name = str(item.get("name") or item.get("filename") or "model").lower()
            filename = item.get("filename") or f"{name}.ckpt"
            model_hash = item.get("hash") or ""
            size = int(item.get("size") or 0)
            tags = item.get("tags") or []
            weight = float(item.get("weight") or MODEL_WEIGHTS.get(name, 5.0))
            task_type = item.get("task_type") or ("image_gen" if filename.endswith((".safetensors", ".ckpt")) else "ai")
            conn.execute(
                """
                INSERT OR REPLACE INTO node_models (node_id, model_name, filename, hash, size, tags, weight, task_type, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (node_id, name, filename, model_hash, size, json.dumps(tags), weight, task_type, now_ts),
            )
            MODEL_WEIGHTS.setdefault(name, weight)
            incoming.add(name)
        for stale in existing - incoming:
            conn.execute("DELETE FROM node_models WHERE node_id=? AND model_name=?", (node_id, stale))
        conn.commit()
        refresh_registered_models()
    log_event("Models registered", node_id=node_id, count=len(models))
    return jsonify({"status": "ok", "registered": len(models)}), 200


@app.route("/link-wallet", methods=["POST"])
def link_wallet() -> Any:
    if not rate_limit(f"link-wallet:{request.remote_addr}", limit=30):
        return jsonify({"error": "rate limit"}), 429
    if not check_join_token():
        return jsonify({"error": "unauthorized"}), 403

    data = request.get_json() or {}
    node_id = data.get("node_id")
    wallet = data.get("wallet")
    node_name = data.get("node_name", node_id)
    if not node_id or not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid payload"}), 400

    with LOCK:
        conn = get_db()
        conn.execute(
            "INSERT OR REPLACE INTO node_wallets (node_id, wallet, node_name, updated_at) VALUES (?, ?, ?, ?)",
            (node_id, wallet, node_name, unix_now()),
        )
        conn.commit()
        node = NODES.setdefault(node_id, {
            "os": "unknown",
            "gpu": {},
            "rewards": 0.0,
            "utilization": 0.0,
            "avg_utilization": 0.0,
            "tasks_completed": 0,
            "current_task": None,
            "last_result": {},
            "reward_history": [],
            "last_reward": 0.0,
            "start_time": unix_now(),
            "role": "worker",
        })
        node["wallet"] = wallet
        node["node_name"] = node_name or node.get("node_name", node_id)
        save_nodes()
    log_event("Wallet linked", node_id=node_id, wallet=wallet)
    return jsonify({"status": "linked"}), 200


@app.route("/tasks/ai", methods=["GET"])
def get_ai_tasks() -> Any:
    node_id = request.args.get("node_id")
    if not node_id:
        return jsonify({"tasks": []}), 200

    with LOCK:
        node_info = NODES.get(node_id)
        if not node_info:
            return jsonify({"tasks": []}), 200

        pending = pending_tasks_for_node(node_id)
        if not pending:
            job = fetch_next_job_for_node(node_id)
            if job:
                cfg = get_model_config(job["model"])
                if cfg:
                    assign_job_to_node(job["id"], node_id)
                    pending = [
                        {
                            "task_id": job["id"],
                            "task_type": job["task_type"] or cfg.get("task_type", "ai"),
                            "model_name": job["model"],
                            "model_url": cfg.get("url", ""),
                            "input_shape": cfg.get("input_shape", [1, 64]),
                            "reward_weight": float(job["weight"] or cfg.get("reward_weight", 1.0)),
                            "status": "pending",
                            "wallet": job["wallet"],
                        }
                    ]
                    node_info["current_task"] = {
                        "task_id": job["id"],
                        "model_name": job["model"],
                        "status": "pending",
                        "task_type": pending[0]["task_type"],
                        "weight": pending[0]["reward_weight"],
                    }
                    save_nodes()
                else:
                    complete_job(job["id"], "failed")
            if not pending:
                fallback = issue_internal_task(node_id)
                pending = [fallback]
                node_info["current_task"] = {
                    "task_id": fallback["task_id"],
                    "model_name": fallback["model_name"],
                    "status": fallback["status"],
                    "task_type": fallback.get("task_type", "ai"),
                    "weight": fallback.get("reward_weight", 1.0),
                }
                save_nodes()

        response_tasks = []
        for task in pending:
            if task["status"] == "pending":
                task["status"] = "assigned"
                task["assigned_at"] = unix_now()
            TASKS[task["task_id"]] = dict(task)
            response_tasks.append(
                {
                    "task_id": task["task_id"],
                    "type": task.get("task_type", "ai"),
                    "model_name": task["model_name"],
                    "model_url": task.get("model_url", ""),
                    "input_shape": task.get("input_shape", [1, 64]),
                    "reward_weight": task.get("reward_weight", 1.0),
                }
            )
    return jsonify({"tasks": response_tasks}), 200


@app.route("/tasks", methods=["GET"])
def tasks_alias() -> Any:
    return get_ai_tasks()


@app.route("/results", methods=["POST"])
def submit_results() -> Any:
    data = request.get_json() or {}
    node_id = data.get("node_id")
    task_id = data.get("task_id")
    status = data.get("status", "unknown")
    metrics = data.get("metrics", {})
    utilization = data.get("utilization")

    if not node_id or not task_id:
        return jsonify({"error": "missing node_id or task_id"}), 400

    with LOCK:
        task = TASKS.get(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404

        task["status"] = status
        task["completed_at"] = unix_now()
        task["result"] = metrics

        node = NODES.get(node_id)
        reward = 0.0
        wallet = task.get("wallet")
        model_name = task.get("model_name", "ai-model")
        task_type = task.get("task_type", "ai")
        if node:
            inference_time = float(metrics.get("inference_time_ms") or metrics.get("duration", 0) * 1000)
            gpu_util = float(utilization or node.get("utilization", 0.0))
            reward_weight = float(task.get("reward_weight", resolve_weight(model_name, 1.0)))
            if inference_time > 0:
                reward = round((gpu_util * reward_weight) / inference_time, 6)
            node["rewards"] = round(node.get("rewards", 0.0) + reward, 6)
            node["last_reward"] = reward
            history = node.setdefault("reward_history", [])
            history.append({"reward": reward, "task_id": task_id, "timestamp": iso_now()})
            if len(history) > 20:
                history.pop(0)
            node["last_seen"] = iso_now()
            node["last_seen_unix"] = unix_now()
            node["last_result"] = {
                "task_id": task_id,
                "status": status,
                "metrics": metrics,
                "model_name": model_name,
                "reward": reward,
                "wallet": wallet,
                "task_type": task_type,
            }
            if status == "success":
                node["tasks_completed"] = node.get("tasks_completed", 0) + 1
            node["current_task"] = None
            if utilization is not None:
                try:
                    util_val = float(utilization)
                    node["utilization"] = util_val
                    samples = node.setdefault("util_samples", [])
                    samples.append(util_val)
                    if len(samples) > 60:
                        samples.pop(0)
                    node["avg_utilization"] = round(sum(samples) / len(samples), 2)
                except (TypeError, ValueError):
                    pass
            save_nodes()

        job = get_job(task_id)
        if job:
            complete_job(task_id, status)
            wallet = wallet or job.get("wallet")

        record_reward(wallet, task_id, reward)

        stats = MODEL_STATS.setdefault(model_name, {"count": 0.0, "total_time": 0.0})
        if status == "success":
            inference_time = float(metrics.get("inference_time_ms", 0))
            if inference_time > 0:
                stats["count"] += 1
                stats["total_time"] += inference_time

        TASKS.pop(task_id, None)

    log_event(
        "Task completed",
        node_id=node_id,
        task_id=task_id,
        model=model_name,
        reward=reward,
        status=status,
    )
    return jsonify({"status": "received", "task": task_id, "reward": reward}), 200


@app.route("/rewards", methods=["GET"])
def rewards_endpoint() -> Any:
    job_summary = get_job_summary()
    with LOCK:
        rewards = {node_id: info.get("rewards", 0.0) for node_id, info in NODES.items()}
    return jsonify({"rewards": rewards, "total": job_summary["total_distributed"]})


@app.route("/logs", methods=["GET"])
def logs_endpoint() -> Any:
    with LOCK:
        entries = list(EVENT_LOGS)[-50:]
    return jsonify({"logs": entries})


@app.route("/feed", methods=["GET"])
def feed_catalog() -> Any:
    return jsonify({"models": build_models_catalog()})


@app.route("/nodes", methods=["GET"])
def nodes_endpoint() -> Any:
    with LOCK:
        now = unix_now()
        payload = []
        total_util = 0.0
        total_rewards = 0.0
        online_count = 0
        for node_id, info in NODES.items():
            last_seen_unix = info.get("last_seen_unix", parse_timestamp(info.get("last_seen")))
            online = (now - last_seen_unix) <= ONLINE_THRESHOLD
            if online:
                online_count += 1
            avg_util = float(info.get("avg_utilization", info.get("utilization", 0.0)))
            total_util += avg_util
            rewards = float(info.get("rewards", 0.0))
            total_rewards += rewards
            start_time = parse_timestamp(info.get("start_time"))
            uptime_seconds = max(0, int(now - start_time))
            last_result = info.get("last_result", {})
            model_name = last_result.get("model_name") or info.get("current_task", {}).get("model_name")
            inference_time = last_result.get("metrics", {}).get("inference_time_ms")
            current_task = info.get("current_task") or {}
            task_type = last_result.get("task_type") or current_task.get("task_type") or "ai"
            weight = (
                last_result.get("metrics", {}).get("reward_weight")
                or current_task.get("weight")
                or MODEL_WEIGHTS.get(model_name or "mlp-classifier", 1.0)
            )
            try:
                weight = float(weight)
            except (TypeError, ValueError):
                weight = resolve_weight(model_name or "mlp-classifier", 1.0)
            payload.append(
                {
                    "node_id": node_id,
                    "node_name": info.get("node_name", node_id),
                    "role": info.get("role", "worker"),
                    "wallet": info.get("wallet"),
                    "task_type": task_type,
                    "model_name": model_name,
                    "model_weight": weight,
                    "inference_time_ms": inference_time,
                    "gpu_utilization": info.get("utilization", 0.0),
                    "avg_utilization": avg_util,
                    "rewards": rewards,
                    "last_reward": info.get("last_reward", 0.0),
                    "last_seen": info.get("last_seen"),
                    "uptime_human": format_duration(uptime_seconds),
                    "status": "online" if online else "offline",
                    "last_result": last_result,
                }
            )
        total_nodes = len(payload)
        summary = {
            "timestamp": iso_now(),
            "total_nodes": total_nodes,
            "online_nodes": online_count,
            "offline_nodes": total_nodes - online_count,
            "avg_utilization": round(total_util / total_nodes, 2) if total_nodes else 0.0,
            "total_rewards": round(total_rewards, 6),
        }
        job_summary = get_job_summary()
        models_catalog = build_models_catalog()
    summary["tasks_backlog"] = job_summary["queued_jobs"]
    return jsonify(
        {
            "nodes": payload,
            "summary": summary,
            "job_summary": job_summary,
            "models_catalog": models_catalog,
        }
    )


@app.route("/dashboard")
def dashboard() -> Any:
    return send_from_directory(STATIC_DIR, "dashboard.html")


@app.route("/")
def root() -> Any:
    return dashboard()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

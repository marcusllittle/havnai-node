from __future__ import annotations

import json
import random
import re
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import onnx
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from onnx import TensorProto, helper

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS_DIR = STATIC_DIR / "models"
REGISTRY_FILE = BASE_DIR / "nodes.json"
DB_PATH = BASE_DIR / "db" / "ledger.db"

MODEL_STATS: Dict[str, Dict[str, float]] = {}
EVENT_LOGS: deque = deque(maxlen=200)
NODES: Dict[str, Dict[str, Any]] = {}
TASKS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()
DB_CONN: Optional[sqlite3.Connection] = None

ONLINE_THRESHOLD = 120  # seconds before a node is considered offline
WALLET_REGEX = re.compile(r"^0x[a-fA-F0-9]{40}$")

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)


# ---------------------------------------------------------------------------
# Time helpers
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


def log_event(message: str, level: str = "info") -> None:
    entry = {"timestamp": iso_now(), "level": level, "message": message}
    EVENT_LOGS.append(entry)
    print(f"[{entry['timestamp']}] ({level.upper()}) {message}")


# ---------------------------------------------------------------------------
# Stage 5A additions: SQLite helpers for public job + reward ledger
# ---------------------------------------------------------------------------

DB_PATH.parent.mkdir(parents=True, exist_ok=True)


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
    conn.commit()


def enqueue_job(wallet: str, model: str, data: str) -> str:
    job_id = f"job-{uuid.uuid4().hex[:12]}"
    conn = get_db()
    conn.execute(
        "INSERT INTO jobs (id, wallet, model, data, status, node_id, timestamp) VALUES (?, ?, ?, ?, 'queued', NULL, ?)",
        (job_id, wallet, model, data, unix_now()),
    )
    conn.commit()
    return job_id


def fetch_next_job() -> Optional[Dict[str, Any]]:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM jobs WHERE status='queued' ORDER BY timestamp ASC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def assign_job_to_node(job_id: str, node_id: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE jobs SET status='running', node_id=?, assigned_at=? WHERE id=?",
        (node_id, unix_now(), job_id),
    )
    conn.commit()


def complete_job(job_id: str, status: str) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE jobs SET status=?, completed_at=? WHERE id=?",
        (status, unix_now(), job_id),
    )
    conn.commit()


def record_reward(wallet: str, task_id: str, reward: float) -> None:
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


def get_job_summary(limit: int = 10) -> Dict[str, Any]:
    conn = get_db()
    queued = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='queued'").fetchone()[0]
    active = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='running'").fetchone()[0]
    total_distributed = conn.execute(
        "SELECT COALESCE(SUM(reward_hai), 0) FROM rewards"
    ).fetchone()[0]
    rows = conn.execute(
        """
        SELECT jobs.id, jobs.wallet, jobs.model, jobs.status, jobs.completed_at, rewards.reward_hai
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
                "status": row["status"],
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
# Model management (Stage 4 foundation)
# ---------------------------------------------------------------------------

def ensure_directories() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


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


MODEL_CATALOG = [
    {
        "name": "mlp-classifier",
        "filename": "mlp-classifier.onnx",
        "input_shape": [1, 64],
        "reward_weight": 1.25,
        "builder": lambda path: build_mlp_model(path, 64, 32, 10),
    },
    {
        "name": "conv-demo",
        "filename": "conv-demo.onnx",
        "input_shape": [1, 1, 16, 16],
        "reward_weight": 1.5,
        "builder": build_conv_model,
    },
]


def get_model_config(name: str) -> Optional[Dict[str, Any]]:
    for cfg in MODEL_CATALOG:
        if cfg["name"] == name:
            return cfg
    return None


def ensure_model_catalog() -> None:
    ensure_directories()
    for cfg in MODEL_CATALOG:
        path = MODELS_DIR / cfg["filename"]
        builder = cfg.get("builder")
        if callable(builder):
            builder(path)
        cfg["path"] = path
        cfg["url"] = f"/static/models/{path.name}"
        MODEL_STATS.setdefault(cfg["name"], {"count": 0.0, "total_time": 0.0})


# ---------------------------------------------------------------------------
# Node persistence
# ---------------------------------------------------------------------------

def load_nodes() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return {}
    with REGISTRY_FILE.open() as f:
        data = json.load(f)
    current = unix_now()
    for info in data.values():
        info.setdefault("os", "unknown")
        info.setdefault("gpu", {})
        info.setdefault("rewards", 0.0)
        info.setdefault("utilization", 0)
        info.setdefault("avg_utilization", info.get("utilization", 0))
        info.setdefault("tasks_completed", 0)
        info.setdefault("current_task", None)
        info.setdefault("last_result", {})
        info.setdefault("reward_history", [])
        info.setdefault("last_reward", 0.0)
        info.setdefault("start_time", current)
        info.setdefault("last_seen", iso_now())
        info["last_seen_unix"] = parse_timestamp(info["last_seen"])
    return data


def save_nodes() -> None:
    payload: Dict[str, Dict[str, Any]] = {}
    for node_id, info in NODES.items():
        serial = dict(info)
        serial.pop("last_seen_unix", None)
        payload[node_id] = serial
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_FILE.open("w") as f:
        json.dump(payload, f, indent=2)


ensure_model_catalog()
init_db()
NODES = load_nodes()
log_event(f"HavnAI telemetry online with {len(NODES)} cached node(s).", "info")


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def pending_tasks_for_node(node_id: str) -> List[Dict[str, Any]]:
    return [task for task in TASKS.values() if task["assigned_to"] == node_id and task["status"] in {"pending", "assigned"}]


# Stage 5A additions: nodes pick up public jobs from queue
def issue_public_job_task(node_id: str) -> Optional[Dict[str, Any]]:
    job = fetch_next_job()
    if not job:
        return None

    cfg = get_model_config(job["model"])
    if not cfg:
        log_event(f"Unknown model '{job['model']}' for job {job['id']}. Marking failed.", "error")
        complete_job(job["id"], "failed")
        return None

    assign_job_to_node(job["id"], node_id)
    task = {
        "task_id": job["id"],
        "type": "ai",
        "model_name": cfg["name"],
        "model_url": cfg["url"],
        "input_shape": cfg["input_shape"],
        "reward_weight": cfg["reward_weight"],
        "assigned_to": node_id,
        "status": "pending",
        "job_id": job["id"],
        "wallet": job["wallet"],
    }
    TASKS[task["task_id"]] = dict(task)
    log_event(f"Node {node_id} claimed public job {job['id']} for wallet {job['wallet']}.", "info")
    return task


def issue_internal_task(node_id: str) -> Dict[str, Any]:
    cfg = random.choice(MODEL_CATALOG)
    task_id = f"ai-{uuid.uuid4().hex[:10]}"
    task = {
        "task_id": task_id,
        "type": "ai",
        "model_name": cfg["name"],
        "model_url": cfg["url"],
        "input_shape": cfg["input_shape"],
        "reward_weight": cfg["reward_weight"],
        "assigned_to": node_id,
        "status": "pending",
        "job_id": None,
        "wallet": None,
    }
    TASKS[task_id] = dict(task)
    return task


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------


@app.route("/submit-job", methods=["POST"])
def submit_job():
    # Stage 5A additions: public job submission endpoint
    payload = request.get_json() or {}
    wallet = str(payload.get("wallet", "")).strip()
    model_name = payload.get("model")
    job_data = payload.get("data", "")

    if not wallet or not WALLET_REGEX.match(wallet):
        return jsonify({"error": "invalid wallet"}), 400
    cfg = get_model_config(model_name)
    if not cfg:
        return jsonify({"error": "unknown model"}), 400

    with LOCK:
        job_id = enqueue_job(wallet, cfg["name"], job_data)
    log_event(f"Public job {job_id} queued for wallet {wallet} using {cfg['name']}.", "info")
    return jsonify({"status": "queued", "job_id": job_id}), 200


@app.route("/register", methods=["POST"])
def register():
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
                "avg_utilization": 0,
                "utilization": data.get("utilization", 0),
                "tasks_completed": 0,
                "current_task": None,
                "last_result": {},
                "reward_history": [],
                "last_reward": 0.0,
                "start_time": data.get("start_time", unix_now()),
            }
            NODES[node_id] = node
            log_event(f"Node {node_id} registered.")

        node["os"] = data.get("os", node.get("os", "unknown"))
        node["gpu"] = data.get("gpu", node.get("gpu", {}))
        util = data.get("gpu", {}).get("utilization") if isinstance(data.get("gpu"), dict) else data.get("utilization")
        util = float(util or node.get("utilization", 0))
        node["utilization"] = util
        samples = node.setdefault("util_samples", [])
        samples.append(util)
        if len(samples) > 60:
            samples.pop(0)
        node["avg_utilization"] = round(sum(samples) / len(samples), 2) if samples else util
        node["last_seen"] = iso_now()
        node["last_seen_unix"] = unix_now()
        node.setdefault("start_time", data.get("start_time", unix_now()))
        save_nodes()

    return jsonify({"status": "ok", "node": node_id}), 200


@app.route("/tasks/ai", methods=["GET"])
def get_ai_tasks():
    node_id = request.args.get("node_id")
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    with LOCK:
        if node_id not in NODES:
            return jsonify({"tasks": []}), 200

        pending = pending_tasks_for_node(node_id)
        if not pending:
            task = issue_public_job_task(node_id)
            if not task:
                task = issue_internal_task(node_id)
            pending = [task]
            NODES[node_id]["current_task"] = {
                "task_id": pending[0]["task_id"],
                "model_name": pending[0]["model_name"],
                "status": pending[0]["status"],
            }
            save_nodes()

        response_tasks = []
        for task in pending:
            if task["status"] == "pending":
                task["status"] = "assigned"
                task["assigned_at"] = unix_now()
            response_tasks.append(
                {
                    "task_id": task["task_id"],
                    "type": "ai",
                    "model_name": task["model_name"],
                    "model_url": task["model_url"],
                    "input_shape": task["input_shape"],
                    "reward_weight": task.get("reward_weight", 1.0),
                }
            )

    return jsonify({"tasks": response_tasks}), 200


@app.route("/tasks", methods=["GET"])
def tasks_alias():
    return get_ai_tasks()


@app.route("/results", methods=["POST"])
def submit_results():
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
        wallet = None
        model_name = task.get("model_name", "ai-model")
        if node:
            wallet = task.get("wallet")
            inference_time = float(metrics.get("inference_time_ms") or metrics.get("duration", 0) * 1000)
            gpu_util = float(utilization or node.get("utilization", 0))
            reward_weight = float(task.get("reward_weight", 1.0))
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

        # Stage 5A additions: reconcile job ledger + reward accounting
        job = get_job(task_id)
        if job:
            complete_job(task_id, status)
            wallet = wallet or job.get("wallet")

        if wallet:
            record_reward(wallet, task_id, reward)

        stats = MODEL_STATS.setdefault(model_name, {"count": 0.0, "total_time": 0.0})
        if status == "success":
            inference_time = float(metrics.get("inference_time_ms", 0))
            if inference_time > 0:
                stats["count"] += 1
                stats["total_time"] += inference_time

        TASKS.pop(task_id, None)

    log_event(
        f"Node {node_id} completed AI task {task_id[:8]} using {model_name} in {metrics.get('inference_time_ms', 'n/a')} ms (reward {reward} HAI).",
        "info",
    )

    return jsonify({"status": "received", "task": task_id, "reward": reward}), 200


@app.route("/nodes", methods=["GET"])
def get_nodes():
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
            avg_util = float(info.get("avg_utilization", info.get("utilization", 0)))
            total_util += avg_util
            rewards = float(info.get("rewards", 0.0))
            total_rewards += rewards
            start_time = parse_timestamp(info.get("start_time"))
            uptime_seconds = max(0, int(now - start_time))
            last_result = info.get("last_result", {})
            model_name = last_result.get("model_name") or info.get("current_task", {}).get("model_name")
            inference_time = last_result.get("metrics", {}).get("inference_time_ms")
            payload.append(
                {
                    "node_id": node_id,
                    "model_name": model_name,
                    "inference_time_ms": inference_time,
                    "gpu_utilization": info.get("utilization", 0),
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
        # Stage 5A additions: expose job/reward telemetry to dashboard
        job_summary = get_job_summary()
    summary["tasks_backlog"] = job_summary["queued_jobs"]
    return jsonify(
        {
            "nodes": payload,
            "summary": summary,
            "job_summary": job_summary,
        }
    ), 200


@app.route("/rewards", methods=["GET"])
def get_rewards():
    with LOCK:
        rewards = {node_id: info.get("rewards", 0.0) for node_id, info in NODES.items()}
    total_distributed = get_job_summary()["total_distributed"]
    return jsonify({"rewards": rewards, "total": total_distributed}), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    with LOCK:
        entries = list(EVENT_LOGS)[-50:]
    return jsonify({"logs": entries}), 200


@app.route("/feed", methods=["GET"])
def feed_catalog():
    response = []
    for cfg in MODEL_CATALOG:
        stats = MODEL_STATS.get(cfg["name"], {"count": 0.0, "total_time": 0.0})
        avg_time = 0.0
        if stats["count"]:
            avg_time = stats["total_time"] / stats["count"]
        response.append(
            {
                "model_name": cfg["name"],
                "input_shape": cfg["input_shape"],
                "reward_weight": cfg["reward_weight"],
                "avg_inference_time_ms": round(avg_time, 2),
            }
        )
    return jsonify({"models": response}), 200


@app.route("/dashboard")
def dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/")
def root():
    return dashboard()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

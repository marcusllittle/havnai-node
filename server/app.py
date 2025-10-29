from __future__ import annotations

import json
import random
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import onnx
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from onnx import TensorProto, helper

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS_DIR = STATIC_DIR / "models"
REGISTRY_FILE = BASE_DIR / "nodes.json"

MODEL_STATS: Dict[str, Dict[str, float]] = {}
EVENT_LOGS: deque = deque(maxlen=200)
NODES: Dict[str, Dict[str, Any]] = {}
TASKS: Dict[str, Dict[str, Any]] = {}
LOCK = threading.Lock()

ONLINE_THRESHOLD = 120  # seconds

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)


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
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def log_event(message: str, level: str = "info") -> None:
    entry = {"timestamp": iso_now(), "level": level, "message": message}
    EVENT_LOGS.append(entry)
    print(f"[{entry['timestamp']}] ({level.upper()}) {message}")


# ---------------------------------------------------------------------------
# Model generation
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
        "builder": lambda path: build_mlp_model(path, input_dim=64, hidden_dim=32, output_dim=10),
    },
    {
        "name": "conv-demo",
        "filename": "conv-demo.onnx",
        "input_shape": [1, 1, 16, 16],
        "reward_weight": 1.5,
        "builder": build_conv_model,
    },
]


def ensure_model_catalog() -> None:
    ensure_directories()
    for item in MODEL_CATALOG:
        path = MODELS_DIR / item["filename"]
        builder = item.get("builder")
        if callable(builder):
            builder(path)
        item["path"] = path
        item["url"] = f"/static/models/{path.name}"
        MODEL_STATS.setdefault(item["name"], {"count": 0.0, "total_time": 0.0})


# ---------------------------------------------------------------------------
# Persistence
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
NODES = load_nodes()
log_event(f"HavnAI telemetry online with {len(NODES)} cached node(s).", "info")


# ---------------------------------------------------------------------------
# Task helpers
# ---------------------------------------------------------------------------

def pending_tasks_for_node(node_id: str) -> List[Dict[str, Any]]:
    return [task for task in TASKS.values() if task["assigned_to"] == node_id and task["status"] in {"pending", "assigned"}]


def issue_ai_task(node_id: str) -> Dict[str, Any]:
    model = random.choice(MODEL_CATALOG)
    task_id = f"ai-{random.randrange(10**12, 10**13)}"
    task = {
        "task_id": task_id,
        "type": "ai",
        "model_name": model["name"],
        "model_url": model["url"],
        "input_shape": model["input_shape"],
        "reward_weight": model["reward_weight"],
        "assigned_to": node_id,
        "status": "pending",
        "created_at": unix_now(),
        "assigned_at": None,
        "completed_at": None,
    }
    TASKS[task_id] = task
    return task


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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
            task = issue_ai_task(node_id)
            pending = [task]
            NODES[node_id]["current_task"] = {
                "task_id": task["task_id"],
                "model_name": task["model_name"],
                "status": task["status"],
            }
            save_nodes()

        response_tasks = []
        for task in pending:
            if task["status"] == "pending":
                task["status"] = "assigned"
                task["assigned_at"] = unix_now()
            response_tasks.append({
                "task_id": task["task_id"],
                "type": "ai",
                "model_name": task["model_name"],
                "model_url": task["model_url"],
                "input_shape": task["input_shape"],
                "reward_weight": task["reward_weight"],
            })

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
        if node:
            model_name = task.get("model_name", "unknown-model")
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
            }
            if status == "success":
                node["tasks_completed"] = node.get("tasks_completed", 0) + 1
            node["current_task"] = None
            if utilization is not None:
                node["utilization"] = float(utilization)
            save_nodes()

        model_name = task.get("model_name", "ai-model")
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
        total_rewards = 0.0
        total_util = 0.0
        online_count = 0
        backlog = 0
        for node_id, info in NODES.items():
            last_seen_unix = info.get("last_seen_unix", parse_timestamp(info.get("last_seen")))
            online = (now - last_seen_unix) <= ONLINE_THRESHOLD
            if online:
                online_count += 1
            pending = len(pending_tasks_for_node(node_id))
            backlog += pending
            avg_util = float(info.get("avg_utilization", info.get("utilization", 0)))
            total_util += avg_util
            rewards = float(info.get("rewards", 0.0))
            total_rewards += rewards
            start_time = parse_timestamp(info.get("start_time"))
            uptime_seconds = max(0, int(now - start_time))
            last_result = info.get("last_result", {})
            model_name = last_result.get("model_name") or info.get("current_task", {}).get("model_name")
            inference_time = last_result.get("metrics", {}).get("inference_time_ms")
            payload.append({
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
            })
        total_nodes = len(payload)
        summary = {
            "timestamp": iso_now(),
            "total_nodes": total_nodes,
            "online_nodes": online_count,
            "offline_nodes": total_nodes - online_count,
            "avg_utilization": round(total_util / total_nodes, 2) if total_nodes else 0.0,
            "tasks_backlog": backlog,
            "total_rewards": round(total_rewards, 6),
        }
    return jsonify({"nodes": payload, "summary": summary}), 200


@app.route("/rewards", methods=["GET"])
def get_rewards():
    with LOCK:
        rewards = {node_id: info.get("rewards", 0.0) for node_id, info in NODES.items()}
        total = round(sum(rewards.values()), 6)
    return jsonify({"rewards": rewards, "total": total}), 200


@app.route("/logs", methods=["GET"])
def get_logs():
    with LOCK:
        entries = list(EVENT_LOGS)[-50:]
    return jsonify({"logs": entries}), 200


@app.route("/feed", methods=["GET"])
def feed_catalog():
    response = []
    for item in MODEL_CATALOG:
        stats = MODEL_STATS.get(item["name"], {"count": 0.0, "total_time": 0.0})
        avg_time = 0.0
        if stats["count"]:
            avg_time = stats["total_time"] / stats["count"]
        response.append(
            {
                "model_name": item["name"],
                "input_shape": item["input_shape"],
                "reward_weight": item["reward_weight"],
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

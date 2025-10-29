from __future__ import annotations

import json
import random
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REGISTRY_FILE = BASE_DIR / "nodes.json"

REWARD_INTERVAL = 60  # seconds
TASK_SCAN_INTERVAL = 20  # seconds
REWARD_MULTIPLIER = 0.01  # reward += avg_utilization * 0.01 each minute
ONLINE_THRESHOLD = 120  # seconds
MAX_LOG_ENTRIES = 200

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
CORS(app)

lock = threading.Lock()
nodes: Dict[str, Dict[str, Any]] = {}
tasks: Dict[str, Dict[str, Any]] = {}
event_logs: deque = deque(maxlen=MAX_LOG_ENTRIES)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def now_ts() -> float:
    return time.time()


def iso_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_timestamp(ts: Any) -> float:
    if ts is None:
        return now_ts()
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            return datetime.fromisoformat(ts.replace("Z", "")).timestamp()
        except ValueError:
            try:
                return float(ts)
            except ValueError:
                return now_ts()
    return now_ts()


def format_duration(seconds: int) -> str:
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
    event_logs.append(entry)
    print(f"[{entry['timestamp']}] ({level.upper()}) {message}")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_nodes() -> Dict[str, Dict[str, Any]]:
    if not REGISTRY_FILE.exists():
        return {}

    with REGISTRY_FILE.open() as f:
        data = json.load(f)

    current = now_ts()
    for node_id, info in data.items():
        info.setdefault("os", "unknown")
        info.setdefault("gpu", {})
        info.setdefault("rewards", 0.0)
        info.setdefault("utilization", 0)
        info.setdefault("avg_utilization", info.get("utilization", 0))
        info.setdefault("utilization_samples", [])
        info.setdefault("tasks_completed", 0)
        info.setdefault("current_task", None)
        info.setdefault("last_result", {})
        info.setdefault("last_seen", iso_now())
        info["last_seen_unix"] = parse_timestamp(info.get("last_seen"))
        start_time = info.get("start_time")
        info["start_time"] = parse_timestamp(start_time) if start_time else current
    return data


def save_nodes(nodes_data: Dict[str, Dict[str, Any]]) -> None:
    serializable: Dict[str, Dict[str, Any]] = {}
    for node_id, info in nodes_data.items():
        data = dict(info)
        # ensure only JSON-friendly values
        data["utilization_samples"] = list(data.get("utilization_samples", []))
        data.pop("last_seen_unix", None)
        serializable[node_id] = data

    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_FILE.open("w") as f:
        json.dump(serializable, f, indent=2)


nodes = load_nodes()
for node in nodes.values():
    node["last_seen_unix"] = parse_timestamp(node.get("last_seen"))

log_event(f"Booted telemetry server with {len(nodes)} cached node(s).")


# ---------------------------------------------------------------------------
# Reward + simulation engines
# ---------------------------------------------------------------------------

def reward_loop() -> None:
    while True:
        time.sleep(REWARD_INTERVAL)
        with lock:
            updated = False
            for info in nodes.values():
                avg_util = info.get("avg_utilization")
                if avg_util is None:
                    avg_util = info.get("utilization", 0)
                reward_increment = float(avg_util) * REWARD_MULTIPLIER
                info["rewards"] = round(info.get("rewards", 0.0) + reward_increment, 6)
                updated = True
            if updated:
                save_nodes(nodes)


def pending_tasks_for_node(node_id: str) -> List[Dict[str, Any]]:
    return [
        task
        for task in tasks.values()
        if task["assigned_to"] == node_id and task["status"] in {"pending", "assigned"}
    ]


def create_task_for_node(node_id: str) -> Dict[str, Any]:
    task_type = random.choice(["matrix_multiply", "vector_dot", "noise_simulation"])
    if task_type == "matrix_multiply":
        size = random.choice([64, 96, 128, 192, 256])
        repeats = random.randint(1, 3)
        payload = {"size": size, "repeats": repeats}
    elif task_type == "vector_dot":
        length = random.choice([50000, 75000, 100000])
        repeats = random.randint(2, 5)
        payload = {"length": length, "repeats": repeats}
    else:  # noise simulation / random workloads
        iterations = random.randint(20000, 60000)
        payload = {"iterations": iterations}

    task = {
        "task_id": str(uuid.uuid4()),
        "type": task_type,
        "payload": payload,
        "assigned_to": node_id,
        "status": "pending",
        "created_at": now_ts(),
        "assigned_at": None,
        "completed_at": None,
    }
    tasks[task["task_id"]] = task
    log_event(f"Scheduled {task_type} task {task['task_id'][:8]} for {node_id}.")
    return task


def simulation_loop() -> None:
    while True:
        time.sleep(TASK_SCAN_INTERVAL)
        with lock:
            current_time = now_ts()
            updated = False
            for node_id, info in nodes.items():
                last_seen = info.get("last_seen_unix", current_time)
                if current_time - last_seen > ONLINE_THRESHOLD:
                    continue  # node offline

                active_tasks = pending_tasks_for_node(node_id)
                if not active_tasks:
                    task = create_task_for_node(node_id)
                    info["current_task"] = {
                        "task_id": task["task_id"],
                        "type": task["type"],
                        "status": task["status"],
                    }
                    updated = True
            if updated:
                save_nodes(nodes)


threading.Thread(target=reward_loop, daemon=True).start()
threading.Thread(target=simulation_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json() or {}
    node_id = data.get("node_id")
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    is_new_node = False
    with lock:
        node = nodes.get(node_id)
        if node is None:
            node = {
                "os": data.get("os", "unknown"),
                "gpu": data.get("gpu", {}),
                "start_time": data.get("start_time", now_ts()),
                "rewards": 0.0,
                "utilization": 0,
                "avg_utilization": 0,
                "utilization_samples": [],
                "tasks_completed": 0,
                "current_task": None,
                "last_result": {},
            }
            nodes[node_id] = node
            is_new_node = True

        node["os"] = data.get("os", node.get("os", "unknown"))
        node["gpu"] = data.get("gpu", node.get("gpu", {}))
        node.setdefault("utilization_samples", [])
        gpu_payload = node["gpu"] if isinstance(node["gpu"], dict) else {}
        utilization = gpu_payload.get("utilization")
        if utilization is None:
            utilization = data.get("utilization", node.get("utilization", 0))
        utilization = float(utilization or 0)
        node["utilization"] = utilization
        samples = node.setdefault("utilization_samples", [])
        samples.append(utilization)
        if len(samples) > 60:
            samples.pop(0)
        node["avg_utilization"] = round(sum(samples) / len(samples), 2) if samples else utilization

        start_time = data.get("start_time")
        if start_time:
            node["start_time"] = parse_timestamp(start_time)
        else:
            node.setdefault("start_time", now_ts())

        node["last_seen"] = iso_now()
        node["last_seen_unix"] = now_ts()
        node.setdefault("tasks_completed", 0)
        node.setdefault("rewards", 0.0)
        node.setdefault("current_task", None)
        node.setdefault("last_result", {})

        save_nodes(nodes)

    if is_new_node:
        log_event(f"Node {node_id} registered ({node['os']}).")

    return jsonify({"status": "ok", "node": node_id}), 200


@app.route("/nodes", methods=["GET"])
def get_nodes():
    with lock:
        now = now_ts()
        nodes_payload: List[Dict[str, Any]] = []
        total_util = 0.0
        online_count = 0
        pending_total = 0
        total_rewards = 0.0

        for node_id, info in nodes.items():
            last_seen_unix = info.get("last_seen_unix", parse_timestamp(info.get("last_seen")))
            is_online = (now - last_seen_unix) <= ONLINE_THRESHOLD
            if is_online:
                online_count += 1

            pending = len(pending_tasks_for_node(node_id))
            pending_total += pending

            avg_util = float(info.get("avg_utilization", info.get("utilization", 0)))
            total_util += avg_util
            total_rewards += float(info.get("rewards", 0.0))

            uptime_seconds = int(now - info.get("start_time", now))

            node_entry = {
                "node_id": node_id,
                "os": info.get("os", "unknown"),
                "gpu": info.get("gpu", {}),
                "utilization": info.get("utilization", 0),
                "avg_utilization": avg_util,
                "rewards": info.get("rewards", 0.0),
                "tasks_completed": info.get("tasks_completed", 0),
                "current_task": info.get("current_task"),
                "last_result": info.get("last_result"),
                "last_seen": info.get("last_seen"),
                "uptime_seconds": uptime_seconds,
                "uptime_human": format_duration(uptime_seconds),
                "pending_tasks": pending,
                "online": is_online,
            }
            nodes_payload.append(node_entry)

        total_nodes = len(nodes_payload)
        avg_utilization = round(total_util / total_nodes, 2) if total_nodes else 0.0
        summary = {
            "timestamp": iso_now(),
            "total_nodes": total_nodes,
            "online_nodes": online_count,
            "offline_nodes": total_nodes - online_count,
            "avg_utilization": avg_utilization,
            "tasks_backlog": pending_total,
            "total_rewards": round(total_rewards, 6),
        }

    return jsonify({"nodes": nodes_payload, "summary": summary}), 200


@app.route("/rewards", methods=["GET"])
def get_rewards():
    with lock:
        rewards_payload = {nid: info.get("rewards", 0.0) for nid, info in nodes.items()}
        total = round(sum(rewards_payload.values()), 6)
    return jsonify({"rewards": rewards_payload, "total": total}), 200


@app.route("/tasks", methods=["GET"])
def get_tasks():
    node_id = request.args.get("node_id")
    if not node_id:
        return jsonify({"error": "missing node_id"}), 400

    with lock:
        if node_id not in nodes:
            return jsonify({"tasks": []}), 200

        pending = pending_tasks_for_node(node_id)
        if not pending:
            task = create_task_for_node(node_id)
            nodes[node_id]["current_task"] = {
                "task_id": task["task_id"],
                "type": task["type"],
                "status": task["status"],
            }
            save_nodes(nodes)
            pending = [task]

        response_tasks = []
        for task in pending:
            if task["status"] == "pending":
                task["status"] = "assigned"
                task["assigned_at"] = now_ts()
            response_tasks.append({
                "task_id": task["task_id"],
                "type": task["type"],
                "payload": task.get("payload", {}),
            })

    return jsonify({"tasks": response_tasks}), 200


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

    with lock:
        task = tasks.get(task_id)
        if not task:
            return jsonify({"error": "task not found"}), 404

        task["status"] = status
        task["completed_at"] = now_ts()
        task["result"] = metrics
        task_type = task.get("type")

        node = nodes.get(node_id)
        if node:
            node["last_seen"] = iso_now()
            node["last_seen_unix"] = now_ts()
            node["last_result"] = {
                "task_id": task_id,
                "type": task_type,
                "status": status,
                "metrics": metrics,
                "completed_at": iso_now(),
            }
            if status == "success":
                node["tasks_completed"] = node.get("tasks_completed", 0) + 1

            node["current_task"] = None

            if utilization is not None:
                try:
                    util_val = float(utilization)
                    node["utilization"] = util_val
                    samples = node.setdefault("utilization_samples", [])
                    samples.append(util_val)
                    if len(samples) > 60:
                        samples.pop(0)
                    node["avg_utilization"] = round(sum(samples) / len(samples), 2)
                except (TypeError, ValueError):
                    pass

            save_nodes(nodes)

        tasks.pop(task_id, None)

    log_event(
        f"Node {node_id} completed task {task_id[:8]} ({task_type}): {status.upper()}"
    )
    return jsonify({"status": "received", "task": task_id}), 200


@app.route("/logs", methods=["GET"])
def logs_endpoint():
    with lock:
        entries = list(event_logs)[-50:]
    return jsonify({"logs": entries}), 200


@app.route("/dashboard", methods=["GET"])
def dashboard():
    return send_from_directory(app.static_folder, "dashboard.html")


@app.route("/")
def root():
    return dashboard()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

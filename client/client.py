"""HavnAI Node Client

Lightweight worker that maintains heartbeats with the coordinator and executes
simulated AI workloads assigned via the /tasks endpoint.
"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import threading
import time
from typing import Any, Dict, Optional

import requests

try:  # optional scientific libs
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

SERVER_BASE = os.environ.get("HAVNAI_SERVER", "http://127.0.0.1:5001")
REGISTER_ENDPOINT = f"{SERVER_BASE}/register"
TASKS_ENDPOINT = f"{SERVER_BASE}/tasks"
RESULTS_ENDPOINT = f"{SERVER_BASE}/results"

NODE_ID = socket.gethostname()
SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})

HEARTBEAT_INTERVAL = 30
TASK_POLL_INTERVAL = 15
BACKOFF_BASE = 5
MAX_BACKOFF = 60

START_TIME = time.time()
utilization_hint = random.randint(5, 20)
lock = threading.Lock()


def log(msg: str, prefix: str = "ℹ️") -> None:
    print(f"{prefix} [{time.strftime('%H:%M:%S')}] {msg}")


def run_command(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
    except Exception:
        return None


def read_gpu_stats() -> Dict[str, Any]:
    """Return best-effort GPU statistics."""
    global utilization_hint

    output = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if output:
        try:
            parts = output.strip().split("\n")[0].split(", ")
            name, mem_total, mem_used, util = parts
            utilization_hint = int(util)
            return {
                "gpu_name": name,
                "memory_total": int(mem_total),
                "memory_used": int(mem_used),
                "utilization": int(util),
            }
        except Exception:
            pass

    if psutil and hasattr(psutil, "cpu_percent"):
        utilization_hint = max(5, int(psutil.cpu_percent(interval=0.2)))

    return {
        "gpu_name": "Simulated",
        "memory_total": 0,
        "memory_used": 0,
        "utilization": utilization_hint,
    }


def heartbeat_loop() -> None:
    backoff = BACKOFF_BASE
    while True:
        payload = {
            "node_id": NODE_ID,
            "os": os.uname().sysname if hasattr(os, "uname") else os.name,
            "gpu": read_gpu_stats(),
            "start_time": START_TIME,
            "uptime": time.time() - START_TIME,
        }
        try:
            resp = SESSION.post(REGISTER_ENDPOINT, data=json.dumps(payload), timeout=5)
            resp.raise_for_status()
            backoff = BACKOFF_BASE
            log("Heartbeat OK", "✅")
        except Exception as exc:
            log(f"Heartbeat failed: {exc}", "⚠️")
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
        else:
            time.sleep(HEARTBEAT_INTERVAL)


def poll_tasks_loop() -> None:
    backoff = BACKOFF_BASE
    while True:
        try:
            resp = SESSION.get(TASKS_ENDPOINT, params={"node_id": NODE_ID}, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            tasks = data.get("tasks", [])
            if tasks:
                log(f"Received {len(tasks)} task(s)", "📥")
            for task in tasks:
                execute_task(task)
            backoff = BACKOFF_BASE
        except Exception as exc:
            log(f"Task poll failed: {exc}", "⚠️")
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
        else:
            time.sleep(TASK_POLL_INTERVAL)


def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    task_type = task.get("type", "generic")
    payload = task.get("payload", {})
    log(f"Executing task {task_id[:8]} ({task_type})", "🚀")

    start = time.time()
    status = "success"
    metrics: Dict[str, Any] = {"task_type": task_type}

    try:
        if task_type == "matrix_multiply":
            metrics.update(run_matrix_workload(payload))
        elif task_type == "vector_dot":
            metrics.update(run_vector_workload(payload))
        elif task_type == "noise_simulation":
            metrics.update(run_noise_workload(payload))
        elif task_type == "onnx_infer":
            metrics.update(run_onnx_workload(payload))
        else:
            metrics["message"] = "Unknown task type"
            status = "skipped"
    except Exception as exc:  # pragma: no cover
        status = "failed"
        metrics["error"] = str(exc)

    duration = time.time() - start
    metrics["duration"] = round(duration, 4)

    util = metrics.get("estimated_utilization", utilization_hint)
    if isinstance(util, (int, float)):
        with lock:
            utilization_hint = int(max(5, min(100, util)))

    result = {
        "node_id": NODE_ID,
        "task_id": task_id,
        "status": status,
        "metrics": metrics,
        "duration": metrics["duration"],
        "utilization": utilization_hint,
        "submitted_at": time.time(),
    }

    try:
        resp = SESSION.post(RESULTS_ENDPOINT, data=json.dumps(result), timeout=8)
        resp.raise_for_status()
        log(f"Task {task_id[:8]} -> {status.upper()}", "✅" if status == "success" else "⚠️")
    except Exception as exc:  # pragma: no cover
        log(f"Failed to submit task result: {exc}", "🚫")


# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------

def run_matrix_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    size = int(payload.get("size", 128))
    repeats = max(1, min(int(payload.get("repeats", 2)), 5))
    if np is not None:
        checksum = 0.0
        start = time.time()
        for _ in range(repeats):
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            checksum += float(np.sum(a @ b))
        wall = time.time() - start
    else:
        size = min(size, 48)
        start = time.time()
        checksum = 0.0
        for _ in range(repeats):
            a = [[random.random() for _ in range(size)] for _ in range(size)]
            b = [[random.random() for _ in range(size)] for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    checksum += sum(a[i][k] * b[k][j] for k in range(size))
        wall = time.time() - start
    estimated = min(100, max(15, size / 2 + repeats * 10))
    return {
        "size": size,
        "repeats": repeats,
        "checksum": round(checksum, 5),
        "estimated_utilization": estimated,
        "wall_time": round(wall, 4),
    }


def run_vector_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    length = int(payload.get("length", 75000))
    repeats = max(1, int(payload.get("repeats", 3)))
    if np is not None:
        start = time.time()
        dot_total = 0.0
        for _ in range(repeats):
            vec_a = np.random.rand(length)
            vec_b = np.random.rand(length)
            dot_total += float(np.dot(vec_a, vec_b))
        wall = time.time() - start
    else:
        length = min(length, 20000)
        start = time.time()
        dot_total = 0.0
        for _ in range(repeats):
            vec_a = [random.random() for _ in range(length)]
            vec_b = [random.random() for _ in range(length)]
            dot_total += sum(a * b for a, b in zip(vec_a, vec_b))
        wall = time.time() - start
    estimated = min(95, max(12, length / 1200 + repeats * 6))
    return {
        "length": length,
        "repeats": repeats,
        "dot": round(dot_total, 4),
        "estimated_utilization": estimated,
        "wall_time": round(wall, 4),
    }


def run_noise_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    iterations = int(payload.get("iterations", 40000))
    iterations = max(5000, min(iterations, 100000))
    start = time.time()
    acc = 0.0
    for _ in range(iterations):
        acc += random.random() * random.random()
    wall = time.time() - start
    estimated = min(70, max(8, iterations / 1600))
    return {
        "iterations": iterations,
        "accumulator": round(acc, 4),
        "estimated_utilization": estimated,
        "wall_time": round(wall, 4),
    }


def run_onnx_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate an ONNX inference benchmark using available libraries."""
    input_size = int(payload.get("input_size", 512))
    repeats = max(1, min(int(payload.get("repeats", 10)), 20))
    start = time.time()

    if torch is not None:
        model = torch.nn.Sequential(
            torch.nn.Linear(input_size, input_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size // 2, input_size // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(input_size // 4, 10),
        )
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(repeats, input_size)
            output = model(dummy)
        checksum = float(output.sum().item())
    elif np is not None:
        checksum = 0.0
        weight1 = np.random.rand(input_size, input_size // 2)
        weight2 = np.random.rand(input_size // 2, input_size // 4)
        weight3 = np.random.rand(input_size // 4, 10)
        for _ in range(repeats):
            sample = np.random.rand(input_size)
            layer1 = np.maximum(sample @ weight1, 0)
            layer2 = np.maximum(layer1 @ weight2, 0)
            out = layer2 @ weight3
            checksum += float(out.sum())
    else:
        checksum = 0.0
        for _ in range(repeats):
            checksum += sum(random.random() for _ in range(input_size))

    wall = time.time() - start
    estimated = min(85, max(10, input_size / 20 + repeats))
    return {
        "input_size": input_size,
        "repeats": repeats,
        "checksum": round(checksum, 4),
        "estimated_utilization": estimated,
        "wall_time": round(wall, 4),
    }


if __name__ == "__main__":
    log(f"Node ID: {NODE_ID}")
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    poll_tasks_loop()

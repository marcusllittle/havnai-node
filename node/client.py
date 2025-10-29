import os
import platform
import random
import socket
import subprocess
import threading
import time
from typing import Any, Dict

import requests

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

SERVER_BASE = os.environ.get("HAVNAI_SERVER", "http://192.168.4.74:5001")
REGISTER_ENDPOINT = f"{SERVER_BASE}/register"
TASKS_ENDPOINT = f"{SERVER_BASE}/tasks"
RESULTS_ENDPOINT = f"{SERVER_BASE}/results"

NODE_ID = socket.gethostname()
SESSION = requests.Session()

START_TIME = time.time()
HEARTBEAT_INTERVAL = 30
TASK_POLL_INTERVAL = 10
CURRENT_UTILIZATION = random.randint(5, 15)
TASKS_COMPLETED = 0
LOCK = threading.Lock()


def get_gpu_info() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]
        )
        name, memory_total, utilization = output.decode().strip().split("\n")[0].split(", ")
        return {
            "gpu_name": name,
            "memory_total": int(memory_total),
            "utilization": int(utilization),
        }
    except Exception:
        return {"gpu_name": "Simulated", "memory_total": 0, "utilization": CURRENT_UTILIZATION}


def heartbeat_loop() -> None:
    global CURRENT_UTILIZATION
    while True:
        gpu = get_gpu_info()
        # if no real GPU metrics, blend simulated utilization
        if gpu.get("utilization", 0) == 0:
            gpu["utilization"] = CURRENT_UTILIZATION

        payload = {
            "node_id": NODE_ID,
            "os": platform.system(),
            "gpu": gpu,
            "start_time": START_TIME,
            "uptime": time.time() - START_TIME,
            "utilization": gpu.get("utilization", 0),
        }
        try:
            resp = SESSION.post(REGISTER_ENDPOINT, json=payload, timeout=5)
            resp.raise_for_status()
            print(f"✅ heartbeat sent: {resp.status_code}")
        except Exception as exc:  # pragma: no cover
            print("❌ heartbeat failed:", exc)
        time.sleep(HEARTBEAT_INTERVAL)


def execute_task(task: Dict[str, Any]) -> None:
    global CURRENT_UTILIZATION, TASKS_COMPLETED

    task_id = task.get("task_id")
    task_type = task.get("type")
    payload = task.get("payload", {})
    started_at = time.time()
    status = "success"
    metrics: Dict[str, Any] = {"task_type": task_type}

    try:
        if task_type == "matrix_multiply":
            metrics.update(run_matrix_workload(payload))
        elif task_type == "vector_dot":
            metrics.update(run_vector_workload(payload))
        elif task_type == "noise_simulation":
            metrics.update(run_noise_workload(payload))
        else:
            status = "skipped"
            metrics["message"] = "Unknown task type"
    except Exception as exc:  # pragma: no cover
        status = "failed"
        metrics["error"] = str(exc)

    finished_at = time.time()
    metrics["duration"] = round(finished_at - started_at, 4)

    # refresh simulated utilization based on workload
    estimated_util = int(metrics.get("estimated_utilization", CURRENT_UTILIZATION))
    with LOCK:
        CURRENT_UTILIZATION = max(5, min(100, estimated_util))
        if status == "success":
            TASKS_COMPLETED += 1

    result_payload = {
        "node_id": NODE_ID,
        "task_id": task_id,
        "status": status,
        "metrics": metrics,
        "utilization": CURRENT_UTILIZATION,
        "submitted_at": finished_at,
    }

    try:
        resp = SESSION.post(RESULTS_ENDPOINT, json=result_payload, timeout=8)
        resp.raise_for_status()
        print(f"📦 reported result for {task_id[:8]} ({status})")
    except Exception as exc:  # pragma: no cover
        print("❌ failed to submit result:", exc)


def poll_tasks() -> None:
    while True:
        try:
            resp = SESSION.get(TASKS_ENDPOINT, params={"node_id": NODE_ID}, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            for task in payload.get("tasks", []):
                print(f"🧮 executing task {task.get('task_id', '')[:8]} ({task.get('type')})")
                execute_task(task)
        except Exception as exc:  # pragma: no cover
            print("❌ task polling failed:", exc)
        time.sleep(TASK_POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------

def run_matrix_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    size = int(payload.get("size", 128))
    repeats = int(payload.get("repeats", 1))
    repeats = max(1, min(repeats, 5))
    start = time.time()

    if np is not None:
        checksum = 0.0
        for _ in range(repeats):
            a = np.random.rand(size, size)
            b = np.random.rand(size, size)
            checksum += float(np.sum(a @ b))
    else:
        # lightweight pure-Python fallback
        size = min(size, 64)
        checksum = 0.0
        for _ in range(repeats):
            a = [[random.random() for _ in range(size)] for _ in range(size)]
            b = [[random.random() for _ in range(size)] for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    checksum += sum(a[i][k] * b[k][j] for k in range(size))

    duration = time.time() - start
    estimated_util = min(100, max(10, size / 2 + repeats * 8))
    return {
        "size": size,
        "repeats": repeats,
        "checksum": round(checksum, 4),
        "estimated_utilization": estimated_util,
        "workload": "matrix_multiply",
        "wall_time": round(duration, 4),
    }


def run_vector_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    length = int(payload.get("length", 50000))
    repeats = int(payload.get("repeats", 3))
    start = time.time()

    if np is not None:
        dot_total = 0.0
        for _ in range(repeats):
            vec_a = np.random.rand(length)
            vec_b = np.random.rand(length)
            dot_total += float(np.dot(vec_a, vec_b))
    else:
        length = min(length, 20000)
        dot_total = 0.0
        for _ in range(repeats):
            vec_a = [random.random() for _ in range(length)]
            vec_b = [random.random() for _ in range(length)]
            dot_total += sum(a * b for a, b in zip(vec_a, vec_b))

    duration = time.time() - start
    estimated_util = min(95, max(8, (length / 1000) + repeats * 4))
    return {
        "length": length,
        "repeats": repeats,
        "dot": round(dot_total, 4),
        "estimated_utilization": estimated_util,
        "workload": "vector_dot",
        "wall_time": round(duration, 4),
    }


def run_noise_workload(payload: Dict[str, Any]) -> Dict[str, Any]:
    iterations = int(payload.get("iterations", 40000))
    iterations = max(5000, min(iterations, 80000))
    start = time.time()
    accumulator = 0.0
    for i in range(iterations):
        accumulator += random.random() * random.random()
    duration = time.time() - start
    estimated_util = min(70, max(5, iterations / 1200))
    return {
        "iterations": iterations,
        "accumulator": round(accumulator, 4),
        "estimated_utilization": estimated_util,
        "workload": "noise_simulation",
        "wall_time": round(duration, 4),
    }


if __name__ == "__main__":
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    poll_tasks()

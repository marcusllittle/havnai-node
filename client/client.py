"""HavnAI Node Client – Stage 4 AI Inference Worker"""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, List

import requests

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import onnxruntime as ort  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required for Stage 4 workloads") from exc

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
TASKS_ENDPOINT = f"{SERVER_BASE}/tasks/ai"
RESULTS_ENDPOINT = f"{SERVER_BASE}/results"

NODE_ID = socket.gethostname()
SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})

HEARTBEAT_INTERVAL = 30
TASK_POLL_INTERVAL = 15
BACKOFF_BASE = 5
MAX_BACKOFF = 60
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

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
    global utilization_hint
    output = run_command([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if output:
        try:
            name, mem_total, mem_used, util = output.strip().split("\n")[0].split(", ")
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
            resp = SESSION.get(TASKS_ENDPOINT, params={"node_id": NODE_ID}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            tasks = data.get("tasks", [])
            if tasks:
                log(f"Received {len(tasks)} AI task(s)", "📥")
            for task in tasks:
                execute_task(task)
            backoff = BACKOFF_BASE
        except Exception as exc:
            log(f"Task poll failed: {exc}", "⚠️")
            time.sleep(backoff)
            backoff = min(MAX_BACKOFF, backoff * 2)
        else:
            time.sleep(TASK_POLL_INTERVAL)


def ensure_model(model_name: str, model_url: str) -> Path:
    target = MODELS_DIR / Path(model_name).with_suffix(".onnx")
    if target.exists():
        return target
    url = model_url
    if url.startswith("/"):
        url = f"{SERVER_BASE}{url}"
    log(f"Downloading model {model_name} from {url}", "⬇️")
    resp = SESSION.get(url, stream=True, timeout=20)
    resp.raise_for_status()
    with target.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return target


def random_input(shape: List[int]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is required for generating input tensors")
    resolved = [max(1, dim) for dim in shape]
    return (np.random.rand(*resolved).astype(np.float32) * 2.0) - 1.0


def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    model_name = task.get("model_name", "model")
    model_url = task.get("model_url")
    input_shape = task.get("input_shape", [1, 64])
    reward_weight = task.get("reward_weight", 1.0)
    log(f"Executing AI task {task_id[:8]} · {model_name}", "🚀")

    try:
        model_path = ensure_model(model_name, model_url)
    except Exception as exc:
        log(f"Model download failed: {exc}", "🚫")
        report_result(task_id, "failed", {"error": str(exc)}, utilization_hint)
        return

    try:
        ort_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = ort_session.get_inputs()[0].name
    except Exception as exc:
        log(f"Failed to initialize inference session: {exc}", "🚫")
        report_result(task_id, "failed", {"error": str(exc)}, utilization_hint)
        return

    if np is None:
        log("NumPy missing: cannot generate input.", "🚫")
        report_result(task_id, "failed", {"error": "numpy missing"}, utilization_hint)
        return

    start_stats = read_gpu_stats()
    payload = random_input(input_shape)
    started = time.time()
    try:
        ort_session.run(None, {input_name: payload})
        status = "success"
    except Exception as exc:
        status = "failed"
        log(f"Inference error: {exc}", "🚫")
        payload = None
    duration = time.time() - started
    end_stats = read_gpu_stats()

    avg_util = int(max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint))
    with lock:
        utilization_hint = avg_util

    metrics = {
        "model_name": model_name,
        "model_path": str(model_path),
        "input_shape": input_shape,
        "reward_weight": reward_weight,
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
    }
    report_result(task_id, status, metrics, avg_util)


def report_result(task_id: str, status: str, metrics: Dict[str, Any], utilization: int) -> None:
    payload = {
        "node_id": NODE_ID,
        "task_id": task_id,
        "status": status,
        "metrics": metrics,
        "utilization": utilization,
        "submitted_at": time.time(),
    }
    try:
        resp = SESSION.post(RESULTS_ENDPOINT, data=json.dumps(payload), timeout=10)
        resp.raise_for_status()
        reward = resp.json().get("reward")
        prefix = "✅" if status == "success" else "⚠️"
        msg = f"Task {task_id[:8]} {status.upper()} · reward {reward} HAI"
        log(msg, prefix)
    except Exception as exc:
        log(f"Failed to submit results: {exc}", "🚫")


if __name__ == "__main__":
    log(f"Node ID: {NODE_ID}")
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    poll_tasks_loop()

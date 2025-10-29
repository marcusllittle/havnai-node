"""HavnAI Node Client – Stage 5A Creator Mode support."""

from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None

try:
    import onnxruntime as ort  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("onnxruntime is required for Stage 4/5A workloads") from exc

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    import diffusers  # type: ignore  # noqa: F401
except ImportError:  # pragma: no cover
    diffusers = None

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

SERVER_BASE = os.environ.get("HAVNAI_SERVER", "http://127.0.0.1:5001")
REGISTER_ENDPOINT = f"{SERVER_BASE}/register"
TASKS_ENDPOINT = f"{SERVER_BASE}/tasks/ai"
RESULTS_ENDPOINT = f"{SERVER_BASE}/results"
MODELS_LIST_ENDPOINT = f"{SERVER_BASE}/models/list"

NODE_ID = socket.gethostname()
SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})

HEARTBEAT_INTERVAL = 30
TASK_POLL_INTERVAL = 15
BACKOFF_BASE = 5
MAX_BACKOFF = 60

START_TIME = time.time()
ROLE = "creator" if os.environ.get("CREATOR_MODE") in {"1", "true", "TRUE"} else "worker"
utilization_hint = random.randint(5, 25 if ROLE == "creator" else 15)
lock = threading.Lock()

# Stage 5A additions: store creator model packs under ~/.havnai/models
LOCAL_MODEL_DIR = Path(os.environ.get("HAVNAI_MODEL_CACHE", Path.home() / ".havnai" / "models"))
LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging & helpers
# ---------------------------------------------------------------------------

def log(msg: str, prefix: str = "ℹ️") -> None:
    print(f"{prefix} [{time.strftime('%H:%M:%S')}] {msg}")


def run_command(cmd: List[str]) -> Optional[str]:
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
            "role": ROLE,
        }
        try:
            resp = SESSION.post(REGISTER_ENDPOINT, data=json.dumps(payload), timeout=5)
            resp.raise_for_status()
            backoff = BACKOFF_BASE
            log(f"Heartbeat OK ({ROLE})", "✅")
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


# ---------------------------------------------------------------------------
# Model download helpers
# ---------------------------------------------------------------------------

def ensure_creator_models_catalog() -> None:
    try:
        resp = SESSION.get(MODELS_LIST_ENDPOINT, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("creator_models", []):
            path = LOCAL_MODEL_DIR / item["filename"]
            item["local_path"] = path
    except Exception:
        pass


def ensure_model(model_name: str, model_url: str, filename_hint: Optional[str] = None) -> Path:
    if not model_url:
        raise RuntimeError("Missing model URL")
    parsed = Path(filename_hint or Path(model_url).name)
    target = LOCAL_MODEL_DIR / parsed.name
    if target.exists():
        return target
    url = model_url
    if url.startswith("/"):
        url = f"{SERVER_BASE}{url}"
    log(f"Downloading model {model_name} from {url}", "⬇️")
    resp = SESSION.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    with target.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return target


def random_input(shape: List[int]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy required for inference")
    dims = [max(1, dim) for dim in shape]
    return (np.random.rand(*dims).astype(np.float32) * 2.0) - 1.0


# ---------------------------------------------------------------------------
# Task execution
# ---------------------------------------------------------------------------

def execute_task(task: Dict[str, Any]) -> None:
    global utilization_hint

    task_id = task.get("task_id", "unknown")
    task_type = task.get("type", "ai")
    model_name = task.get("model_name", "model")
    model_url = task.get("model_url")
    reward_weight = task.get("reward_weight", 1.0)
    input_shape = task.get("input_shape", [1, 64])

    if task_type == "image_gen" and ROLE != "creator":
        log(f"Skipping heavy task {task_id[:8]} – not a creator node", "⚠️")
        return

    log(f"Executing {task_type} task {task_id[:8]} · {model_name}", "🚀")

    if task_type == "image_gen":
        metrics, util = run_image_generation(model_name, model_url, reward_weight)
    else:
        metrics, util = run_ai_inference(model_name, model_url, input_shape, reward_weight)

    with lock:
        utilization_hint = util

    result = {
        "node_id": NODE_ID,
        "task_id": task_id,
        "status": metrics.pop("status", "success"),
        "metrics": metrics,
        "utilization": utilization_hint,
        "submitted_at": time.time(),
    }

    try:
        resp = SESSION.post(RESULTS_ENDPOINT, data=json.dumps(result), timeout=10)
        resp.raise_for_status()
        reward = resp.json().get("reward")
        prefix = "✅" if result["status"] == "success" else "⚠️"
        log(f"Task {task_id[:8]} {result['status'].upper()} · reward {reward} HAI", prefix)
    except Exception as exc:
        log(f"Failed to submit result: {exc}", "🚫")


def run_ai_inference(model_name: str, model_url: str, input_shape: List[int], reward_weight: float) -> (Dict[str, Any], int):
    try:
        model_path = ensure_model(model_name, model_url)
    except Exception as exc:
        return ({"status": "failed", "error": str(exc)}, utilization_hint)

    try:
        ort_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_name = ort_session.get_inputs()[0].name
    except Exception as exc:
        return ({"status": "failed", "error": f"session init: {exc}"}, utilization_hint)

    if np is None:
        return ({"status": "failed", "error": "numpy missing"}, utilization_hint)

    tensor = random_input(input_shape)
    start_stats = read_gpu_stats()
    started = time.time()
    status = "success"
    try:
        ort_session.run(None, {input_name: tensor})
    except Exception as exc:
        status = "failed"
        error = str(exc)
    duration = time.time() - started
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)
    metrics = {
        "status": status,
        "model_name": model_name,
        "model_path": str(model_path),
        "input_shape": input_shape,
        "reward_weight": reward_weight,
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
    }
    if status == "failed":
        metrics["error"] = locals().get("error", "inference error")
    return metrics, int(util)


def run_image_generation(model_name: str, model_url: str, reward_weight: float) -> (Dict[str, Any], int):
    try:
        model_path = ensure_model(model_name, model_url, filename_hint=model_name + Path(model_url).suffix)
    except Exception as exc:
        return ({"status": "failed", "error": str(exc)}, utilization_hint)

    start_stats = read_gpu_stats()
    started = time.time()

    # Stage 5A additions: simulate heavy workload (diffusers optional)
    try:
        if torch is not None and diffusers is not None:
            # Lightweight stub: allocate tensor and run a dummy conv
            latent = torch.randn((1, 4, 64, 64))
            kernel = torch.randn((4, 4, 3, 3))
            torch.nn.functional.conv2d(latent, kernel, padding=1)
        else:
            time.sleep(random.uniform(1.0, 2.0))
        status = "success"
    except Exception as exc:
        status = "failed"
        error = str(exc)

    duration = time.time() - started
    end_stats = read_gpu_stats()
    util = max(start_stats.get("utilization", 0), end_stats.get("utilization", 0), utilization_hint)
    util = int(max(util, 65 if ROLE == "creator" else util))
    metrics = {
        "status": status,
        "model_name": model_name,
        "model_path": str(model_path),
        "reward_weight": reward_weight,
        "task_type": "image_gen",
        "inference_time_ms": round(duration * 1000, 3),
        "gpu_util_start": start_stats.get("utilization", 0),
        "gpu_util_end": end_stats.get("utilization", 0),
    }
    if status == "failed":
        metrics["error"] = locals().get("error", "image generation error")
    return metrics, util


if __name__ == "__main__":
    ensure_creator_models_catalog()
    log(f"Node ID: {NODE_ID} · Role: {ROLE.upper()}" )
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    poll_tasks_loop()

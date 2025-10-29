import time, json, requests, platform, socket, subprocess

SERVER_URL = "http://192.168.4.74:5001/register"
NODE_ID = socket.gethostname()

def get_gpu_info():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"]
        ).decode().strip().split("\n")[0].split(", ")
        return {
            "gpu_name": output[0],
            "memory_total": int(output[1]),
            "utilization": int(output[2])
        }
    except Exception as e:
        return {"gpu_name": "None", "memory_total": 0, "utilization": 0, "error": str(e)}

def heartbeat():
    while True:
        gpu = get_gpu_info()
        payload = {
            "node_id": NODE_ID,
            "os": platform.system(),
            "gpu": gpu,
            "uptime": time.time(),
        }
        try:
            r = requests.post(SERVER_URL, json=payload, timeout=5)
            print("✅ heartbeat sent:", r.status_code, r.text)
        except Exception as e:
            print("❌ heartbeat failed:", e)
        time.sleep(30)  # every 30 s

if __name__ == "__main__":
    heartbeat()

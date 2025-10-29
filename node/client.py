import sys
from pathlib import Path

CLIENT_PATH = Path(__file__).resolve().parents[1] / "client" / "client.py"

if __name__ == "__main__":
    if not CLIENT_PATH.exists():
        print("Client script not found:", CLIENT_PATH)
        sys.exit(1)
    with CLIENT_PATH.open() as f:
        code = compile(f.read(), str(CLIENT_PATH), "exec")
        exec(code, {"__name__": "__main__"})

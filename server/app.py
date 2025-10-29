from flask import Flask, request, jsonify
from datetime import datetime
import json, os

app = Flask(__name__)
REGISTRY_FILE = "nodes.json"

def load_nodes():
    if os.path.exists(REGISTRY_FILE):
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {}

def save_nodes(nodes):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(nodes, f, indent=2)

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    nodes = load_nodes()
    node_id = data["node_id"]
    nodes[node_id] = {
        "os": data["os"],
        "gpu": data["gpu"],
        "last_seen": datetime.utcnow().isoformat()
    }
    save_nodes(nodes)
    return jsonify({"status": "ok", "node": node_id}), 200

@app.route("/nodes", methods=["GET"])
def get_nodes():
    return jsonify(load_nodes()), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

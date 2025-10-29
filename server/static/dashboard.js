const NODE_ENDPOINT = "/nodes";
const REWARD_ENDPOINT = "/rewards";
const LOG_ENDPOINT = "/logs";
const POLL_INTERVAL = 10000;

const nodeTable = document.getElementById("nodeTable");
const statusEl = document.getElementById("status");
const summaryTotal = document.getElementById("summaryTotal");
const summaryOnline = document.getElementById("summaryOnline");
const summaryUtil = document.getElementById("summaryUtil");
const summaryRewards = document.getElementById("summaryRewards");
const summaryBacklog = document.getElementById("summaryBacklog");
const summarySync = document.getElementById("summarySync");
const logList = document.getElementById("logList");

function setStatus(message = "", tone = "info") {
  statusEl.textContent = message;
  statusEl.style.color = tone === "error" ? "var(--error)" : "var(--text-muted)";
}

function renderNodes(nodes, rewardsMap) {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    nodeTable.innerHTML = `
      <tr>
        <td colspan="7" style="text-align:center; padding: 1.6rem; color: var(--text-muted);">
          No nodes connected yet.
        </td>
      </tr>`;
    return;
  }

  const rows = nodes
    .slice()
    .sort((a, b) => a.node_id.localeCompare(b.node_id))
    .map((node) => {
      const reward = rewardsMap[node.node_id] ?? node.rewards ?? 0;
      const model = node.model_name || "—";
      const inference = node.inference_time_ms != null ? Number(node.inference_time_ms).toFixed(2) : "—";
      const util = Number(node.gpu_utilization ?? 0).toFixed(0);
      const rewardFmt = Number(reward).toFixed(6);
      const lastSeen = node.last_seen ? new Date(node.last_seen).toLocaleTimeString() : "—";
      return `
        <tr>
          <td><span class="badge">${node.node_id}</span></td>
          <td>${model}</td>
          <td>${inference}</td>
          <td class="util">${util}%</td>
          <td class="reward">${rewardFmt}</td>
          <td>${lastSeen}</td>
          <td>${node.uptime_human || "—"}</td>
        </tr>`;
    })
    .join("");

  nodeTable.innerHTML = rows;
}

function renderSummary(summary, rewardTotal) {
  summary = summary || {};
  summaryTotal.textContent = summary.total_nodes ?? 0;
  summaryOnline.textContent = summary.online_nodes ?? 0;
  const util = Number(summary.avg_utilization ?? 0).toFixed(1);
  summaryUtil.textContent = `${util}%`;
  summaryRewards.textContent = `${Number(rewardTotal ?? 0).toFixed(6)} HAI`;
  summaryBacklog.textContent = summary.tasks_backlog ?? 0;
  summarySync.textContent = new Date().toLocaleTimeString();
}

function renderLogs(logs) {
  if (!Array.isArray(logs) || logs.length === 0) {
    logList.innerHTML = '<li><span class="time">—</span>No recent events.</li>';
    return;
  }
  const items = logs
    .slice(-8)
    .reverse()
    .map((entry) => {
      const time = entry.timestamp ? new Date(entry.timestamp).toLocaleTimeString() : "";
      return `<li><span class="time">${time}</span>${entry.message}</li>`;
    })
    .join("");
  logList.innerHTML = items;
}

async function fetchTelemetry() {
  try {
    const [nodesRes, rewardsRes, logsRes] = await Promise.all([
      fetch(NODE_ENDPOINT),
      fetch(REWARD_ENDPOINT),
      fetch(LOG_ENDPOINT),
    ]);

    if (!nodesRes.ok) throw new Error(`Nodes request failed: ${nodesRes.status}`);
    if (!rewardsRes.ok) throw new Error(`Rewards request failed: ${rewardsRes.status}`);
    if (!logsRes.ok) throw new Error(`Logs request failed: ${logsRes.status}`);

    const nodesJson = await nodesRes.json();
    const rewardsJson = await rewardsRes.json();
    const logsJson = await logsRes.json();

    renderNodes(nodesJson.nodes ?? [], rewardsJson.rewards ?? {});
    renderSummary(nodesJson.summary, rewardsJson.total);
    renderLogs(logsJson.logs ?? []);
    setStatus(`Sync successful. Next update in ${POLL_INTERVAL / 1000}s.`);
  } catch (error) {
    console.error("Dashboard refresh failed", error);
    setStatus(error.message || "Failed to fetch telemetry.", "error");
  }
}

fetchTelemetry();
setInterval(fetchTelemetry, POLL_INTERVAL);

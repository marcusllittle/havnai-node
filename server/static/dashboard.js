const NODE_ENDPOINT = "/nodes";
const REWARD_ENDPOINT = "/rewards";
const LOG_ENDPOINT = "/logs";
const POLL_INTERVAL = 10000;

const nodeTable = document.getElementById("nodeTable");
const statusEl = document.getElementById("status");
const summaryTotal = document.getElementById("summaryTotal");
const summaryOnline = document.getElementById("summaryOnline");
const summaryOffline = document.getElementById("summaryOffline");
const summaryUtil = document.getElementById("summaryUtil");
const summaryRewards = document.getElementById("summaryRewards");
const summaryBacklog = document.getElementById("summaryBacklog");
const logList = document.getElementById("logList");

function setStatus(message = "", tone = "info") {
  statusEl.textContent = message;
  statusEl.style.color = tone === "error" ? "var(--error)" : "var(--text-muted)";
}

function renderUtilBar(utilization) {
  const percent = Math.min(100, Math.max(0, Number(utilization) || 0));
  return `
    <div class="util-bar">
      <div class="fill" style="width: ${percent}%;"></div>
      <span>${percent.toFixed(0)}%</span>
    </div>`;
}

function renderTaskChip(node) {
  const current = node.current_task;
  const last = node.last_result || {};

  if (current && current.task_id) {
    return `<span class="task-chip running">Running · ${current.type || current.task_id.slice(0, 6)}</span>`;
  }

  if (last && last.status) {
    const status = String(last.status).toLowerCase();
    const cls = {
      success: "task-chip idle",
      skipped: "task-chip skipped",
      failed: "task-chip failed",
    }[status] || "task-chip idle";
    const label = last.type ? last.type : (last.task_id ? last.task_id.slice(0, 6) : "Task");
    const statusText = status.toUpperCase();
    return `<span class="${cls}">${statusText} · ${label}</span>`;
  }

  return '<span class="task-chip idle">Idle</span>';
}

function renderNodes(nodes, rewardsMap) {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    nodeTable.innerHTML = `
      <tr>
        <td colspan="8" style="text-align:center; padding: 1.8rem; color: var(--text-muted);">
          No nodes connected yet.
        </td>
      </tr>`;
    return;
  }

  const rows = nodes
    .slice()
    .sort((a, b) => a.node_id.localeCompare(b.node_id))
    .map((node) => {
      const gpu = (node.gpu && typeof node.gpu === "object") ? node.gpu : {};
      const reward = rewardsMap[node.node_id] ?? node.rewards ?? 0;
      const gpuLabel = gpu.gpu_name || gpu.name || "—";
      return `
        <tr>
          <td><span class="badge">${node.node_id}</span></td>
          <td>${gpuLabel}</td>
          <td>${renderUtilBar(node.avg_utilization ?? node.utilization)}</td>
          <td>${Number(reward).toFixed(6)}</td>
          <td>${renderTaskChip(node)}</td>
          <td>${node.uptime_human || "—"}</td>
          <td>${node.last_seen ? new Date(node.last_seen).toLocaleTimeString() : "—"}</td>
          <td>${renderResultTooltip(node.last_result)}</td>
        </tr>`;
    })
    .join("");

  nodeTable.innerHTML = rows;
}

function renderResultTooltip(result) {
  if (!result || !result.status) {
    return '<span class="task-chip idle">Awaiting</span>';
  }
  const status = String(result.status).toLowerCase();
  const cls = {
    success: "task-chip idle",
    failed: "task-chip failed",
    skipped: "task-chip skipped",
  }[status] || "task-chip idle";
  const label = result.type || (result.task_id ? result.task_id.slice(0, 6) : "Result");
  return `<span class="${cls}">${status.toUpperCase()} · ${label}</span>`;
}

function renderSummary(summary, rewardTotal) {
  summary = summary || {};
  summaryTotal.textContent = summary.total_nodes ?? 0;
  summaryOnline.textContent = summary.online_nodes ?? 0;
  summaryOffline.textContent = summary.offline_nodes ?? 0;
  const util = Number(summary.avg_utilization ?? 0).toFixed(1);
  summaryUtil.textContent = `${util}%`;
  summaryRewards.textContent = `${Number(rewardTotal ?? 0).toFixed(6)} HAI`;
  summaryBacklog.textContent = summary.tasks_backlog ?? 0;
}

function renderLogs(logs) {
  if (!Array.isArray(logs) || logs.length === 0) {
    logList.innerHTML = '<li>No recent events.</li>';
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
      fetch(LOG_ENDPOINT)
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

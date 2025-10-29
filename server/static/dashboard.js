const NODE_ENDPOINT = "/nodes";
const REWARD_ENDPOINT = "/rewards";
const LOG_ENDPOINT = "/logs";
const POLL_INTERVAL = 10000;

const nodeTable = document.getElementById("nodeTable");
const statusEl = document.getElementById("status");
const summaryTotal = document.getElementById("summaryTotal");
const summaryOnline = document.getElementById("summaryOnline");
const summaryUtil = document.getElementById("summaryUtil");
const summaryActive = document.getElementById("summaryActive");
const summaryRewards = document.getElementById("summaryRewards");
const summaryBacklog = document.getElementById("summaryBacklog");
const summarySync = document.getElementById("summarySync");
const logList = document.getElementById("logList");
const jobFeedBody = document.getElementById("jobFeed");

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

function renderSummary(nodeSummary, jobSummary) {
  nodeSummary = nodeSummary || {};
  jobSummary = jobSummary || {};
  summaryTotal.textContent = nodeSummary.total_nodes ?? 0;
  summaryOnline.textContent = nodeSummary.online_nodes ?? 0;
  const util = Number(nodeSummary.avg_utilization ?? 0).toFixed(1);
  summaryUtil.textContent = `${util}%`;
  summaryActive.textContent = jobSummary.active_jobs ?? 0;
  summaryBacklog.textContent = jobSummary.queued_jobs ?? nodeSummary.tasks_backlog ?? 0;
  summaryRewards.textContent = `${Number(jobSummary.total_distributed ?? 0).toFixed(6)} HAI`;
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

function renderJobFeed(feed) {
  if (!Array.isArray(feed) || feed.length === 0) {
    jobFeedBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align:center; padding: 1.4rem; color: var(--text-muted);">
          No public jobs submitted yet.
        </td>
      </tr>`;
    return;
  }

  const rows = feed.map((item) => {
    const reward = Number(item.reward ?? 0).toFixed(6);
    const completed = item.completed_at ? new Date(item.completed_at).toLocaleTimeString() : "—";
    const status = (item.status || "—").toUpperCase();
    return `
      <tr>
        <td><span class="badge">${item.job_id}</span></td>
        <td>${item.wallet}</td>
        <td>${item.model}</td>
        <td>${status}</td>
        <td class="reward">${reward}</td>
        <td>${completed}</td>
      </tr>`;
  }).join("");

  jobFeedBody.innerHTML = rows;
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
    renderSummary(nodesJson.summary, nodesJson.job_summary);
    renderJobFeed(nodesJson.job_summary?.feed ?? []);
    renderLogs(logsJson.logs ?? []);
    setStatus(`Sync successful. Next update in ${POLL_INTERVAL / 1000}s.`);
  } catch (error) {
    console.error("Dashboard refresh failed", error);
    setStatus(error.message || "Failed to fetch telemetry.", "error");
  }
}

fetchTelemetry();
setInterval(fetchTelemetry, POLL_INTERVAL);

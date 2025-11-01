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
const metricNodesOnline = document.getElementById("metricNodesOnline");
const metricJobsToday = document.getElementById("metricJobsToday");
const metricTotalHai = document.getElementById("metricTotalHai");
const logList = document.getElementById("logList");
const jobFeedBody = document.getElementById("jobFeed");
const modelCatalogBody = document.getElementById("modelCatalog");

function setStatus(message = "", tone = "info") {
  statusEl.textContent = message;
  statusEl.style.color = tone === "error" ? "var(--error)" : "var(--text-muted)";
}

function renderNodes(nodes, rewardsMap) {
  if (!Array.isArray(nodes) || nodes.length === 0) {
    nodeTable.innerHTML = `
      <tr>
        <td colspan="4" style="text-align:center; padding: 1.6rem; color: var(--text-muted);">
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
      const util = Number(node.avg_utilization ?? node.gpu_utilization ?? 0).toFixed(1);
      const rewardFmt = Number(reward).toFixed(6);
      const badgeClass = node.role === "creator" ? "badge creator" : "badge";
      const displayName = node.node_name || node.node_id;
      const creatorBadge = node.role === "creator" ? '<span class="badge creator">Creator</span>' : '<span class="badge">Worker</span>';
      const typeClass = "job-type-creator";
      return `
        <tr class="${typeClass}">
          <td><span class="${badgeClass}">${displayName}</span></td>
          <td>${creatorBadge}</td>
          <td class="util">${util}%</td>
          <td class="reward">${rewardFmt} HAI</td>
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
  if (metricNodesOnline) {
    metricNodesOnline.textContent = nodeSummary.online_nodes ?? 0;
  }
  if (metricJobsToday) {
    metricJobsToday.textContent = jobSummary.jobs_completed_today ?? 0;
  }
  if (metricTotalHai) {
    metricTotalHai.textContent = `${Number(jobSummary.total_distributed ?? 0).toFixed(2)} HAI`;
  }
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
        <td colspan="7" style="text-align:center; padding: 1.4rem; color: var(--text-muted);">
          No public jobs submitted yet.
        </td>
      </tr>`;
    return;
  }

  const rows = feed.map((item) => {
    const reward = Number(item.reward_hai ?? item.reward ?? 0).toFixed(6);
    const completed = item.completed_at ? new Date(item.completed_at).toLocaleTimeString() : "—";
    const status = (item.status || "QUEUED").toUpperCase();
    const statusMap = {
      QUEUED: "🟡 Queued",
      RUNNING: "🔵 Running",
      COMPLETED: "🟢 Completed",
      FAILED: "🔴 Failed",
    };
    const statusClass = status.toLowerCase();
    const statusLabel = statusMap[status] || status;
    const preview = item.image_url
      ? `<div class="job-card"><img src="${item.image_url}" alt="${item.job_id} preview" width="96" height="96" loading="lazy" /></div>`
      : "—";
    const rowClass = "job-type-creator";
    return `
      <tr class="${rowClass}">
        <td><span class="badge">${item.job_id}</span></td>
        <td>${item.model}</td>
        <td>${item.wallet}</td>
        <td><span class="status-badge ${statusClass}">${statusLabel}</span></td>
        <td class="reward">${reward}</td>
        <td>${preview}</td>
        <td>${completed}</td>
      </tr>`;
  }).join("");

  jobFeedBody.innerHTML = rows;
}

function renderModelCatalog(models) {
  if (!Array.isArray(models) || models.length === 0) {
    modelCatalogBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align:center; padding: 1.2rem; color: var(--text-muted);">
          No models registered yet.
        </td>
      </tr>`;
    return;
  }

  const rows = models.map((model) => {
    const nodes = Array.isArray(model.nodes) ? model.nodes.join(", ") : (model.nodes || "server");
    const tags = Array.isArray(model.tags) ? model.tags.join(", ") : (model.tags || "");
    const size = model.size ? `${(Number(model.size) / (1024 * 1024)).toFixed(2)} MB` : "—";
    return `
      <tr>
        <td>${model.model}</td>
        <td>${Number(model.weight ?? 1).toFixed(2)}</td>
        <td>${model.source || "unknown"}</td>
        <td>${nodes}</td>
        <td>${tags}</td>
        <td>${size}</td>
      </tr>`;
  }).join("");

  modelCatalogBody.innerHTML = rows;
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
    renderModelCatalog(nodesJson.models_catalog ?? []);
    renderLogs(logsJson.logs ?? []);
    setStatus(`Sync successful. Next update in ${POLL_INTERVAL / 1000}s.`);
  } catch (error) {
    console.error("Dashboard refresh failed", error);
    setStatus(error.message || "Failed to fetch telemetry.", "error");
  }
}

fetchTelemetry();
setInterval(fetchTelemetry, POLL_INTERVAL);

/* ── Queue Monitor Frontend ──────────────────────────────────────────── */

const MAX_CARDS = 6;
let eventCount = 0;
let popupTimer = null;

// ── DOM refs ────────────────────────────────────────────────────────────
const $clock      = document.getElementById("clock");
const $lane1      = document.getElementById("m-lane1");
const $lane2      = document.getElementById("m-lane2");
const $store      = document.getElementById("m-store");
const $status     = document.getElementById("m-status");
const $statusDot  = document.getElementById("status-dot");
const $cards      = document.getElementById("cards");
const $popup      = document.getElementById("popup");
const $eventCount = document.getElementById("event-count");

// ── Clock ───────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  $clock.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
}
updateClock();
setInterval(updateClock, 1000);

// ── Metrics polling ─────────────────────────────────────────────────────
async function pollMetrics() {
  try {
    const res = await fetch("/metrics", { cache: "no-store" });
    if (!res.ok) return;
    const m = await res.json();

    $lane1.textContent  = `${m.queue1 ?? 0} people`;
    $lane2.textContent  = `${m.queue2 ?? 0} people`;
    $store.textContent  = m.store_count ?? 0;
    $status.textContent = m.status || "MONITORING";

    const isAlert = m.status === "ALERT";
    $statusDot.classList.toggle("alert", isAlert);
    $status.style.color = isAlert ? "#fca5a5" : "";
  } catch (_) { /* retry next tick */ }
}
pollMetrics();
setInterval(pollMetrics, 1500);

// ── SSE: agent events ───────────────────────────────────────────────────
function connectSSE() {
  const source = new EventSource("/events");

  source.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }
    if (data.type !== "agent_decision") return;

    eventCount++;
    $eventCount.textContent = `${eventCount} event${eventCount !== 1 ? "s" : ""}`;

    addCard(data);

    if (data.urgency === "high") {
      showPopup(data);
    }
  };

  source.onerror = () => {
    source.close();
    setTimeout(connectSSE, 3000);
  };
}
connectSSE();

// ── Card rendering ──────────────────────────────────────────────────────
function addCard(data) {
  const card = document.createElement("div");
  const urg = (data.urgency || "low").toLowerCase();
  card.className = `card urgency-${urg}`;

  let fieldsHtml = "";

  if (data.situation) {
    fieldsHtml += field("Situation", data.situation);
  }
  if (data.reasoning) {
    fieldsHtml += field("Reasoning", data.reasoning);
  }
  if (data.action && data.action.toLowerCase() !== "none") {
    fieldsHtml += field("Action", data.action, "action-text");
  }
  if (data.tool_result) {
    fieldsHtml += field("Result", data.tool_result, "result-text");
  }

  card.innerHTML = `
    <div class="card-top">
      <span class="card-time">${esc(data.timestamp || "")}</span>
      <span class="urgency-badge ${urg}">${urg.toUpperCase()}</span>
    </div>
    ${fieldsHtml}
  `;

  // Insert at top
  $cards.prepend(card);

  // Trim excess cards
  const cards = $cards.children;
  while (cards.length > MAX_CARDS) {
    const last = cards[cards.length - 1];
    last.classList.add("fading");
    setTimeout(() => last.remove(), 400);
    // Don't remove immediately to allow fade animation, but break to avoid
    // removing multiple in one tick
    break;
  }
  // Hard cap after animation
  setTimeout(() => {
    while ($cards.children.length > MAX_CARDS + 1) {
      $cards.lastElementChild.remove();
    }
  }, 500);
}

function field(label, value, extraClass) {
  const cls = extraClass ? ` ${extraClass}` : "";
  return `
    <div class="card-field">
      <div class="card-field-label">${esc(label)}</div>
      <div class="card-field-value${cls}">${esc(value)}</div>
    </div>
  `;
}

// ── Popup overlay ───────────────────────────────────────────────────────
function showPopup(data) {
  clearTimeout(popupTimer);

  $popup.innerHTML = `
    <div class="popup-tag">High Priority Alert</div>
    <div class="popup-situation">${esc(data.situation || "")}</div>
    ${data.action && data.action.toLowerCase() !== "none"
      ? `<div class="popup-action">${esc(data.action)}</div>` : ""}
  `;

  $popup.classList.remove("popup-hidden", "popup-exit");
  $popup.classList.add("popup-enter");

  popupTimer = setTimeout(() => {
    $popup.classList.remove("popup-enter");
    $popup.classList.add("popup-exit");
    setTimeout(() => {
      $popup.classList.add("popup-hidden");
      $popup.classList.remove("popup-exit");
    }, 300);
  }, 4000);
}

// ── Utility ─────────────────────────────────────────────────────────────
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ── Queue Monitor Frontend ──────────────────────────────────────────── */

const MAX_CARDS = 6;
let eventCount = 0;
let popupTimer = null;
const pendingCvCards = new Map();

// ── Checkout state ────────────────────────────────────────────────────────
let checkoutCount = 2; // Base: checkout 1 & 2 always present
let extraCounts = { 3: 0, 4: 0 };

// ── DOM refs ────────────────────────────────────────────────────────────
const $clock      = document.getElementById("clock");
const $lane1      = document.getElementById("m-lane1");
const $lane2      = document.getElementById("m-lane2");
const $totalWait  = document.getElementById("m-total-wait");
const $store      = document.getElementById("m-store");
const $status     = document.getElementById("m-status");
const $statusDot  = document.getElementById("status-dot");
const $cards      = document.getElementById("cards");
const $popup      = document.getElementById("popup");
const $eventCount = document.getElementById("event-count");
const $extraCheckouts = document.getElementById("extra-checkouts");

// ── Clock ───────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  $clock.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
}
updateClock();
setInterval(updateClock, 1000);

// ── Render extra checkouts (3+) ─────────────────────────────────────────
function renderExtraCheckouts() {
  let html = "";
  for (let i = 3; i <= checkoutCount; i++) {
    const count = i === 3 ? (extraCounts[3] ?? 0) : (extraCounts[4] ?? 0);

    html += `
      <div class="metric-sep"></div>
      <div class="mblock" id="checkout-${i}">
        <div class="mblock-label">Checkout ${i}</div>
        <div class="mblock-main">${count} <span class="mblock-unit">waiting</span></div>
        <div class="mblock-sub">active</div>
      </div>
    `;
  }
  $extraCheckouts.innerHTML = html;
}

// ── Metrics polling ─────────────────────────────────────────────────────
let lastQueue1Wait = null;
let lastQueue2Wait = null;

async function pollMetrics() {
  try {
    const res = await fetch("/metrics", { cache: "no-store" });
    if (!res.ok) return;
    const m = await res.json();

    extraCounts = {
      3: m.queue3 ?? 0,
      4: m.queue4 ?? 0,
    };
    
    // Update checkout count if changed from LLM action
    if (m.checkouts_open !== checkoutCount) {
      checkoutCount = m.checkouts_open;
    }
    renderExtraCheckouts();
    
    // Update real queues
    $lane1.innerHTML  = `${m.queue1 ?? 0} <span class="mblock-unit">waiting</span>`;
    $lane2.innerHTML  = `${m.queue2 ?? 0} <span class="mblock-unit">waiting</span>`;
    
    // Update wait times
    const lane1Sub = document.getElementById("m-lane1-sub");
    const lane2Sub = document.getElementById("m-lane2-sub");
    
    if (m.queue1_avg_wait !== null) {
      lane1Sub.textContent = `avg wait ${m.queue1_avg_wait.toFixed(1)}s`;
      lastQueue1Wait = m.queue1_avg_wait;
    } else {
      lane1Sub.textContent = "avg wait —";
      lastQueue1Wait = null;
    }
    
    if (m.queue2_avg_wait !== null) {
      lane2Sub.textContent = `avg wait ${m.queue2_avg_wait.toFixed(1)}s`;
      lastQueue2Wait = m.queue2_avg_wait;
    } else {
      lane2Sub.textContent = "avg wait —";
      lastQueue2Wait = null;
    }
    
    // Calculate and display total average wait
    let totalWait = "—";
    if (lastQueue1Wait !== null && lastQueue2Wait !== null) {
      totalWait = ((lastQueue1Wait + lastQueue2Wait) / 2).toFixed(1);
    } else if (lastQueue1Wait !== null) {
      totalWait = lastQueue1Wait.toFixed(1);
    } else if (lastQueue2Wait !== null) {
      totalWait = lastQueue2Wait.toFixed(1);
    }
    $totalWait.innerHTML = `${totalWait} <span class="mblock-unit">seconds</span>`;
    
    $store.textContent  = m.store_count ?? m.clients_visible ?? 0;
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

    if (data.type === "queue_alert") {
      addCvAlertCard(data);
      eventCount++;
      $eventCount.textContent = `${eventCount} event${eventCount !== 1 ? "s" : ""}`;
      return;
    }

    if (data.type === "agent_decision") {
      updateCvAlertWithDecision(data);
      if ((data.urgency || "").toLowerCase() === "high") {
        showPopup(data);
      }
      return;
    }

    if (data.type === "scheduled_llm_call") {
      addScheduledCard(data);
      eventCount++;
      $eventCount.textContent = `${eventCount} event${eventCount !== 1 ? "s" : ""}`;
    }
  };

  source.onerror = () => {
    source.close();
    setTimeout(connectSSE, 3000);
  };
}
connectSSE();

// ── Typewriter ───────────────────────────────────────────────────────────
function typewriter(el, text, speed = 18) {
  el.textContent = "";
  let i = 0;
  function tick() {
    if (i < text.length) {
      el.textContent += text[i++];
      setTimeout(tick, speed);
    }
  }
  tick();
}

// ── Card rendering ──────────────────────────────────────────────────────
function addCvAlertCard(data) {
  const card = document.createElement("div");
  card.className = "card card-event-cv";
  const alertId = data.alert_id || `cv-${Date.now()}`;
  card.dataset.alertId = alertId;
  card.dataset.alertMessage = data.message || `Lane ${data.lane || "?"} spike detected`;

  card.innerHTML = `
    <div class="card-top">
      <span class="card-time">${esc(data.timestamp || "")}</span>
      <span class="event-badge event-badge-cv">CV ALERT</span>
    </div>
    <div class="card-field">
      <div class="card-field-label">Alert</div>
      <div class="card-field-value">${esc(card.dataset.alertMessage)}</div>
    </div>
    <div class="card-field">
      <div class="card-field-label">LLM Call</div>
      <div class="card-field-value card-pending">Waiting for decision...</div>
    </div>
  `;

  pendingCvCards.set(alertId, card);
  prependAndTrim(card);
}

function updateCvAlertWithDecision(data) {
  const alertId = data.alert_id;
  const card = alertId ? pendingCvCards.get(alertId) : null;
  const alertMessage = card ? card.dataset.alertMessage : null;

  if (!card) {
    // Fallback: if unmatched, still render as CV decision card.
    const fallback = document.createElement("div");
    fallback.className = "card card-event-cv";
    fallback.innerHTML = buildDecisionBody("CV ALERT", "event-badge-cv", data, null);
    prependAndTrim(fallback);
    startWhyTypewriter(fallback, data);
    return;
  }

  card.innerHTML = buildDecisionBody("CV ALERT", "event-badge-cv", data, alertMessage);
  pendingCvCards.delete(alertId);
  startWhyTypewriter(card, data);
}

function addScheduledCard(data) {
  const card = document.createElement("div");
  card.className = "card card-event-scheduled";
  card.innerHTML = buildDecisionBody("SCHEDULED", "event-badge-scheduled", data, null);
  prependAndTrim(card);
  startWhyTypewriter(card, data);
}

function startWhyTypewriter(card, data) {
  const whyEl = card.querySelector(".card-why-value");
  if (whyEl) {
    typewriter(whyEl, data.reasoning || data.situation || "No reasoning provided.");
  }
}

function buildDecisionBody(label, badgeClass, data, alertMessage) {
  const action = data.action && data.action.toLowerCase() !== "none" ? data.action : "none";
  const metrics = data.metrics || {};
  return `
    <div class="card-top">
      <span class="card-time">${esc(data.timestamp || "")}</span>
      <span class="event-badge ${badgeClass}">${label}</span>
    </div>
    ${alertMessage ? `
    <div class="card-field">
      <div class="card-field-label">Alert</div>
      <div class="card-field-value">${esc(alertMessage)}</div>
    </div>` : ""}
    <div class="card-field">
      <div class="card-field-label">Input Params</div>
      <div class="card-metrics-inline">
        <span class="param-chip">Q1 ${num(metrics.queue1)}</span>
        <span class="param-chip">Q2 ${num(metrics.queue2)}</span>
        <span class="param-chip">Q3 ${num(metrics.queue3)}</span>
        <span class="param-chip">Q4 ${num(metrics.queue4)}</span>
        <span class="param-chip">Open ${num(metrics.checkouts_open)}</span>
      </div>
    </div>
    <div class="card-field">
      <div class="card-field-label">Decision</div>
      <div class="card-field-value"><strong>${esc(action)}</strong></div>
    </div>
    <div class="card-field">
      <div class="card-field-label">Why</div>
      <div class="card-field-value card-why-value"></div>
    </div>
  `;
}

function num(v) {
  return v === null || v === undefined ? 0 : v;
}

function prependAndTrim(card) {
  $cards.prepend(card);

  const cards = $cards.children;
  while (cards.length > MAX_CARDS) {
    const last = cards[cards.length - 1];
    last.classList.add("fading");
    setTimeout(() => last.remove(), 400);
    break;
  }

  setTimeout(() => {
    while ($cards.children.length > MAX_CARDS + 1) {
      $cards.lastElementChild.remove();
    }
  }, 500);
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

/* ── Queue Monitor Frontend ──────────────────────────────────────────── */

const MAX_CARDS = 6;
let eventCount = 0;
let popupTimer = null;
const pendingCvCards = new Map();

// ── Checkout state ────────────────────────────────────────────────────────
let checkoutCount = 2;
let prevCounts = { 1: -1, 2: -1, 3: -1, 4: -1 };
const alertedCheckouts = new Set(); // lanes currently in alert state

// ── DOM refs ────────────────────────────────────────────────────────────
const $clock      = document.getElementById("clock");
const $lane1      = document.getElementById("m-lane1");
const $lane2      = document.getElementById("m-lane2");
const $lane3      = document.getElementById("m-lane3");
const $lane4      = document.getElementById("m-lane4");
const $totalWait  = document.getElementById("m-total-wait");
const $store      = document.getElementById("m-store");
const $status     = document.getElementById("m-status");
const $statusDot  = document.getElementById("status-dot");
const $cards      = document.getElementById("cards");
const $popup      = document.getElementById("popup");
const $eventCount = document.getElementById("event-count");
const $dots = {
  1: document.getElementById("dot-co1"),
  2: document.getElementById("dot-co2"),
  3: document.getElementById("dot-co3"),
  4: document.getElementById("dot-co4"),
};

// ── Clock ───────────────────────────────────────────────────────────────
function updateClock() {
  const now = new Date();
  $clock.textContent = now.toLocaleTimeString("en-GB", { hour12: false });
}

function resolveStoreCount(m) {
  const raw = m.store_count ?? m.customers_in_store ?? m.in_store_count ?? m.clients_visible;
  const n = Number(raw);
  return Number.isFinite(n) ? n : 0;
}

updateClock();
setInterval(updateClock, 1000);

// ── Update checkout dots and dim closed lanes ────────────────────────────
function updateCheckoutDots(open) {
  for (let i = 1; i <= 4; i++) {
    const isOpen = i <= open;
    if ($dots[i]) $dots[i].classList.toggle("open", isOpen);
    const block = document.getElementById(`mblock-co${i}`);
    if (block) block.classList.toggle("mblock-closed", !isOpen);
  }
}

function animateMblock(el) {
  if (!el) return;
  el.classList.remove("mblock-count-update");
  void el.offsetWidth;
  el.classList.add("mblock-count-update");
}

// ── Metrics polling ─────────────────────────────────────────────────────
let lastQueue1Wait = null;
let lastQueue2Wait = null;

async function pollMetrics() {
  try {
    const res = await fetch("/metrics", { cache: "no-store" });
    if (!res.ok) return;
    const m = await res.json();

    checkoutCount = m.checkouts_open ?? checkoutCount;
    updateCheckoutDots(checkoutCount);

    // Update all 4 lane counts
    const counts = {
      1: m.queue1 ?? 0,
      2: m.queue2 ?? 0,
      3: m.queue3 ?? 0,
      4: m.queue4 ?? 0,
    };
    const laneEls = { 1: $lane1, 2: $lane2, 3: $lane3, 4: $lane4 };

    for (let i = 1; i <= 4; i++) {
      const q = counts[i];
      if (q !== prevCounts[i]) {
        laneEls[i].innerHTML = `${q} <span class="mblock-unit">waiting</span>`;
        animateMblock(document.getElementById(`mblock-co${i}`));
        prevCounts[i] = q;
      }
    }

    // Clear alert highlight when count drops below threshold
    for (const lane of alertedCheckouts) {
      if (counts[lane] < 4) setCheckoutAlert(lane, false);
    }

    // Update sub-labels for lanes 3 & 4
    const lane3Sub = document.getElementById("m-lane3-sub");
    const lane4Sub = document.getElementById("m-lane4-sub");
    if (lane3Sub) lane3Sub.textContent = checkoutCount >= 3 ? "active" : "closed";
    if (lane4Sub) lane4Sub.textContent = checkoutCount >= 4 ? "active" : "closed";

    // Update avg wait times for lanes 1 & 2
    const lane1Sub = document.getElementById("m-lane1-sub");
    const lane2Sub = document.getElementById("m-lane2-sub");
    if (m.queue1_avg_wait !== null && m.queue1_avg_wait !== undefined) {
      lane1Sub.textContent = `avg wait ${m.queue1_avg_wait.toFixed(1)}s`;
      lastQueue1Wait = m.queue1_avg_wait;
    } else {
      lane1Sub.textContent = "avg wait —";
      lastQueue1Wait = null;
    }
    if (m.queue2_avg_wait !== null && m.queue2_avg_wait !== undefined) {
      lane2Sub.textContent = `avg wait ${m.queue2_avg_wait.toFixed(1)}s`;
      lastQueue2Wait = m.queue2_avg_wait;
    } else {
      lane2Sub.textContent = "avg wait —";
      lastQueue2Wait = null;
    }

    // Total average wait across all open lanes
    let totalWait = "—";
    if (lastQueue1Wait !== null && lastQueue2Wait !== null) {
      totalWait = ((lastQueue1Wait + lastQueue2Wait) / 2).toFixed(1);
    } else if (lastQueue1Wait !== null) {
      totalWait = lastQueue1Wait.toFixed(1);
    } else if (lastQueue2Wait !== null) {
      totalWait = lastQueue2Wait.toFixed(1);
    }
    $totalWait.innerHTML = `${totalWait} <span class="mblock-unit">seconds</span>`;

    $store.textContent  = resolveStoreCount(m);
    $status.textContent = m.status || "MONITORING";

    const isAlert = m.status === "ALERT";
    $statusDot.classList.toggle("alert", isAlert);
    $status.style.color = isAlert ? "#fca5a5" : "";
  } catch (_) { /* retry next tick */ }
}
pollMetrics();
setInterval(pollMetrics, 1500);

// ── Checkout alert highlight ─────────────────────────────────────────────
function setCheckoutAlert(lane, active) {
  const el = document.getElementById(`mblock-co${lane}`);
  if (!el) return;
  el.classList.toggle("mblock-alert", active);
  if (active) alertedCheckouts.add(lane);
  else alertedCheckouts.delete(lane);
}

// ── SSE: agent events ───────────────────────────────────────────────────
function connectSSE() {
  const source = new EventSource("/events");

  source.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    if (data.type === "queue_alert") {
      addCvAlertCard(data);
      setCheckoutAlert(data.lane, true);
      eventCount++;
      $eventCount.textContent = `${eventCount} event${eventCount !== 1 ? "s" : ""}`;
      return;
    }

    if (data.type === "close_alert") {
      addCloseAlertCard(data);
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
      <span class="event-badge event-badge-cv">EVENT</span>
      <span class="card-time">${esc(data.timestamp || "")}</span>
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

function addCloseAlertCard(data) {
  const card = document.createElement("div");
  card.className = "card card-event-close";
  const alertId = data.alert_id || `close-${Date.now()}`;
  card.dataset.alertId = alertId;
  card.dataset.alertMessage = data.message || `Checkout ${data.lane || "?"} underutilised`;

  card.innerHTML = `
    <div class="card-top">
      <span class="event-badge event-badge-close">CLOSE SUGGESTION</span>
      <span class="card-time">${esc(data.timestamp || "")}</span>
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

function decisionClass(action) {
  const a = (action || "none").toLowerCase();
  if (a.includes("open_register"))  return "card-decision-open";
  if (a.includes("close_register")) return "card-decision-close";
  return "card-decision-none";
}

function updateCvAlertWithDecision(data) {
  const alertId = data.alert_id;
  const card = alertId ? pendingCvCards.get(alertId) : null;
  const alertMessage = card ? card.dataset.alertMessage : null;

  if (!card) {
    const fallback = document.createElement("div");
    fallback.className = `card ${decisionClass(data.action)}`;
    fallback.innerHTML = buildDecisionBody("EVENT", "event-badge-cv", data, null);
    prependAndTrim(fallback);
    startWhyTypewriter(fallback, data);
    return;
  }

  card.className = `card ${decisionClass(data.action)}`;
  card.innerHTML = buildDecisionBody("EVENT", "event-badge-cv", data, alertMessage);
  pendingCvCards.delete(alertId);
  startWhyTypewriter(card, data);
}

function addScheduledCard(data) {
  const card = document.createElement("div");
  card.className = "card card-event-scheduled";
  card.innerHTML = `
    <div class="card-top">
      <span class="event-badge event-badge-scheduled">SCHEDULED</span>
      <span class="card-time">${esc(data.timestamp || "")}</span>
    </div>
    <div class="card-field">
      <div class="card-field-label">Trend Report</div>
      <div class="card-field-value card-report-value"></div>
    </div>
  `;
  prependAndTrim(card);
  const reportEl = card.querySelector(".card-report-value");
  if (reportEl) typewriter(reportEl, data.report || "No report available.");
  eventCount++;
  $eventCount.textContent = `${eventCount} event${eventCount !== 1 ? "s" : ""}`;
}

function startWhyTypewriter(card, data) {
  const whyEl = card.querySelector(".card-why-value");
  if (whyEl) {
    typewriter(whyEl, data.reasoning || data.situation || "No reasoning provided.");
  }
}

function actionLabel(action) {
  const a = (action || "none").toLowerCase();
  if (a.includes("open_register"))  return { text: "Opening Checkout", cls: "decision-label-open" };
  if (a.includes("close_register")) return { text: "Closing Checkout", cls: "decision-label-close" };
  return { text: "No Action Taken",        cls: "decision-label-none" };
}

function buildDecisionBody(label, badgeClass, data, alertMessage) {
  const metrics = data.metrics || {};
  const { text: decisionText, cls: decisionCls } = actionLabel(data.action);
  return `
    <div class="card-top">
      <span class="event-badge ${badgeClass}">${label}</span>
      <span class="card-time">${esc(data.timestamp || "")}</span>
    </div>
    ${alertMessage ? `
    <div class="card-field card-field-alert">
      <div class="card-field-label">Alert</div>
      <div class="card-field-value">${esc(alertMessage)}</div>
    </div>` : ""}
    <div class="decision-label ${decisionCls}">${decisionText}</div>
    <div class="card-field">
    <div class="card-field-label">Why</div>
    <div class="card-field-value card-why-value"></div>
    </div>
    <div class="card-field">
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

// ── Debug reset (Shift + R) ──────────────────────────────────────────────
document.addEventListener("keydown", async (e) => {
  if (e.key === "R" && e.shiftKey && !e.ctrlKey && !e.metaKey) {
    await fetch("/reset", { method: "POST" });
    // Reset local frontend state
    prevCounts = { 1: -1, 2: -1, 3: -1, 4: -1 };
    checkoutCount = 2;
    updateCheckoutDots(2);
    $cards.innerHTML = "";
    pendingCvCards.clear();
    eventCount = 0;
    $eventCount.textContent = "0 events";
    showResetToast();
  }
});

function showResetToast() {
  let toast = document.getElementById("reset-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "reset-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = "RESET";
  toast.classList.remove("toast-hide");
  toast.classList.add("toast-show");
  setTimeout(() => {
    toast.classList.replace("toast-show", "toast-hide");
  }, 1200);
}

// ── Utility ─────────────────────────────────────────────────────────────
function esc(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

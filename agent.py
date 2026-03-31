"""
LangGraph Agent — Queue Management Decision Maker
Uses Ollama (llama3.1) to analyze queue metrics and decide actions.
"""

import re
import time
from typing import Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def open_register(lane_id: int) -> str:
    """Open a checkout register for the specified lane."""
    return f"Register at lane {lane_id} is now OPEN. Cashier dispatched."


@tool
def close_register(lane_id: int) -> str:
    """Close a checkout register for the specified lane."""
    return f"Register at lane {lane_id} is now CLOSED."


@tool
def alert_supervisor(message: str, urgency: str) -> str:
    """Alert a floor supervisor with a message and urgency level."""
    return f"Supervisor alerted [{urgency.upper()}]: {message}"


@tool
def flag_anomaly(description: str) -> str:
    """Flag an anomaly detected in the store for review."""
    return f"Anomaly flagged for review: {description}"


@tool
def generate_shift_report() -> str:
    """Generate a summary report for the current shift."""
    return "Shift report generated and queued for delivery to management."


TOOLS = [open_register, close_register, alert_supervisor, flag_anomaly, generate_shift_report]
TOOL_MAP = {t.name: t for t in TOOLS}


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an AI queue management agent monitoring a supermarket checkout area in real-time.
You receive queue metrics every 5 seconds and must decide if any action is needed.

Available tools:
  open_register(lane_id)        — open a new checkout lane
  close_register(lane_id)       — close an idle checkout lane
  alert_supervisor(message, urgency) — page the floor supervisor
  flag_anomaly(description)     — flag unusual activity for review
  generate_shift_report()       — create a shift summary

RULES:
- Do NOT act every cycle. Only act when genuinely necessary.
- Queue counts of 0-2 per lane are normal. 3-4 is busy. 5+ needs action.
- Growing trends with 4+ people warrant opening a register.
- Only alert the supervisor for truly unusual or urgent situations.
- If nothing notable is happening, say so and set urgency to low.
- Keep responses concise — one sentence per field.

Respond in EXACTLY this format (no extra text):
SITUATION: <one sentence>
REASONING: <one sentence>
ACTION: <tool_name(params)> or none
URGENCY: <low | medium | high>
"""


# ── LangGraph state & nodes ─────────────────────────────────────────────────

class AgentState(TypedDict):
    metrics: dict
    situation: str
    reasoning: str
    action: str
    urgency: str
    tool_result: Optional[str]
    raw: str


def analyze_node(state: AgentState) -> dict:
    """Call the LLM with current metrics."""
    m = state["metrics"]
    human_msg = (
        f"Current queue metrics:\n"
        f"- Lane 1: {m.get('queue1', 0)} people (trend: {m.get('queue1_trend', 'stable')})\n"
        f"- Lane 2: {m.get('queue2', 0)} people (trend: {m.get('queue2_trend', 'stable')})\n"
        f"- Total in store: {m.get('store_count', 0)}\n"
        f"- Employees visible: {m.get('employees', 0)}\n"
        f"- Avg wait Lane 1: {_fmt_wait(m.get('queue1_avg_wait'))}\n"
        f"- Avg wait Lane 2: {_fmt_wait(m.get('queue2_avg_wait'))}\n"
    )
    try:
        import os
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        llm = ChatOllama(model="llama3.1", temperature=0.3, base_url=ollama_host)
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_msg),
        ])
        parsed = parse_agent_response(response.content)
        return parsed
    except Exception as e:
        return {
            "situation": "Agent connection unavailable",
            "reasoning": f"Could not reach Ollama: {str(e)[:120]}",
            "action": "none",
            "urgency": "low",
            "raw": str(e),
        }


def tool_node(state: AgentState) -> dict:
    """Execute the tool specified in the action field."""
    result = _execute_tool(state["action"])
    return {"tool_result": result}


def _should_use_tool(state: AgentState) -> str:
    action = state.get("action", "none").strip().lower()
    if action in ("none", "no action", ""):
        return END
    return "execute_tool"


# ── Build graph ──────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(AgentState)
    g.add_node("analyze", analyze_node)
    g.add_node("execute_tool", tool_node)
    g.add_edge(START, "analyze")
    g.add_conditional_edges("analyze", _should_use_tool, {"execute_tool": "execute_tool", END: END})
    g.add_edge("execute_tool", END)
    return g.compile()


_agent = _build_graph()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fmt_wait(val):
    if val is None:
        return "N/A"
    return f"{val:.1f}s"


def parse_agent_response(text: str) -> dict:
    """Parse the structured SITUATION/REASONING/ACTION/URGENCY response."""
    result = {
        "situation": "",
        "reasoning": "",
        "action": "none",
        "urgency": "low",
        "raw": text,
    }
    for line in text.strip().split("\n"):
        line = line.strip()
        upper = line.upper()
        if upper.startswith("SITUATION:"):
            result["situation"] = line.split(":", 1)[1].strip()
        elif upper.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
        elif upper.startswith("ACTION:"):
            result["action"] = line.split(":", 1)[1].strip()
        elif upper.startswith("URGENCY:"):
            result["urgency"] = line.split(":", 1)[1].strip().lower()
    return result


def _execute_tool(action_str: str) -> Optional[str]:
    """Parse 'tool_name(args)' and invoke the matching tool."""
    if action_str.strip().lower() in ("none", "no action", ""):
        return None
    match = re.match(r"(\w+)\((.*)\)", action_str.strip())
    if not match:
        return f"Could not parse action: {action_str}"
    name = match.group(1)
    if name not in TOOL_MAP:
        return f"Unknown tool: {name}"
    args_raw = match.group(2).strip()

    tool_fn = TOOL_MAP[name]
    try:
        # Build kwargs from the tool's parameters
        if name == "open_register" or name == "close_register":
            lane_id = int(re.search(r"\d+", args_raw).group())
            return tool_fn.invoke({"lane_id": lane_id})
        elif name == "alert_supervisor":
            parts = args_raw.split(",", 1)
            msg = parts[0].strip().strip("\"'")
            urg = parts[1].strip().strip("\"'") if len(parts) > 1 else "medium"
            return tool_fn.invoke({"message": msg, "urgency": urg})
        elif name == "flag_anomaly":
            desc = args_raw.strip().strip("\"'")
            return tool_fn.invoke({"description": desc})
        elif name == "generate_shift_report":
            return tool_fn.invoke({})
        else:
            return f"Tool {name} executed."
    except Exception as e:
        return f"Tool error: {e}"


# ── Public API ───────────────────────────────────────────────────────────────

def run_agent(metrics: dict) -> dict:
    """
    Run the LangGraph agent with current metrics.
    Returns dict with: situation, reasoning, action, urgency, tool_result, raw, timestamp.
    """
    initial_state = {
        "metrics": metrics,
        "situation": "",
        "reasoning": "",
        "action": "none",
        "urgency": "low",
        "tool_result": None,
        "raw": "",
    }
    try:
        result = _agent.invoke(initial_state)
        result["timestamp"] = time.strftime("%H:%M:%S")
        result.pop("metrics", None)
        return result
    except Exception as e:
        return {
            "situation": "Agent error",
            "reasoning": str(e)[:150],
            "action": "none",
            "urgency": "low",
            "tool_result": None,
            "raw": str(e),
            "timestamp": time.strftime("%H:%M:%S"),
        }

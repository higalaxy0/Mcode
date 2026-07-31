"""Message bus + protocol state machine."""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import MAILBOX_DIR, WORKDIR, IDLE_POLL_INTERVAL, IDLE_TIMEOUT
from .context import ctx
from .tasks import scan_unclaimed_tasks, claim_task
from .utils import new_request_id


class MessageBus:
    """In-process message bus backed by JSONL files.

    Each agent owns a ``.mailboxes/<name>.jsonl`` inbox.
    ``read_inbox`` uses atomic rename to avoid concurrent duplicate reads.
    """

    _read_counter = 0
    _read_lock = threading.Lock()

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None) -> None:
        """Append a message to *to_agent*'s inbox."""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        print(f" \033[33m[bus] {from_agent} -> {to_agent}: "
              f"({msg_type}) {content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        """Read and clear *agent*'s inbox (atomic rename)."""
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        with self._read_lock:
            self._read_counter += 1
            tmp = MAILBOX_DIR / f"{agent}.jsonl.reading_{self._read_counter}"
        try:
            inbox.rename(tmp)
        except FileNotFoundError:
            return []
        except Exception as e:
            print(f"  \033[31m[bus] read_inbox rename failed for "
                  f"{agent}: {e}\033[0m")
            return []
        msgs = [json.loads(line) for line in tmp.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        return msgs


@dataclass
class ProtocolState:
    """State of a single protocol interaction (shutdown / plan_approval)."""
    request_id: str
    type: str
    sender: str
    target: str
    status: str
    payload: str
    created_at: float = field(default_factory=time.time)


def match_response(response_type: str, request_id: str, approve: bool) -> None:
    """Update the ProtocolState matching a protocol response received by lead."""
    state = ctx.pending_requests.get(request_id)
    if not state:
        print(f"  \033[31m[protocol] unknown request_id:{request_id}\033[0m")
        return
    if state.type == "shutdown" and response_type != "shutdown_response":
        print(f"  \033[31m[protocol] type mismatch: expected shutdown_response, "
              f"got {response_type}\033[0m")
        return
    if state.type == "plan_approval" and response_type != "plan_approval_response":
        print(f"  \033[31m[protocol] type mismatch: expected plan_approval_response, "
              f"got {response_type}\033[0m")
        return
    state.status = "approved" if approve else "rejected"
    icon = "✅" if approve else "❌"
    color = "32" if approve else "31"
    print(f"  \033[{color}m[protocol] {state.type} {icon} "
          f"({request_id}: {state.status})\033[0m")


def consume_lead_inbox(route_protocol: bool = True) -> list[dict]:
    """Read lead's inbox, optionally routing protocol response messages."""
    msgs = ctx.bus.read_inbox("lead")
    if not msgs:
        return []
    if route_protocol:
        for msg in msgs:
            meta = msg.get("metadata", {})
            req_id = meta.get("request_id", "")
            msg_type = msg.get("type", "")
            if req_id and msg_type.endswith("_response"):
                approve = meta.get("approve", False)
                match_response(msg_type, req_id, approve)
    return msgs


def idle_poll(agent_name: str, messages: list,
              name: str, role: str) -> str:
    """Teammate idle poll: check inbox/unclaimed tasks; return ``timeout`` on timeout."""
    for _ in range(IDLE_TIMEOUT // IDLE_POLL_INTERVAL):
        time.sleep(IDLE_POLL_INTERVAL)

        inbox = ctx.bus.read_inbox(agent_name)
        if inbox:
            for msg in inbox:
                if msg.get("type") == "shutdown_request":
                    req_id = msg.get("metadata", {}).get("request_id", "")
                    ctx.bus.send(name, "lead", "Shutting down gracefully.",
                                 "shutdown_response",
                                 {"request_id": req_id, "approve": True})
                    print(f"  \033[35m[protocol] {name} approved shutdown "
                          f"in idle ({req_id})\033[0m")
                    return "shutdown"
            messages.append({"role": "user",
                             "content": "<inbox>" + json.dumps(inbox) + "</inbox>"})
            print(f"  \033[36m[idle] {name} found inbox messages\033[0m")
            return "work"

        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            task = unclaimed[0]
            result = claim_task(task["id"], agent_name)
            if "Claimed" in result:
                messages.append({"role": "user",
                                 "content": f"<auto-claimed>Task {task['id']}: "
                                            f"{task['subject']}</auto-claimed>"})
                print(f"  \033[32m[idle]{name} auto-claimed: "
                      f"{task['subject']}\033[0m")
                return "work"
            print(f"  \033[33m[idle] {name} claim failed: "
                  f"{result}\033[0m")
    print(f"  \033[31m[idle] {name} timeout ({IDLE_TIMEOUT}s)\033[0m")
    return "timeout"


# -- Protocol operations (lead-side handlers) --------------------------------- #

def _teammate_submit_plan(from_name: str, plan: str) -> str:
    """A teammate submits a plan to lead for approval."""
    req_id = new_request_id()
    ctx.pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="plan_approval",
        sender=from_name, target="lead",
        status="pending", payload=plan)
    ctx.bus.send(from_name, "lead", plan,
                 "plan_approval_request",
                 {"request_id": req_id})
    return f"Plan submitted ({req_id}). Waiting for approval..."


def run_request_shutdown(teammate: str) -> str:
    """Lead requests a teammate to shut down."""
    req_id = new_request_id()
    ctx.pending_requests[req_id] = ProtocolState(
        request_id=req_id, type="shutdown",
        sender="lead", target=teammate,
        status="pending", payload="")
    ctx.bus.send("lead", teammate, "Please shut down gracefully.",
                 "shutdown_request",
                 {"request_id": req_id})
    print(f"  \033[35m[protocol] shutdown_request -> {teammate} "
          f"({req_id})\033[0m")
    return f"Shutdown request sent to {teammate} (req: {req_id})"


def run_request_plan(teammate: str, task: str) -> str:
    """Lead asks a teammate to submit a plan."""
    ctx.bus.send("lead", teammate, f"Please submit a plan for: {task}",
                 "message")
    return f"Asked {teammate} to submit a plan"


def run_review_plan(request_id: str, approve: bool, feedback: str = "") -> str:
    """Lead reviews a plan submitted by a teammate."""
    state = ctx.pending_requests.get(request_id)
    if not state:
        return f"Request {request_id} not found"
    if state.status != "pending":
        return f"Request {request_id} already {state.status}"
    state.status = "approved" if approve else "rejected"
    ctx.bus.send("lead", state.sender, feedback or ("Approved" if approve else "Rejected"),
                 "plan_approval_response",
                 {"request_id": request_id, "approve": approve})
    return f"Plan {'approved' if approve else 'rejected'} ({request_id})"


def run_send_message(to: str, content: str) -> str:
    """Lead sends a message to a teammate."""
    ctx.bus.send("lead", to, content)
    return f"Sent to {to}"


def run_check_inbox(include_read: bool = False) -> str:
    """Lead checks the inbox."""
    if isinstance(include_read, str):
        include_read = include_read.strip().lower() == "true"
    include_read = bool(include_read)
    msgs = consume_lead_inbox(route_protocol=True)
    if not msgs:
        return "(inbox empty)"
    lines = []
    for m in msgs:
        meta = m.get("metadata", {})
        req_id = meta.get("request_id", "")
        tag = f"  [{m['type']} req:{req_id}]" if req_id else f"  [{m['type']}]"
        lines.append(f"  [{m['from']}]{tag} {m['content'][:200]}")
    return "\n".join(lines)

"""Message bus + protocol state machine."""

from __future__ import annotations

import json
import random
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .config import MAILBOX_DIR, WORKDIR, IDLE_POLL_INTERVAL, IDLE_TIMEOUT, debug
from .context import ctx
from .tasks import scan_unclaimed_tasks, claim_task, list_owned_inprogress
from .utils import new_request_id


class MessageBus:
    """In-process message bus backed by JSONL files.

    Each agent owns a ``.mailboxes/<name>.jsonl`` inbox.
    ``read_inbox`` uses atomic rename to avoid concurrent duplicate reads.
    """

    _read_counter = 0
    _io_lock = threading.Lock()

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message", metadata: dict = None) -> None:
        """Append a message to *to_agent*'s inbox."""
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time(), "metadata": metadata or {}}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        # Serialize concurrent appends so messages are never interleaved
        # or lost (especially when multiple teammates write to lead's
        # inbox simultaneously).  The same lock guards ``read_inbox``'s
        # rename so that a writer and reader never collide on Windows
        # (WinError 32 "file in use by another process").
        with self._io_lock:
            with open(inbox, "a", encoding="utf-8") as f:
                f.write(json.dumps(msg, ensure_ascii=False) + "\n")
        debug(f"[bus] {from_agent} -> {to_agent}: ({msg_type}) {content[:50]}")

    def read_inbox(self, agent: str) -> list[dict]:
        """Read and clear *agent*'s inbox (atomic rename)."""
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        # Hold _io_lock only for the rename (the critical section where a
        # concurrent writer could otherwise cause WinError 32).  Parsing
        # happens outside the lock because the temp file is exclusively
        # ours after a successful rename.
        with self._io_lock:
            self._read_counter += 1
            tmp = MAILBOX_DIR / f"{agent}.jsonl.reading_{self._read_counter}"
            try:
                inbox.rename(tmp)
            except FileNotFoundError:
                return []
            except OSError as e:
                debug(f"[bus] read_inbox rename failed for {agent}: {e}")
                return []
        lines = tmp.read_text(encoding="utf-8").splitlines()
        msgs = []
        for line in lines:
            if not line.strip():
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                debug(f"[bus] skipping corrupted inbox line for {agent}: {line[:80]}")
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
    task_id: str = ""  # binds a plan-approval to a specific task (F-2)
    created_at: float = field(default_factory=time.time)


def match_response(response_type: str, request_id: str, approve: bool) -> None:
    """Update the ProtocolState matching a protocol response received by lead."""
    with _requests_lock:
        state = ctx.pending_requests.get(request_id)
        if not state:
            debug(f"[protocol] unknown request_id:{request_id}")
            return
        if state.type == "shutdown" and response_type != "shutdown_response":
            debug(f"[protocol] type mismatch: expected shutdown_response, got {response_type}")
            return
        if state.type == "plan_approval" and response_type != "plan_approval_response":
            debug(f"[protocol] type mismatch: expected plan_approval_response, got {response_type}")
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


# Lock guarding ``ctx.pending_requests`` (F-9).  Lead-side review and
# teammate-side submit both touch this dict from different threads.
_requests_lock = threading.Lock()


def idle_poll(agent_name: str, messages: list,
              name: str, role: str = "") -> str:
    """Teammate idle poll: check inbox/unclaimed tasks; return ``timeout`` on timeout.

    Improvements over the original:
    - Jitter: sleep ``IDLE_POLL_INTERVAL + random(0..2)`` seconds so
      multiple idle teammates do not wake in lockstep and thunder on the
      same first task.
    - Fall-through: when ``claim_task`` fails on ``unclaimed[i]`` we try
      ``unclaimed[i+1]``, ... in the same cycle instead of sleeping and
      retrying the same slot.
    - Protocol routing: inbox messages are processed via the same
      ``handle_inbox_msg`` callback as the main loop so plan-approval
      responses are structured correctly.
    - Batch shutdown: if a shutdown_request is found, remaining messages
      in the same batch are still processed before returning.
    """
    for _ in range(IDLE_TIMEOUT // max(IDLE_POLL_INTERVAL, 1)):
        # Jitter: 0-2 s random addition prevents synchronized wake-ups
        jitter = random.uniform(0, 2)
        time.sleep(IDLE_POLL_INTERVAL + jitter)

        inbox = ctx.bus.read_inbox(agent_name)
        if inbox:
            shutdown_found = False
            protocol_processed = False
            non_protocol = []
            for msg in inbox:
                msg_type = msg.get("type", "message")
                meta = msg.get("metadata", {})
                req_id = meta.get("request_id", "")

                if msg_type == "shutdown_request":
                    ctx.bus.send(name, "lead", "Shutting down gracefully.",
                                 "shutdown_response",
                                 {"request_id": req_id, "approve": True})
                    print(f"  \033[35m[protocol] {name} approved shutdown "
                          f"in idle ({req_id})\033[0m")
                    shutdown_found = True
                    # continue processing remaining messages, do NOT return yet
                    continue

                if msg_type == "plan_approval_response":
                    approve = meta.get("approve", False)
                    if approve:
                        messages.append({"role": "user",
                                         "content": "[Plan approved] "
                                         "Proceed with the task."})
                    else:
                        messages.append({"role": "user",
                                         "content": f"[Plan rejected] "
                                         f"Feedback: {msg.get('content', '')}"})
                    protocol_processed = True
                    continue

                # Ordinary message - collect for LLM
                non_protocol.append(msg)

            if non_protocol:
                messages.append({"role": "user",
                                 "content": "<inbox>" +
                                 json.dumps(non_protocol) + "</inbox>"})
                debug(f"[idle] {name} found inbox messages")

            if shutdown_found:
                return "shutdown"
            if non_protocol or protocol_processed:
                return "work"

        # Resume owned in_progress tasks first (e.g. after turn-budget
        # exhaustion).  These are tasks this agent already claimed but
        # has not completed yet.  We re-inject a reminder so the LLM
        # picks up where it left off.
        owned = list_owned_inprogress(agent_name)
        if owned:
            parts = [f"{t['id']}: {t['subject']}" for t in owned]
            messages.append({"role": "user",
                             "content": f"<resume>You still own these "
                                        f"in_progress tasks: {'; '.join(parts)}. "
                                        f"Continue working on them.</resume>"})
            debug(f"[idle] {name} resuming {len(owned)} owned in_progress task(s)")
            return "work"

        unclaimed = scan_unclaimed_tasks()
        if unclaimed:
            # Try each unclaimed task in order (fall-through on failure)
            for task in unclaimed:
                result = claim_task(task["id"], agent_name)
                if "Claimed" in result:
                    messages.append({"role": "user",
                                     "content": f"<auto-claimed>Task {task['id']}: "
                                                f"{task['subject']}</auto-claimed>"})
                    debug(f"[idle] {name} auto-claimed: {task['subject']}")
                    return "work"
                # else: try next task in the same cycle
    debug(f"[idle] {name} timeout ({IDLE_TIMEOUT}s)")
    return "timeout"


# -- Protocol operations (lead-side handlers) --------------------------------- #

def _teammate_submit_plan(from_name: str, plan: str,
                          task_id: str = "") -> str:
    """A teammate submits a plan to lead for approval.

    If *task_id* is provided, ownership is validated: the task must be
    ``in_progress`` and owned by *from_name*.  This prevents a teammate
    from submitting a plan for a task it does not own.
    """
    from .tasks import load_task
    if task_id:
        try:
            task = load_task(task_id)
        except FileNotFoundError:
            return (f"Cannot submit plan: task {task_id} not found")
        if task.status != "in_progress":
            return (f"Cannot submit plan: task {task_id} is {task.status},"
                    f"not in_progress")
        if task.owner and task.owner != from_name:
            return (f"Cannot submit plan: task {task_id} is owned by "
                    f"{task.owner},not {from_name}")
    with _requests_lock:
        req_id = new_request_id()
        ctx.pending_requests[req_id] = ProtocolState(
            request_id=req_id, type="plan_approval",
            sender=from_name, target="lead",
            status="pending", payload=plan, task_id=task_id)
    ctx.bus.send(from_name, "lead", plan,
                 "plan_approval_request",
                 {"request_id": req_id, "task_id": task_id})
    return f"Plan submitted ({req_id}). Waiting for approval..."


def run_request_shutdown(teammate: str) -> str:
    """Lead requests a teammate to shut down."""
    with _requests_lock:
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
    """Lead asks a teammate to submit a plan for *task*.

    Creates a ``ProtocolState`` so the request is tracked and can be
    linked to the teammate's subsequent ``submit_plan`` response.
    """
    with _requests_lock:
        req_id = new_request_id()
        ctx.pending_requests[req_id] = ProtocolState(
            request_id=req_id, type="plan_approval",
            sender="lead", target=teammate,
            status="pending", payload=task, task_id=task)
    ctx.bus.send("lead", teammate, f"Please submit a plan for: {task}",
                 "message",
                 {"request_id": req_id})
    print(f"  \033[35m[protocol] plan_request -> {teammate} "
          f"({req_id})\033[0m")
    return f"Asked {teammate} to submit a plan (req: {req_id})"


def run_review_plan(request_id: str, approve: bool, feedback: str = "") -> str:
    """Lead reviews a plan submitted by a teammate."""
    with _requests_lock:
        state = ctx.pending_requests.get(request_id)
        if not state:
            return f"Request {request_id} not found"
        if state.status != "pending":
            return f"Request {request_id} already {state.status}"
        state.status = "approved" if approve else "rejected"
        sender = state.sender
    ctx.bus.send("lead", sender, feedback or ("Approved" if approve else "Rejected"),
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

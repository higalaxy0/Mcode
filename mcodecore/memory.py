"""Memory system: read/write / extract / consolidate / async loading."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path

from .config import (MEMORY_DIR, MEMORY_INDEX, CONSOLIDATE_THRESHOLD,
                     LLM_MODEL, client)
from .context import ctx
from .utils import parse_frontmatter


def _now_iso() -> str:
    """Compact ISO-8601 timestamp for frontmatter (second precision)."""
    return time.strftime("%Y%m%dT%H%M%S", time.localtime())


def _slugify(name: str) -> str:
    """Convert a memory name to a filesystem-safe slug.

    Lowercases, replaces spaces and slashes with hyphens, and strips
    characters that are illegal in Windows filenames.
    """
    slug = name.lower().replace(" ", "-").replace("/", "-")
    # Remove characters that are illegal in Windows filenames.
    slug = re.sub(r'[<>:"|?*\\]', "", slug)
    # Collapse multiple hyphens.
    slug = re.sub(r'-{2,}', '-', slug).strip('-')
    return slug or "memory"


def _resolve_filepath(name: str, directory: Path) -> tuple[Path, bool]:
    """Determine the filepath for *name*, handling slug collisions.

    Returns ``(filepath, is_update)`` where *is_update* is True when the
    file already exists **and** its stored ``name`` field matches *name*
    (a legitimate update).  When the file exists but the stored name
    differs (a slug collision from a distinct memory), a numeric suffix
    (``-2``, ``-3``, ...) is appended until a free slot is found, and
    *is_update* is False.
    """
    slug = _slugify(name)
    candidate = directory / f"{slug}.md"
    if not candidate.exists():
        return candidate, False
    # File exists -- is it the same memory (update) or a slug collision?
    existing_meta, _ = parse_frontmatter(candidate.read_text())
    if existing_meta.get("name", "") == name:
        return candidate, True
    # Slug collision: find the next available suffix.
    for i in range(2, 100):
        candidate = directory / f"{slug}-{i}.md"
        if not candidate.exists():
            return candidate, False
        existing_meta, _ = parse_frontmatter(candidate.read_text())
        if existing_meta.get("name", "") == name:
            return candidate, True
    # Fallback: use timestamp to guarantee uniqueness.
    return directory / f"{slug}-{int(time.time())}.md", False


def _build_frontmatter(meta: dict) -> str:
    """Serialize a metadata dict to YAML-like frontmatter text."""
    lines = ["---"]
    for key in ("name", "description", "type",
                "created_at", "updated_at",
                "hit_count", "last_used", "expires_at"):
        if key in meta and meta[key] is not None:
            lines.append(f"{key}: {meta[key]}")
    lines.append("---")
    return "\n".join(lines)


def _rebuild_index(directory: Path | None = None) -> None:
    """Rebuild the MEMORY.md index from scratch (full directory scan).

    If *directory* is None the live :data:`MEMORY_DIR` is used.
    """
    target_dir = directory or MEMORY_DIR
    index_path = target_dir / "MEMORY.md"
    lines = []
    for f in sorted(target_dir.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) - {desc}")
    index_path.write_text("\n".join(lines) + "\n" if lines else "")


def _write_memory_file_no_index(name: str, mem_type: str,
                                description: str, body: str,
                                directory: Path | None = None,
                                *, created_at: str | None = None,
                                expires_at: str | None = None,
                                ) -> Path:
    """Write a memory .md file WITHOUT rebuilding the index.

    Used by batch writers (extract/consolidate) which rebuild the index once
    at the end via :func:`_rebuild_index`, avoiding O(N) rebuilds per file.

    If the target file already exists **and** its stored ``name`` matches,
    the original ``created_at`` is preserved and ``updated_at`` is refreshed
    (update semantics).  If a file with the same slug exists but stores a
    *different* name, a numeric suffix is used so both memories coexist
    (collision avoidance -- no silent overwrite).

    ``expires_at`` (optional) sets a TTL deadline in ``YYYYMMDDTHHMMSS``
    format.  When ``None`` and the file already exists, the existing
    ``expires_at`` is preserved.
    """
    if directory is None:
        directory = MEMORY_DIR
    filepath, is_update = _resolve_filepath(name, directory)
    now = _now_iso()
    # Preserve created_at from existing file (update path only).
    if created_at is None:
        if is_update and filepath.exists():
            old_meta, _ = parse_frontmatter(filepath.read_text())
            created_at = old_meta.get("created_at", now)
        else:
            created_at = now
    # Preserve hit_count / last_used / expires_at from existing file if present.
    hit_count = "0"
    last_used = now
    if is_update and filepath.exists():
        old_meta, _ = parse_frontmatter(filepath.read_text())
        hit_count = old_meta.get("hit_count", "0")
        last_used = old_meta.get("last_used", now)
        if expires_at is None:
            expires_at = old_meta.get("expires_at")
    meta = {
        "name": name,
        "description": description,
        "type": mem_type,
        "created_at": created_at,
        "updated_at": now,
        "hit_count": hit_count,
        "last_used": last_used,
    }
    if expires_at:
        meta["expires_at"] = expires_at
    filepath.write_text(f"{_build_frontmatter(meta)}\n\n{body}\n")
    return filepath


def write_memory_file(name: str, mem_type: str, description: str, body: str):
    """Write a memory .md file and rebuild the index."""
    path = _write_memory_file_no_index(name, mem_type, description, body, MEMORY_DIR)
    _rebuild_index()
    return path


# --------------------------------------------------------------------------- #
# TTL / dead-memory utilities (Plan D)
# --------------------------------------------------------------------------- #

# Dead-memory threshold: never accessed (hit_count == 0) and last_used older
# than this many days.  These are candidates for removal during consolidation.
DEAD_MEMORY_DAYS = 7


def _parse_iso(ts: str) -> float:
    """Parse a ``YYYYMMDDTHHMMSS`` timestamp to epoch seconds.  0 on error."""
    try:
        return time.mktime(time.strptime(ts, "%Y%m%dT%H%M%S"))
    except (ValueError, TypeError):
        return 0.0


def is_expired(meta: dict) -> bool:
    """Return True if the memory's ``expires_at`` deadline has passed."""
    exp = meta.get("expires_at", "")
    if not exp:
        return False
    deadline = _parse_iso(exp)
    if deadline == 0:
        return False
    return time.time() > deadline


def is_dead_memory(meta: dict, days: int = DEAD_MEMORY_DAYS) -> bool:
    """Return True if a memory is 'dead': never accessed and stale.

    A dead memory has ``hit_count == 0`` **and** ``last_used`` older than
    *days* days.  These are the first candidates for removal during
    consolidation.
    """
    try:
        hc = int(meta.get("hit_count", "0") or "0")
    except (ValueError, TypeError):
        hc = 0
    if hc > 0:
        return False
    last = _parse_iso(meta.get("last_used", ""))
    if last == 0:
        # No last_used at all -- treat as dead if created_at is old.
        last = _parse_iso(meta.get("created_at", ""))
    if last == 0:
        return False
    age_seconds = time.time() - last
    return age_seconds > days * 86400


def cleanup_stale_memories() -> int:
    """Remove expired and dead memories from the store.

    Returns the number of files removed.  Called during the post-turn
    hook **before** consolidation, so the consolidation LLM sees a cleaner
    catalog.  Feedback-type memories are never auto-removed (user guidance
    is always kept regardless of TTL or access count).
    """
    removed = 0
    for f in MEMORY_DIR.glob("*.md"):
        if f.name == "MEMORY.md":
            continue
        try:
            raw = f.read_text()
            meta, _ = parse_frontmatter(raw)
        except Exception:
            continue
        # Feedback memories are never auto-removed.
        if meta.get("type") == "feedback":
            continue
        if is_expired(meta) or is_dead_memory(meta):
            try:
                f.unlink()
                removed += 1
            except Exception:
                pass
    if removed:
        _rebuild_index()
    return removed


def read_memory_index() -> str:
    """Read the MEMORY.md index text."""
    if not MEMORY_INDEX.exists():
        return ""
    text = MEMORY_INDEX.read_text().strip()
    return text if text else ""


def read_memory_file(filename: str) -> str | None:
    """Read the content of a single memory file."""
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


def _touch_memory(filename: str) -> None:
    """Increment ``hit_count`` and refresh ``last_used`` for a memory file.

    Called when a memory is actually injected into the conversation, so that
    consolidation can distinguish frequently-used ("hot") memories from
    never-accessed ("dead") ones.  Must be called while holding
    ``ctx.memory_lock``.  Failures are silently ignored (metadata is
    best-effort).
    """
    path = MEMORY_DIR / filename
    if not path.exists():
        return
    try:
        raw = path.read_text()
        meta, body = parse_frontmatter(raw)
        meta["hit_count"] = str(int(meta.get("hit_count", "0") or "0") + 1)
        meta["last_used"] = _now_iso()
        # Rebuild only the keys we know about.
        full_meta = {
            "name": meta.get("name", path.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "created_at": meta.get("created_at", _now_iso()),
            "updated_at": meta.get("updated_at", _now_iso()),
            "hit_count": meta["hit_count"],
            "last_used": meta["last_used"],
        }
        if meta.get("expires_at"):
            full_meta["expires_at"] = meta["expires_at"]
        path.write_text(f"{_build_frontmatter(full_meta)}\n\n{body}\n")
    except Exception:
        pass


def list_memory_files() -> list[dict]:
    """List all memory files (excluding the MEMORY.md index)."""
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("type", "user"),
            "body": body,
            "created_at": meta.get("created_at", ""),
            "updated_at": meta.get("updated_at", ""),
            "hit_count": int(meta.get("hit_count", "0") or "0"),
            "last_used": meta.get("last_used", ""),
            "expires_at": meta.get("expires_at", ""),
        })
    return result


def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    """Select memory filenames relevant to the current conversation.

    Strategy (LLM-first with keyword fallback):

    1. **feedback always-inject**: memories with ``type == 'feedback'`` are
       always included (user guidance / constraints are global, not
       context-dependent).
    2. **LLM selection**: the catalog includes ``type`` so the model can
       prefer ``user`` > ``project`` > ``reference`` when space is tight.
    3. **Keyword fallback**: when the LLM call fails, matching is done
       against ``name + description + body[:200]`` (body participates to
       improve recall), and feedback memories are still force-included.
    """
    files = list_memory_files()
    if not files:
        return []

    # --- Always-inject feedback memories ---
    feedback_files = [f for f in files if f["type"] == "feedback"]
    feedback_names = {f["filename"] for f in feedback_files}

    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(getattr(b, "text", "")) for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]
    if not recent.strip():
        # Even with no recent text, feedback memories are always injected.
        return [f["filename"] for f in feedback_files][:max_items]

    catalog_lines = []
    for i, f in enumerate(files):
        type_tag = f"[{f['type']}]"
        catalog_lines.append(f"{i}: {type_tag} {f['name']} - {f['description']}")
    catalog = "\n".join(catalog_lines)

    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n"
        "Priority: user > feedback > project > reference "
        "(but feedback-type memories are always relevant as they encode "
        "global user guidance).\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}")

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            timeout=30,
        )
        _raw = response.choices[0].message.content
        text = (_raw or "").strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(files):
                    fn = files[idx]["filename"]
                    if fn not in selected:
                        selected.append(fn)
                    if len(selected) >= max_items:
                        break
            # Merge feedback always-inject (if not already selected).
            for f in feedback_files:
                if f["filename"] not in selected and len(selected) < max_items:
                    selected.append(f["filename"])
            return selected[:max_items]
    except Exception:
        pass

    # --- Keyword fallback: body[:200] participates in matching ---
    keywords = [w.lower() for w in recent.split() if len(w) > 3]
    selected: list[str] = []
    # Always include feedback memories first.
    for f in feedback_files:
        selected.append(f["filename"])
        if len(selected) >= max_items:
            return selected[:max_items]
    for f in files:
        if f["filename"] in selected:
            continue
        # Body participates: first 200 chars for matching.
        text = (f["name"] + " " + f["description"] + " " + f["body"][:200]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected[:max_items]


def load_memories(messages: list) -> str:
    """Load relevant memory contents into a ``<relevant_memories>`` text block."""
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""
    parts = ["<relevant_memories>"]
    if not ctx.memory_lock.acquire(timeout=ctx.memory_lock_timeout):
        return ""
    try:
        for filename in selected_files:
            content = read_memory_file(filename)
            if content:
                parts.append(content)
                _touch_memory(filename)
    finally:
        ctx.memory_lock.release()
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def _parse_json_array_robust(text: str) -> list | None:
    """Parse a JSON array from LLM output, tolerating truncation.

    Strategy (ordered by cost):
    1. ``raw_decode`` from the first ``[`` -- handles the common case.
    2. If that fails, attempt to auto-close unbalanced brackets/braces and
       retry -- handles truncation mid-item.
    Returns ``None`` if no JSON array could be recovered.
    """
    start = text.find('[')
    if start == -1:
        return None
    try:
        items, _ = json.JSONDecoder().raw_decode(text[start:])
        return items if isinstance(items, list) else None
    except json.JSONDecodeError:
        pass
    # Fallback: auto-close truncated JSON by balancing brackets.
    frag = text[start:]
    opens_b = frag.count('[') - frag.count(']')
    opens_c = frag.count('{') - frag.count('}')
    if opens_b > 0 or opens_c > 0:
        # Trim any dangling incomplete key/value before closing.
        frag = frag.rstrip().rstrip(',').rstrip()
        frag += '}' * max(opens_c, 0) + ']' * max(opens_b, 0)
        try:
            items = json.loads(frag)
            if isinstance(items, list):
                return items
        except json.JSONDecodeError:
            pass
    return None


def _extract_memories_from_response(response) -> list:
    """Extract the JSON memory list from an LLM response, handling truncation."""
    if response is None or not getattr(response, "choices", None):
        return []
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    items = _parse_json_array_robust(text)
    if items is not None:
        return items
    # If the response was truncated (finish_reason == 'length'), the JSON may
    # be cut mid-item. Attempt a second, more aggressive repair: drop the last
    # incomplete object (everything after the last complete ``}`` followed by
    # ``,`` or ``]``) and re-close.
    if getattr(choice, "finish_reason", None) == "length":
        frag = text[text.find('['):]
        # Keep up to the last balanced '}' that closes a top-level item.
        last_close = frag.rfind('}')
        if last_close > 0:
            frag = frag[:last_close + 1].rstrip().rstrip(',') + ']'
            try:
                items = json.loads(frag)
                if isinstance(items, list):
                    return items
            except json.JSONDecodeError:
                pass
    return []


def extract_memories(messages: list) -> None:
    """Extract new memories from the recent conversation after a turn."""
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", "")) for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)
    if not dialogue.strip():
        return
    existing = list_memory_files()
    existing_desc = "\n".join(f"- {m['name']}: {m['description']}" for m in existing) if existing else "(none)"
    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body, expires_at}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "- expires_at: optional TTL deadline in YYYYMMDDTHHMMSS format. "
        "Set this ONLY for volatile project facts that will become stale "
        "(e.g. a temporary branch name, a sprint-specific task). "
        "Omit or set to null for permanent preferences/feedback.\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            timeout=60,
        )
    except Exception:
        return
    items = _extract_memories_from_response(response)
    if not items:
        return
    count = 0
    for mem in items:
        name = mem.get("name", f"memory_{int(time.time())}")
        mem_type = mem.get("type", "user")
        desc = mem.get("description", "")
        body = mem.get("body", "")
        expires_at = mem.get("expires_at") or None
        if desc and body:
            _write_memory_file_no_index(name, mem_type, desc, body,
                                        expires_at=expires_at)
            count += 1
    if count:
        _rebuild_index()
        print(f"\n\033[33m[Memory: extracted {count} new memories]\033[0m")


def consolidate_memories() -> None:
    """Merge duplicate/stale memories (triggered when file count >= threshold).

    Atomicity contract:
      All new memory files are written to a temporary directory first; only
      after every file is successfully written is the old directory swapped
      out (renamed to a backup) and the temp promoted. If any step fails the
      backup is restored, so the memory store is never left in a half-written
      state.
    """
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return
    catalog = "\n\n".join(
        f"## {f['filename']}\n"
        f"name: {f['name']}\ndescription: {f['description']}\n"
        f"type: {f['type']}\ncreated_at: {f.get('created_at','?')}\n"
        f"updated_at: {f.get('updated_at','?')}\n"
        f"hit_count: {f.get('hit_count',0)}\nlast_used: {f.get('last_used','?')}\n"
        f"expires_at: {f.get('expires_at') or 'never'}"
        f"{' [EXPIRED]' if is_expired(f) else ''}"
        f"{' [DEAD: never used, stale]' if is_dead_memory(f) else ''}\n"
        f"{f['body']}"
        for f in files
    )
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "5. When two memories conflict, keep the one with the later "
        "updated_at (newer wins)\n"
        "6. Prefer memories with high hit_count or recent last_used; "
        "memories with hit_count 0 and old last_used are candidates for removal\n"
        "7. Memories marked [EXPIRED] or [DEAD] should be removed first "
        "(unless they encode permanent user preferences)\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8000,
            timeout=120,
        )
    except Exception:
        return
    items = _extract_memories_from_response(response)
    if not items:
        print("\n\033[33m[Memory: consolidation produced no items, "
              "keeping originals]\033[0m")
        return

    # ---- Atomic swap: write everything to a temp dir, then swap. ----
    ts = int(time.time())
    backup_dir = MEMORY_DIR.parent / f".memory_backup_{ts}"
    temp_dir = MEMORY_DIR.parent / f".memory_tmp_{ts}"
    try:
        temp_dir.mkdir(exist_ok=False)
        # Copy the MEMORY.md index so the temp dir is self-consistent.
        if MEMORY_INDEX.exists():
            shutil.copy2(MEMORY_INDEX, temp_dir / "MEMORY.md")
        written = 0
        for mem in items:
            name = mem.get("name", f"memory_{ts}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                _write_memory_file_no_index(name, mem_type, desc, body, temp_dir)
                written += 1
        if written == 0:
            print("\n\033[33m[Memory: consolidation produced no valid "
                  "memories, keeping originals]\033[0m")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return
        # Rebuild the index inside the temp dir before promoting.
        _rebuild_index_in(temp_dir)
        # Swap: rename current -> backup, temp -> live.
        shutil.move(str(MEMORY_DIR), str(backup_dir))
        try:
            shutil.move(str(temp_dir), str(MEMORY_DIR))
        except Exception:
            # Promote failed: roll back immediately.
            shutil.move(str(backup_dir), str(MEMORY_DIR))
            raise
        # Success -- remove the backup.
        shutil.rmtree(backup_dir, ignore_errors=True)
        print(f"\n\033[33m[Memory: consolidated {len(files)} -> "
              f"{written} memories]\033[0m")
    except Exception:
        # Last-resort safety: ensure MEMORY_DIR exists and is populated.
        if not MEMORY_DIR.exists():
            if backup_dir.exists():
                shutil.move(str(backup_dir), str(MEMORY_DIR))
            else:
                MEMORY_DIR.mkdir(exist_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _rebuild_index_in(directory: Path) -> None:
    """Rebuild the MEMORY.md index inside *directory* (used by atomic swap)."""
    _rebuild_index(directory)


def _post_turn_memory(messages_snapshot: list) -> None:
    """Run after each turn: extract memories, clean stale, consolidate."""
    try:
        if not ctx.memory_lock.acquire(timeout=ctx.memory_lock_timeout):
            return
        try:
            extract_memories(messages_snapshot)
            removed = cleanup_stale_memories()
            if removed:
                print(f"\n\033[33m[Memory: cleaned {removed} stale "
                      f"memories]\033[0m")
            consolidate_memories()
        finally:
            ctx.memory_lock.release()
    except Exception:
        pass


def _load_memories_async(messages: list):
    """Load relevant memories in a background thread; return a ``["", thread]`` holder."""
    messages_snapshot = list(messages)
    holder = ["", None]

    def _worker():
        try:
            holder[0] = load_memories(messages_snapshot)
        except Exception:
            holder[0] = ""

    t = threading.Thread(target=_worker, daemon=True, name="memory-load")
    holder[1] = t
    t.start()
    return holder


def _await_memories(holder) -> str:
    """Wait for async memory loading to finish and return the result."""
    thread = holder[1]
    if thread is not None:
        thread.join(timeout=60)
    return holder[0]

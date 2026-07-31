"""Filesystem & shell tools."""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from pathlib import Path

from .config import (WORKDIR, BASH_TIMEOUT, _BG_OUTPUT_DIR,
                     _IS_WINDOWS, _POPEN_KWARGS)
from .exceptions import AgentInterrupt
from .utils import parse_bg_command, parse_explicit_timeout


def safe_path(p: str) -> Path:
    """Resolve a relative path to an absolute path inside the workspace; raise ValueError if it escapes."""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill a process tree cross-platform."""
    if proc.poll() is not None:
        return
    try:
        if _IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True, timeout=10,
            )
        else:
            import signal
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _run_background(command: str, log_name: str | None) -> str:
    """Run a command as a background process, redirecting output to a log file."""
    _BG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not log_name:
        log_name = f"bg_{int(time.time() * 1000)}.log"
    log_path = _BG_OUTPUT_DIR / log_name
    log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
    try:
        proc = subprocess.Popen(
            command, shell=True, cwd=WORKDIR,
            stdout=log_fh, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            **_POPEN_KWARGS,
        )
    finally:
        log_fh.close()
    rel = log_path.relative_to(WORKDIR).as_posix() if log_path.is_relative_to(WORKDIR) else str(log_path)
    return (
        f"[background] PID={proc.pid}\n"
        f"  log: {rel}\n"
        f"  read:use read_file('{rel}') to check output\n"
        f"  stop:run bash 'taskkill /PID {proc.pid} /T /F' (Win) or 'kill {proc.pid}' (Linux)\n"
        f"  status: run bash 'tasklist /FI \"PID eq {proc.pid}\"' (Win) or 'ps -p {proc.pid}' (Linux)"
    )


def run_bash(command: str) -> str:
    """Run a shell command; supports ``bg:`` prefix for background and ``# timeout=N`` for explicit timeout."""
    try:
        is_bg, log_name, bg_cmd = parse_bg_command(command)
        if is_bg:
            return _run_background(bg_cmd, log_name)

        timeout, command = parse_explicit_timeout(command)

        proc = subprocess.Popen(
            command, shell=True, cwd=WORKDIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, encoding='utf-8', errors='replace',
            **_POPEN_KWARGS,
        )
        output_lines: list[str] = []
        timed_out = False
        import queue as _queue
        out_q: _queue.Queue[str | None] = _queue.Queue()

        def _reader():
            try:
                for line in proc.stdout:
                    out_q.put(line)
            except Exception:
                pass
            finally:
                out_q.put(None)

        reader_t = threading.Thread(target=_reader, daemon=True, name="bash-reader")
        reader_t.start()

        try:
            deadline = time.monotonic() + timeout
            eof = False
            while not eof:
                try:
                    item = out_q.get(timeout=0.1)
                except _queue.Empty:
                    item = "__EMPTY__"

                if item is None:
                    eof = True
                elif item != "__EMPTY__":
                    output_lines.append(item)

                if not eof and time.monotonic() > deadline:
                    timed_out = True
                    break

                if not eof and proc.poll() is not None:
                    drained_any = False
                    while True:
                        try:
                            item2 = out_q.get(timeout=0.05)
                        except _queue.Empty:
                            break
                        if item2 is None:
                            eof = True
                            break
                        output_lines.append(item2)
                        drained_any = True
                    if not drained_any:
                        eof = True
        except AgentInterrupt:
            _kill_process_tree(proc)
            raise

        if timed_out:
            _kill_process_tree(proc)
            out = "".join(output_lines).strip()
            preview = out[:50000] if out else "(no output yet)"
            return (
                f"[timeout after {timeout}s - process killed]\n"
                f"--- output so far ---\n{preview}\n"
                f"--- tip: use 'bg: <command>' prefix to run in background, or '# timeout=N' for longer ---"
            )

        out = "".join(output_lines).strip()
        return out[:50000] if out else "(no output)"
    except AgentInterrupt:
        raise
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"
    except UnicodeDecodeError as e:
        return f"Error: Encoding issue - {e}"


def run_read(path: str, offset: int | None = None, limit: int | None = None) -> str:
    """Read file contents with line-number prefixes and pagination."""
    DEFAULT_LIMIT = 2000
    MAX_LIMIT = 5000
    offset = max(1, offset or 1)
    limit = min(max(1, limit or DEFAULT_LIMIT), MAX_LIMIT)
    try:
        file_path = safe_path(path)
        if file_path.is_dir():
            return f"Error: {path} is a directory, not a file. Use glob or bash ls to list contents."
        with open(file_path, "rb") as bf:
            head = bf.read(4096)
        if b"\x00" in head:
            size = file_path.stat().st_size
            return f"<Binary file> {size} bytes -cannot display as text"
        text = None
        for enc in ("utf-8-sig", "utf-8", "gbk", "latin-1"):
            try:
                text = file_path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        total = len(lines)
        if total == 0:
            return "(empty file)"
        start = offset - 1
        if start >= total:
            return f"(offset {offset} is beyond end of file; file has {total} lines)"
        end = min(start + limit, total)
        selected = lines[start:end]
        width = len(str(end))
        numbered = [f"{str(offset + i).rjust(width)}->{line}" for i, line in enumerate(selected)]
        parts = []
        if start > 0 and end < total:
            parts.append(f"(showing lines {offset}-{end} of {total}; {start} lines above omitted, {total - end} below)")
        elif start > 0:
            parts.append(f"(showing lines {offset}-{end} of {total}; {start} lines above omitted)")
        elif end < total:
            parts.append(f"(showing lines {offset}-{end} of {total})")
        if end < total:
            remaining = total - end
            numbered.append(f"... ({remaining} more lines below; use offset={end + 1} to continue)")
        body = "\n".join(numbered)
        header = " ".join(parts)
        return f"{header}\n{body}" if header else body
    except Exception as e:
        return f"Error:{e}"


def run_write(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """Replace exact text in a file (first match)."""
    try:
        file_path = safe_path(path)
        text = file_path.read_text(encoding='utf-8', errors='replace')
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1), encoding='utf-8')
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"


def run_glob(pattern: str) -> str:
    """Find files matching a glob pattern, sorted by mtime descending, limited to 100 results."""
    import glob as g
    GLOB_MAX = 100
    try:
        workdir = WORKDIR.resolve()
        seen = set()
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR, recursive=True):
            resolved = (WORKDIR / match).resolve()
            if not resolved.is_relative_to(workdir):
                continue
            rel = resolved.relative_to(workdir).as_posix()
            if rel in seen:
                continue
            seen.add(rel)
            try:
                mtime = resolved.stat().st_mtime
            except OSError:
                mtime = 0.0
            results.append((mtime, rel))
        results.sort(key=lambda t: t[0], reverse=True)
        if len(results) > GLOB_MAX:
            top = results[:GLOB_MAX]
            out = "\n".join(r for _, r in top)
            out += f"\n... ({len(results) - GLOB_MAX} more matches, refine pattern)"
            return out
        return "\n".join(r for _, r in results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def run_grep(pattern: str, path: str = ".", include: str | None = None) -> str:
    """Recursively search file contents for a regex pattern."""
    try:
        search_root = safe_path(path)
        if not search_root.exists():
            return f"Error: path not found: {path}"
        regex = re.compile(pattern)
        results = []
        workdir = WORKDIR.resolve()
        if include:
            import fnmatch
        if search_root.is_file():
            if include and not fnmatch.fnmatch(search_root.name, include):
                return "(no matches)"
            files_to_search = [search_root]
        else:
            files_to_search = []
            for p in search_root.rglob("*"):
                if p.is_file() and p.resolve().is_relative_to(workdir):
                    if include and not fnmatch.fnmatch(p.name, include):
                        continue
                    files_to_search.append(p)
        for fpath in files_to_search:
            try:
                text = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    try:
                        rel = fpath.resolve().relative_to(workdir)
                    except ValueError:
                        rel = fpath
                    results.append(f"{rel}:{lineno}: {line.rstrip()}")
                    if len(results) >= 50:
                        results.append("...(truncated at 50 matches)")
                        return "\n".join(results)
        return "\n".join(results) if results else "(no matches)"
    except re.error as e:
        return f"Error: invalid regex: {e}"
    except Exception as e:
        return f"Error:{e}"

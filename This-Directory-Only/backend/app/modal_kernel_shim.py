"""
Runs INSIDE a Modal sandbox, never in the API process.

One real IPython kernel per sandbox, driven over the sandbox's stdin/stdout:
one JSON object per line in, one JSON object per line out. This is what lets
a notebook cell say `%pip install …` or `!wget …` and mean it — the fork
runner exec()s source, and a line magic is a SyntaxError to exec().

Wire format:
  in   {"op": "exec", "code": str, "timeout": int}
       {"op": "shutdown"}
  out  {"ready": true}                                     once, at start
       {"stdout": str, "stderr": str, "success": bool,
        "outputs": [{mime: data, ...}, ...], "timed_out": bool}

`outputs` are the kernel's display_data / execute_result mimebundles — a
plotly figure, a PNG, an HTML table — which is exactly the part of a notebook
the fork runner had no channel for. Bytes are capped per cell so one
`fig.show()` of a 100 MB tensor cannot wedge the pipe.

stdlib + jupyter_client only. Nothing from `app.*` is importable here, and
nothing should be: this file is copied into the image by path.
"""

from __future__ import annotations

import json
import os
import queue
import re
import sys
import time

from jupyter_client import KernelManager

# The protocol channel is the process's ORIGINAL stdout, duplicated before the
# kernel exists; fd 1 itself is then pointed at stderr. The kernel inherits
# fd 1, and anything that bypasses IPython's stream capture — `os.system`,
# a C library's printf, a subprocess with inherited stdout — would otherwise
# land in the middle of a JSON reply and read as "unreadable reply" upstream.
# 🔴 Must run before KernelManager.start_kernel().
_PROTO = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

# Mimetypes worth sending back. Anything else (application/json, vnd.jupyter
# widget state, …) is dropped rather than shipped to a client that cannot
# draw it.
KEEP_MIMES = (
    "application/vnd.plotly.v1+json",
    "text/html",
    "image/png",
    "image/svg+xml",
    "text/markdown",
    "text/plain",
)
MAX_OUTPUT_BYTES = 8 * 1024 * 1024
# A cell that ignores SIGINT gets this long to unwind before the kernel is
# declared dead.
INTERRUPT_GRACE_SECONDS = 5.0

# The client's cell harness (`practice/notebook.js`) compiles learner source
# under `<cell N>`; its own frames are `<harness>`. Plain-mode IPython
# tracebacks list a frame as `  File "<name>", line N, in fn` followed by one
# source line — both are dropped for internal names, the same trim the fork
# runner's `_user_traceback` does.
_INTERNAL_FRAMES = ("<harness>", "<preamble>", "<ipython-input", "<string>")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# Both spellings IPython uses: `File "<x>", line 3` and `File <x>:3, in f`.
_FRAME_RE = re.compile(r'^\s+(?:File "?(<[^>"]*>|[^":,]*)"?(?::\d+|, line \d+)|Cell In\[\d+\], line \d+)')


def _clean_traceback(lines: list[str]) -> str:
    out: list[str] = []
    skip_next = False
    for raw in lines:
        for line in _ANSI.sub("", raw).split("\n"):
            if skip_next:
                skip_next = False
                if not _FRAME_RE.match(line):
                    continue
            m = _FRAME_RE.match(line)
            # `Cell In[N]` is IPython's name for the top-level input — here
            # always the harness call itself, never the learner's code.
            if m and (m.group(1) is None or m.group(1).startswith(_INTERNAL_FRAMES)):
                skip_next = True
                continue
            out.append(line)
    # Frames compiled from a string have no source line to show; the blank
    # lines IPython leaves for them collapse to one.
    text = re.sub(r"\n{2,}", "\n", "\n".join(out)).strip("\n")
    return text + "\n" if text else ""


def _bundle(data: dict) -> tuple[dict, int]:
    kept: dict = {}
    size = 0
    for mime in KEEP_MIMES:
        if mime not in data:
            continue
        value = data[mime]
        # jupyter_client already decodes JSON mimetypes to objects; sizes are
        # measured on the wire form the API will forward.
        encoded = value if isinstance(value, str) else json.dumps(value)
        size += len(encoded)
        kept[mime] = value
    return kept, size


class Shim:
    def __init__(self) -> None:
        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=120)
        # Plain tracebacks are the ones `_clean_traceback` knows how to trim;
        # the default "Context" mode interleaves source windows.
        self._run("%xmode Plain", timeout=30)

    def _run(self, code: str, timeout: int) -> dict:
        msg_id = self.kc.execute(code, allow_stdin=False, store_history=False)
        result = {"stdout": "", "stderr": "", "success": True, "outputs": [],
                  "timed_out": False}
        deadline = time.monotonic() + timeout
        interrupted_at: float | None = None
        out_bytes = 0
        while True:
            now = time.monotonic()
            if interrupted_at is not None:
                wait = interrupted_at + INTERRUPT_GRACE_SECONDS - now
                if wait <= 0:
                    result["success"] = False
                    result["stderr"] += (
                        f"Cell timed out after {timeout} seconds and did not respond "
                        "to an interrupt; the session was restarted.\n")
                    result["dead"] = True
                    return result
            else:
                wait = deadline - now
                if wait <= 0:
                    self.km.interrupt_kernel()
                    interrupted_at = time.monotonic()
                    result["timed_out"] = True
                    result["stderr"] += (
                        f"Cell timed out after {timeout} seconds and was interrupted. "
                        "The session is still alive.\n")
                    continue
            try:
                msg = self.kc.get_iopub_msg(timeout=max(wait, 0.05))
            except queue.Empty:
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "stream":
                key = "stdout" if content.get("name") == "stdout" else "stderr"
                result[key] += content.get("text", "")
            elif kind in ("display_data", "execute_result"):
                bundle, size = _bundle(content.get("data") or {})
                if not bundle:
                    continue
                out_bytes += size
                if out_bytes > MAX_OUTPUT_BYTES:
                    result["stderr"] += (
                        "[output truncated: this cell produced more than "
                        f"{MAX_OUTPUT_BYTES // (1024 * 1024)} MB of rich output]\n")
                    continue
                result["outputs"].append(bundle)
            elif kind == "error":
                result["success"] = False
                tb = _clean_traceback(content.get("traceback") or [])
                if not tb:
                    tb = f'{content.get("ename", "Error")}: {content.get("evalue", "")}\n'
                result["stderr"] += tb
            elif kind == "status" and content.get("execution_state") == "idle":
                return result

    def serve(self) -> None:
        _PROTO.write(json.dumps({"ready": True}) + "\n")
        _PROTO.flush()
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if message.get("op") == "shutdown":
                break
            result = self._run(message.get("code") or "",
                               int(message.get("timeout") or 30))
            _PROTO.write(json.dumps(result) + "\n")
            _PROTO.flush()
            if result.get("dead"):
                break
        self.km.shutdown_kernel(now=True)


if __name__ == "__main__":
    Shim().serve()

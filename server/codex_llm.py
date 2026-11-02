"""Codex CLI as the language model (talk + Code-as-Policies). Uses the local `codex` login."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("codex")
CODEX = shutil.which("codex") or "/opt/homebrew/bin/codex"
MODEL = os.environ.get("GOBBLET_CODEX_MODEL", os.environ.get("STACKCHAN_CODEX_MODEL", "gpt-5.6-terra"))
WORKDIR = Path(__file__).resolve().parents[1] / "out" / "codex"


def ask(prompt: str, timeout: float = 90.0, image: Path | None = None) -> str | None:
    """One-shot Codex answer (no tools needed, read-only sandbox). None on failure/timeout."""
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("r", suffix=".txt", delete=False, dir=WORKDIR) as f:
        out_path = Path(f.name)
    cmd = [CODEX, "exec", prompt, "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
           "--model", MODEL, "-o", str(out_path)]
    if image is not None:
        cmd += ["--image", str(image)]
    try:
        r = subprocess.run(cmd, cwd=WORKDIR, capture_output=True, text=True, timeout=timeout,
                           stdin=subprocess.DEVNULL)
        if r.returncode != 0:
            log.warning("codex failed: %s", r.stderr[-500:])
            return None
        text = out_path.read_text().strip()
        return text or None
    except subprocess.TimeoutExpired:
        log.warning("codex timed out after %.0fs", timeout)
        return None
    finally:
        out_path.unlink(missing_ok=True)

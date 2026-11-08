"""tools.subprocess_utils.run_with_timeout — timeout must not hang on Windows.

Reproduces the `npx.cmd` pattern without Node: a Python *parent* (standing in
for cmd.exe) spawns a sleeping *child* (standing in for node) that inherits the
stdout/stderr pipes. Plain ``subprocess.run(timeout=...)`` on Windows kills
only the parent and then blocks in ``communicate()`` until the orphaned child
exits.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from tools import subprocess_utils
from tools.subprocess_utils import run_with_timeout

windows_only = pytest.mark.skipif(os.name != "nt", reason="Windows .cmd shim behaviour")


def _parent_with_orphanable_child(pid_file: Path, child_sleep: float) -> list[str]:
    script = textwrap.dedent(
        f"""
        import subprocess, sys, time
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep({child_sleep})"]
        )  # inherits our stdout/stderr pipes, like node under npx.cmd
        open({str(pid_file)!r}, "w").write(str(child.pid))
        print("parent-started", flush=True)
        time.sleep({child_sleep})
        """
    )
    return [sys.executable, "-c", script]


def _pid_alive(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    return str(pid) in out


def _wait_for(path: Path, deadline: float = 10.0) -> None:
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if path.exists() and path.read_text().strip():
            return
        time.sleep(0.05)
    raise AssertionError(f"{path} never written")


@windows_only
def test_plain_subprocess_run_blocks_on_orphaned_grandchild(tmp_path):
    """Control: the bug is real — stdlib run() waits for the orphan."""
    pid_file = tmp_path / "child.pid"
    child_sleep = 6
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            _parent_with_orphanable_child(pid_file, child_sleep),
            capture_output=True,
            text=True,
            timeout=1,
        )
    elapsed = time.monotonic() - start
    # Timeout was 1s, yet run() only returned once the orphan exited.
    assert elapsed >= child_sleep - 1.5, elapsed


@windows_only
def test_run_with_timeout_returns_promptly_and_kills_tree(tmp_path):
    pid_file = tmp_path / "child.pid"
    timeout = 2
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run_with_timeout(
            _parent_with_orphanable_child(pid_file, 120),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    elapsed = time.monotonic() - start
    assert elapsed < timeout + subprocess_utils.DEFAULT_KILL_GRACE, elapsed
    assert excinfo.value.timeout == timeout
    # Output produced before the kill is preserved on the exception.
    assert "parent-started" in (excinfo.value.stdout or "")

    _wait_for(pid_file)
    child_pid = int(pid_file.read_text())
    assert not _pid_alive(child_pid), "grandchild survived the tree kill"


@windows_only
def test_run_with_timeout_bounded_even_if_pipes_stay_held(tmp_path, monkeypatch):
    """If the tree kill misses a descendant, the drain is still bounded."""
    pid_file = tmp_path / "child.pid"
    monkeypatch.setattr(subprocess_utils, "kill_process_tree", lambda proc, **_: proc.kill())
    start = time.monotonic()
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            run_with_timeout(
                _parent_with_orphanable_child(pid_file, 30),
                capture_output=True,
                timeout=1,
                kill_grace=1,
            )
        assert time.monotonic() - start < 5
    finally:
        if pid_file.exists():
            subprocess.run(
                ["taskkill", "/F", "/PID", pid_file.read_text().strip()],
                capture_output=True,
                timeout=10,
            )


@pytest.mark.skipif(os.name != "nt", reason="exercises the Windows Popen path")
def test_run_with_timeout_success_and_check_semantics():
    ok = run_with_timeout(
        [sys.executable, "-c", "import sys; print('hi'); sys.stderr.write('err')"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert ok.returncode == 0
    assert ok.stdout.strip() == "hi"
    assert ok.stderr == "err"

    echoed = run_with_timeout(
        [sys.executable, "-c", "import sys; print(sys.stdin.read().upper())"],
        input="abc",
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert echoed.stdout.strip() == "ABC"

    with pytest.raises(subprocess.CalledProcessError) as excinfo:
        run_with_timeout(
            [sys.executable, "-c", "import sys; print('out'); sys.exit(3)"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
    assert excinfo.value.returncode == 3
    assert excinfo.value.stdout.strip() == "out"


def test_delegates_to_subprocess_run_without_timeout_or_off_windows(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(subprocess_utils.subprocess, "run", fake_run)

    run_with_timeout(["x"], capture_output=True, text=True)
    assert calls[-1][1]["timeout"] is None

    monkeypatch.setattr(subprocess_utils, "_IS_WINDOWS", False)
    run_with_timeout(["y"], capture_output=True, timeout=7, check=True)
    assert calls[-1] == (
        ["y"],
        {"input": None, "capture_output": True, "timeout": 7, "check": True},
    )


def test_probe_cli_uses_tree_kill_runner_and_caches_timeout(monkeypatch):
    from tools.video import hyperframes_compose as hc

    calls = []

    def fake_runner(args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout"))

    monkeypatch.setattr(hc, "run_with_timeout", fake_runner)
    monkeypatch.setattr(hc.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(hc.HyperFramesCompose, "_cli_probe_cache", None)

    first = hc.HyperFramesCompose._probe_cli()
    second = hc.HyperFramesCompose._probe_cli()

    assert "timed out" in first["error"]
    assert second == first
    assert len(calls) == 1, "a timed-out doctor must not re-run on every get_info()"
    assert calls[0][1:] == ["--yes", "hyperframes", "doctor", "--json"]

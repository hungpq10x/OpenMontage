"""Timeout-safe subprocess execution for Windows ``.cmd`` shims.

On Windows, ``npx``/``npm``/``mmdc`` resolve to ``*.cmd`` batch wrappers, so
``subprocess.run([npx, ...], timeout=...)`` actually starts ``cmd.exe``, which
starts ``node``. When the timeout fires, ``subprocess.run`` kills only
``cmd.exe``; the ``node`` grandchild survives as an orphan that still holds
the stdout/stderr pipes, and the follow-up ``communicate()`` that ``run()``
performs to collect output blocks until that orphan exits — possibly forever.

:func:`run_with_timeout` is a drop-in replacement for ``subprocess.run`` that,
on Windows and only when a ``timeout`` is given, kills the whole process tree
(``taskkill /T /F``) while the shell is still alive (so its children are still
linked to it), then drains the pipes with a bound. On POSIX, or without a
timeout, it simply delegates to ``subprocess.run`` — behaviour is unchanged.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any, Optional, Sequence, Union

__all__ = ["run_with_timeout", "kill_process_tree"]

# How long to wait for pipes to drain after the tree kill before giving up on
# the remaining output.
DEFAULT_KILL_GRACE = 5.0

_IS_WINDOWS = os.name == "nt"


def kill_process_tree(proc: subprocess.Popen, *, timeout: float = 10.0) -> None:
    """Forcefully kill ``proc`` and all of its descendants. Never raises."""
    if _IS_WINDOWS:
        try:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError):
            pass
    # Belt and braces: make sure at least the direct child is gone (also the
    # whole story on POSIX, where there is no shim layer to worry about).
    try:
        proc.kill()
    except OSError:
        pass


def run_with_timeout(
    args: Union[str, Sequence[Any]],
    *,
    timeout: Optional[float] = None,
    input: Optional[Any] = None,
    capture_output: bool = False,
    check: bool = False,
    kill_grace: float = DEFAULT_KILL_GRACE,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    """Like :func:`subprocess.run`, but a timeout cannot hang on Windows.

    Raises :class:`subprocess.TimeoutExpired` within roughly
    ``timeout + kill_grace`` seconds even if a grandchild ignores the kill of
    its parent shell. Accepts the same keyword arguments as ``subprocess.run``.
    """
    if timeout is None or not _IS_WINDOWS:
        return subprocess.run(
            args,
            input=input,
            capture_output=capture_output,
            timeout=timeout,
            check=check,
            **kwargs,
        )

    if input is not None:
        if kwargs.get("stdin") is not None:
            raise ValueError("stdin and input arguments may not both be used.")
        kwargs["stdin"] = subprocess.PIPE
    if capture_output:
        if kwargs.get("stdout") is not None or kwargs.get("stderr") is not None:
            raise ValueError(
                "stdout and stderr arguments may not be used with capture_output."
            )
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE

    # Deliberately not `with Popen(...)`: its __exit__ closes the pipes and
    # waits, which is exactly what can block when an orphan still holds them.
    proc = subprocess.Popen(args, **kwargs)
    try:
        stdout, stderr = proc.communicate(input, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        stdout = stderr = None
        try:
            # Collect whatever was produced before the kill. With the tree
            # gone the pipes hit EOF promptly; the bound covers anything
            # taskkill could not reach.
            stdout, stderr = proc.communicate(timeout=kill_grace)
        except subprocess.TimeoutExpired:
            # Some descendant still holds the pipes. Abandon them: the
            # reader threads are daemons and the handles are released when
            # that process finally exits. Closing them here could block.
            pass
        except (OSError, ValueError):
            pass
        raise subprocess.TimeoutExpired(
            proc.args, timeout, output=stdout, stderr=stderr
        ) from None
    except BaseException:
        # KeyboardInterrupt etc.: don't leave the tree running.
        kill_process_tree(proc)
        raise

    retcode = proc.poll()
    if retcode is None:  # pragma: no cover - communicate() returned, so it exited
        retcode = proc.wait()
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass

    if check and retcode:
        raise subprocess.CalledProcessError(
            retcode, proc.args, output=stdout, stderr=stderr
        )
    return subprocess.CompletedProcess(proc.args, retcode, stdout, stderr)

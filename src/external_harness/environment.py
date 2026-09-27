"""Where the agent's commands run: a clean environment, optionally inside a bubblewrap sandbox.

mini-SWE-agent's own LocalEnvironment passes the whole process environment to every command, so an
agent could read API keys or the paths of other data with ``env``. Here every command gets only the
variables the runner lists (the knowledge-base index, PATH, a home inside the run folder).

With ``sandbox: bubblewrap`` each command also runs in a fresh namespace: no network, the run folder
is the only writable place, and the only other visible folders are the system (read-only), the Python
runtime of this package and the index. That is what makes the baseline fair: the agent cannot read
the gold set, other runs or anything else on the machine. Without bubblewrap the commands run in the
run folder but can read what the user can; ``outside_paths`` in result.json flags commands that look
outside it.
"""

from __future__ import annotations

import contextlib
import os
import platform
import re
import signal
import subprocess
from pathlib import Path
from typing import Any, Literal

from minisweagent.environments.local import LocalEnvironment
from pydantic import BaseModel

SYSTEM_READ_ONLY = ("/usr", "/bin", "/lib", "/lib64", "/lib32", "/etc/alternatives", "/etc/ld.so.cache",
                    "/etc/localtime", "/etc/ssl", "/etc/passwd", "/etc/group", "/etc/nsswitch.conf")


class KbEnvironmentConfig(BaseModel):
    cwd: str
    env: dict[str, str]                       # the complete environment of every command
    timeout: int = 60
    sandbox: Literal["none", "bubblewrap"] = "none"
    read_only: list[str] = []                 # folders visible (read-only) inside the sandbox
    bwrap: str = "bwrap"


class KbEnvironment(LocalEnvironment):
    """LocalEnvironment with a clean environment, an optional bubblewrap sandbox, and a handle on the
    running command so a stopped run can kill it (commands run in their own session)."""

    def __init__(self, **kwargs: Any) -> None:
        self.config = KbEnvironmentConfig(**kwargs)
        self.current: subprocess.Popen | None = None

    def argv(self, command: str) -> list[str]:
        if self.config.sandbox == "none":
            return ["bash", "-c", command]
        work = str(Path(self.config.cwd).resolve())
        args = [self.config.bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
        for path in SYSTEM_READ_ONLY:
            args += ["--ro-bind-try", path, path]
        # A private /tmp first: the run folder and the index may live under /tmp and are bound after it.
        args += ["--tmpfs", "/tmp", "--proc", "/proc", "--dev", "/dev"]
        for path in dict.fromkeys(str(Path(item).resolve()) for item in self.config.read_only):
            args += ["--ro-bind", path, path]
        args += ["--bind", work, work, "--chdir", work]
        for key, value in self.config.env.items():
            args += ["--setenv", key, value]
        return [*args, "bash", "-c", command]

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        command = action.get("command", "")
        try:
            process = subprocess.Popen(
                self.argv(command), cwd=self.config.cwd, env=self.config.env if self.config.sandbox == "none" else {},
                text=True, encoding="utf-8", errors="replace", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            self.current = process
            try:
                stdout, _ = process.communicate(timeout=timeout or self.config.timeout)
            except subprocess.TimeoutExpired:
                self.kill_current()
                stdout, _ = process.communicate()
                raise subprocess.TimeoutExpired(command, timeout or self.config.timeout, output=stdout) from None
            output = {"output": stdout, "returncode": process.returncode, "exception_info": ""}
        except Exception as exc:
            raw = getattr(exc, "output", None)
            raw = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else (raw or "")
            output = {"output": raw, "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {exc}",
                      "extra": {"exception_type": type(exc).__name__, "exception": str(exc)}}
        finally:
            self.current = None
        self._check_finished(output)
        return output

    def kill_current(self) -> None:
        process = self.current
        if process is not None and process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        # Unlike LocalEnvironment, never the process environment (it may hold keys).
        return {**self.config.model_dump(exclude={"env"}), **platform.uname()._asdict(), **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config.model_dump(mode="json", exclude={"env"}),
                                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}"}}}


_ABSOLUTE = re.compile(r"(?:^|[\s'\"=<>|;&(])(/[A-Za-z0-9_.\-/]+)")
_ALLOWED_ABSOLUTE = ("/dev/null", "/dev/stdin", "/dev/stdout", "/dev/stderr", "/tmp", "/usr/bin/env")


def outside_paths(commands: list[str], work: Path) -> list[str]:
    """Commands that name a path outside the run folder or read the environment: an audit trail for runs
    without the sandbox (with it, such commands find nothing)."""
    flagged = []
    root = str(work.resolve())
    for command in commands:
        paths = [path for path in _ABSOLUTE.findall(command)
                 if not path.startswith(root) and not path.startswith(_ALLOWED_ABSOLUTE)]
        if paths or re.search(r"(^|\s|/)\.\.(/|\s|$)", command) or re.search(r"(^|[;&|]\s*)(env|printenv)\b", command):
            flagged.append(command[:200])
    return flagged

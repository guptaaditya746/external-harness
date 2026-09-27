"""One question through mini-SWE-agent, with the knowledge-base commands on PATH.

The runner owns everything around the agent loop and nothing inside it: it prepares a working
folder, renders the prompts, runs mini-SWE-agent's DefaultAgent unchanged (a subclass only times
model calls and commands for the event stream), then checks answer.json against the contract and
writes result.json next to the full trajectory.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time
import uuid
from collections.abc import Callable
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from . import contract, kb
from .config import Config

Emit = Callable[[dict[str, Any]], None]

OBSERVATION = (
    "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
    "<returncode>{{output.returncode}}</returncode>\n"
    "{% if output.output | length < 8000 %}<output>\n{{ output.output }}</output>"
    "{% else %}<output_head>\n{{ output.output[:5000] }}</output_head>\n"
    "<elided_chars>{{ output.output | length - 7000 }} characters elided</elided_chars>\n"
    "<output_tail>\n{{ output.output[-2000:] }}</output_tail>{% endif %}"
)
TEXT_FORMAT = """

Every reply must contain exactly one bash command in a block like this, after a short explanation:

```mswea_bash_command
kbsearch "MethodX BenchY" -k 10
```
"""


def _template(value: str, name: str) -> str:
    if value == "default":
        return resources.files("external_harness").joinpath("prompts", f"{name}.j2").read_text(encoding="utf-8")
    return Path(value).read_text(encoding="utf-8")


def _usage(message: dict[str, Any]) -> dict[str, int]:
    usage = ((message.get("extra") or {}).get("response") or {}).get("usage") or {}
    return {"prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0)}


def run(config: Config, question: str, *, run_id: str | None = None, out: Path | None = None,
        emit: Emit | None = None, model: Any | None = None) -> dict[str, Any]:
    """Answer ``question``; returns the result document (also written to ``<run folder>/result.json``).
    ``model`` replaces the configured model (tests pass mini-SWE-agent's DeterministicModel)."""
    started = time.monotonic()
    run_id = run_id or f"x-{uuid.uuid4().hex[:8]}"
    folder = (out or config.environment.workdir_root / run_id).resolve()
    work = folder / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "answer.json").unlink(missing_ok=True)          # never score an earlier run's answer
    # mini-SWE-agent reads a global .env from its config folder: point it at an empty one of ours.
    os.environ["MSWEA_GLOBAL_CONFIG_DIR"] = str(folder / ".mini-swe-agent")
    os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
    os.environ.setdefault("MSWEA_COST_TRACKING", config.model.cost_tracking)
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.models import get_model

    from .environment import KbEnvironment, outside_paths

    emit = emit or (lambda _event: None)
    kbc = config.knowledge_base
    index = kb.ensure(kbc.corpus_dir, kbc.ontology, kbc.index_dir)
    with kb.connect(index) as conn:
        counts = kb.stats(conn)
        fingerprint = kb.meta(conn).get("fingerprint", "")
    events: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        event = {"t": round(time.monotonic() - started, 2), **event}
        events.append(event)
        emit(event)

    class TracedAgent(DefaultAgent):
        """DefaultAgent unchanged, plus one event per model call and per command."""

        def query(self) -> dict:
            began = time.monotonic()
            try:
                message = super().query()
            except Exception as exc:
                if type(exc).__name__ not in {"LimitsExceeded", "TimeExceeded"}:
                    record({"type": "model_call", "ok": False, "seconds": round(time.monotonic() - began, 2),
                            "error": f"{type(exc).__name__}: {str(exc)[:200]}"})
                raise
            record({"type": "model_call", "ok": True, "seconds": round(time.monotonic() - began, 2),
                    "model": config.model.model_name, **_usage(message)})
            return message

        def execute_actions(self, message: dict) -> list[dict]:
            for action in (message.get("extra") or {}).get("actions", []):
                record({"type": "command", "command": str(action.get("command", ""))[:300]})
            return super().execute_actions(message)

    textbased = config.model.model_class == "litellm_textbased"
    model = model or get_model(config={**config.model.model_dump(), "observation_template": OBSERVATION})
    sandboxed = config.environment.sandbox == "bubblewrap"
    # What the commands need to run: the Python runtime and this package (its source folder when it is
    # installed in editable mode).
    runtime = [sys.prefix, sys.base_prefix, str(Path(sys.executable).resolve().parent),
               str(Path(__file__).resolve().parent.parent)]
    # Everything a command sees: nothing is inherited from this process (no keys, no other paths).
    command_env = {
        "XH_KB_DB": str(index), "XH_MAX_HITS": str(kbc.max_hits), "XH_MAX_TRIPLES": str(config.output.max_triples),
        # The venv's bin folder holds kbsearch, kbread, ... (console scripts of this package).
        "PATH": os.pathsep.join([str(Path(sys.executable).parent), "/usr/local/bin", "/usr/bin", "/bin"]),
        "HOME": str(work), "TMPDIR": "/tmp" if sandboxed else str(work), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "TERM": "dumb", "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment = KbEnvironment(
        cwd=str(work), env=command_env, timeout=config.agent.command_timeout_seconds,
        sandbox=config.environment.sandbox, bwrap=config.environment.bwrap,
        read_only=[*runtime, str(index.parent), *map(str, config.environment.extra_read_only)])
    trajectory = folder / "trajectory.json"
    agent = TracedAgent(
        model, environment,
        system_template=_template(config.prompts.system, "system") + (TEXT_FORMAT if textbased else ""),
        instance_template=_template(config.prompts.instance, "instance"),
        step_limit=config.agent.step_limit, cost_limit=0.0,
        wall_time_limit_seconds=config.agent.wall_time_limit_seconds,
        max_consecutive_format_errors=config.agent.max_consecutive_format_errors,
        output_path=trajectory,
    )

    def stop(signum: int, _frame: Any) -> None:
        # The harness stops a run with SIGTERM: end the running command too (it has its own session).
        environment.kill_current()
        raise SystemExit(128 + signum)

    try:
        previous = signal.signal(signal.SIGTERM, stop)
    except ValueError:                                          # not the main thread (tests)
        previous = None
    record({"type": "start", "run_id": run_id, "harness": "mini-swe-agent", "model": config.model.model_name,
            "sandbox": config.environment.sandbox})
    error = None
    try:
        exit_info = agent.run(question, n_papers=counts["papers"], n_passages=counts["passages"],
                              max_triples=config.output.max_triples)
    except Exception as exc:
        exit_info = {"exit_status": type(exc).__name__, "submission": ""}
        error = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        environment.kill_current()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)
    exit_status = str(exit_info.get("exit_status") or "")
    answer_file = work / "answer.json"
    written = answer_file.is_file()
    document = contract.parse(answer_file.read_text(encoding="utf-8", errors="replace")) if written else None
    if document is None and not written and exit_info.get("submission"):
        document = contract.parse(str(exit_info["submission"]))
    with kb.connect(index) as conn:
        checked = contract.check(document, conn, config.output.max_triples)
    if error:
        status = "error"
    elif checked.ok:
        status = "answered"
    elif exit_status in {"LimitsExceeded", "TimeExceeded"}:
        status = "limits_exceeded"
    else:
        status = "invalid_answer" if written or document is not None else "no_answer"
    calls = [event for event in events if event["type"] == "model_call"]
    commands = [event["command"] for event in events if event["type"] == "command"]
    result = {
        "run_id": run_id,
        "question": question,
        "status": status,
        "exit_status": exit_status,
        "error": error,
        "problem": checked.problem,
        "answer": checked.answer,
        "triples": [triple.model_dump() for triple in checked.triples],
        "dropped_triples": checked.dropped,
        "evidence_not_in_source": checked.evidence_not_in_source,
        "outside_paths": [] if sandboxed else outside_paths(commands, work),
        "usage": {
            "model_calls": len(calls),
            "failed_model_calls": sum(not event["ok"] for event in calls),
            "commands": len(commands),
            "prompt_tokens": sum(event.get("prompt_tokens", 0) for event in calls),
            "completion_tokens": sum(event.get("completion_tokens", 0) for event in calls),
            "model_seconds": round(sum(event["seconds"] for event in calls), 2),
        },
        "duration_s": round(time.monotonic() - started, 2),
        "harness": {"name": "mini-swe-agent", "version": metadata.version("mini-swe-agent"),
                    "external_harness": metadata.version("external-harness")},
        "config": {"name": config.name, "digest": config.digest(), "path": str(config.source or ""),
                   "model": config.model.model_name, "step_limit": config.agent.step_limit,
                   "sandbox": config.environment.sandbox},
        "knowledge_base": {"index": str(index), "corpus_dir": str(kbc.corpus_dir), "fingerprint": fingerprint,
                           **counts},
        "trajectory": str(trajectory),
        "events": events,
    }
    record({"type": "finish", "status": status, "triples": len(checked.triples)})
    (folder / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

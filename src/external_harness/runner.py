"""One question through mini-SWE-agent, with the knowledge-base commands on PATH.

The runner owns everything around the agent loop and nothing inside it: it prepares a working
folder, renders the prompts, runs mini-SWE-agent's DefaultAgent unchanged (a subclass only times
model calls and commands for the event stream), then checks answer.json against the contract and
writes result.json next to the full trajectory.

The answer is answer.json, or else mini-SWE-agent's own submission (what the final command printed after
COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT). Nothing else is rescued: a reply without a tool call is a format
error inside mini-SWE-agent (LitellmModel.query raises before any assistant message is stored; the reply
text survives only in the format-error message's ``extra["response"]``), and like mini-SWE-agent we do not
read answers from it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
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

# Our observation template: output of 8000 characters or more is cut to its first 5000 and last 2000
# characters (mini-SWE-agent's SWE-bench config cuts at 10000 to 5000 + 5000 and adds a warning).
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


SANDBOX_WARNING = ("sandbox is none: commands can read every file this user can (the gold set, other runs); "
                   "use environment.sandbox: bubblewrap for evaluation runs")

CORPUS_COMMANDS = re.compile(r"(^|[\s;&|(])(kbsearch|kbread|kbpapers|kbfacts|onto)\b")


def run(config: Config, question: str, *, run_id: str | None = None, out: Path | None = None,
        emit: Emit | None = None, model: Any | None = None, evidence_doc: Path | None = None) -> dict[str, Any]:
    """Answer ``question``; returns the result document (also written to ``<run folder>/result.json``).
    ``model`` replaces the configured model (tests pass mini-SWE-agent's DeterministicModel).

    ``evidence_doc``: a JSON list of triples (triple_id, subject, predicate, object, paper_id, source_id,
    evidence, interpretation) that is the run's only input. The agent then has only ``kbdoc`` and
    ``kbcheck`` on its PATH and the instance prompt ``instance_doc``; result.json records the input and
    any command that still reached for the corpus commands."""
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
    doc_rows: list[dict[str, Any]] | None = None
    tool_dirs = [str(Path(sys.executable).parent)]
    doc_env: dict[str, str] = {}
    doc_read_only: list[str] = []
    if evidence_doc is not None:
        raw = Path(evidence_doc).read_bytes()
        doc_rows = json.loads(raw)
        inputs, tools_dir = folder / "input", folder / "bin"
        inputs.mkdir(exist_ok=True)
        tools_dir.mkdir(exist_ok=True)
        (inputs / "evidence_doc.json").write_bytes(raw)
        for name in ("kbdoc", "kbcheck"):                       # the only commands of this run
            link = tools_dir / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(Path(sys.executable).parent / name)
        tool_dirs = [str(tools_dir)]
        doc_env = {"XH_DOC": str(inputs / "evidence_doc.json")}
        doc_read_only = [str(inputs), str(tools_dir)]
        doc_digest = hashlib.sha256(raw).hexdigest()

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
                    "model": model_used, **_usage(message)})
            return message

        def execute_actions(self, message: dict) -> list[dict]:
            for action in (message.get("extra") or {}).get("actions", []):
                record({"type": "command", "command": str(action.get("command", ""))[:300]})
            return super().execute_actions(message)

    textbased = config.model.model_class == "litellm_textbased"
    model = model or get_model(config={**config.model.model_dump(), "observation_template": OBSERVATION})
    # What actually ran: the model object's name (a test model has its own) and the sandbox of the environment.
    model_used = str(getattr(getattr(model, "config", None), "model_name", "") or type(model).__name__)
    sandboxed = config.environment.sandbox == "bubblewrap"
    warnings = [] if sandboxed else [SANDBOX_WARNING]
    # What the commands need to run: the Python runtime and this package (its source folder when it is
    # installed in editable mode).
    runtime = [sys.prefix, sys.base_prefix, str(Path(sys.executable).resolve().parent),
               str(Path(__file__).resolve().parent.parent)]
    # Everything a command sees: nothing is inherited from this process (no keys, no other paths).
    command_env = {
        "XH_KB_DB": str(index), "XH_MAX_HITS": str(kbc.max_hits), "XH_MAX_TRIPLES": str(config.output.max_triples),
        # The venv's bin folder holds kbsearch, kbread, ... (console scripts of this package); with an evidence
        # document, a folder holding only kbdoc and kbcheck.
        "PATH": os.pathsep.join([*tool_dirs, "/usr/local/bin", "/usr/bin", "/bin"]), **doc_env,
        "HOME": str(work), "TMPDIR": "/tmp" if sandboxed else str(work), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "TERM": "dumb", "PAGER": "cat", "MANPAGER": "cat", "LESS": "-R", "PYTHONDONTWRITEBYTECODE": "1",
    }
    environment = KbEnvironment(
        cwd=str(work), env=command_env, timeout=config.agent.command_timeout_seconds,
        sandbox=config.environment.sandbox, bwrap=config.environment.bwrap,
        read_only=[*runtime, str(index.parent), *doc_read_only, *map(str, config.environment.extra_read_only)])
    trajectory = folder / "trajectory.json"
    agent = TracedAgent(
        model, environment,
        system_template=_template(config.prompts.system, "system") + (TEXT_FORMAT if textbased else ""),
        instance_template=_template(config.prompts.instance, "instance_doc" if doc_rows is not None else "instance"),
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
    record({"type": "start", "run_id": run_id, "harness": "mini-swe-agent", "model": model_used,
            "sandbox": environment.config.sandbox})
    for warning in warnings:
        record({"type": "warning", "message": warning})
    error = None
    try:
        exit_info = agent.run(question, n_papers=counts["papers"], n_passages=counts["passages"],
                              max_triples=config.output.max_triples, max_hits=kbc.max_hits, textbased=textbased,
                              n_triples=len(doc_rows or []))
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
    answer_source = "answer.json" if document is not None else None
    if document is None and not written and exit_info.get("submission"):
        document = contract.parse(str(exit_info["submission"]))
        answer_source = "submission" if document is not None else None
    with kb.connect(index) as conn:
        checked = contract.check(document, conn, config.output.max_triples, doc=doc_rows)
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
        # Triples cut by output.max_triples (0 = no cap); doc mode: triples dropped because they are not D's.
        "truncated_triples": checked.truncated,
        "max_triples": config.output.max_triples,
        "not_in_doc": checked.not_in_doc,
        "answer_source": answer_source,
        # What this run actually used, so an evaluation can check it (also in config and the start event).
        "model": model_used,
        "sandbox": environment.config.sandbox,
        "warnings": warnings,
        "evidence_not_in_source": checked.evidence_not_in_source,
        "outside_paths": [] if sandboxed else outside_paths(commands, work),
        "input": ({"kind": "evidence_doc", "triples": len(doc_rows), "sha256": doc_digest,
                   "doc_matches": checked.doc_matches,
                   "corpus_commands": [command for command in commands if CORPUS_COMMANDS.search(command)]}
                  if doc_rows is not None else {"kind": "corpus"}),
        "usage": {
            "model_calls": len(calls),
            "failed_model_calls": sum(not event["ok"] for event in calls),
            # Replies without a command (no tool call, or not in the expected format).
            "format_errors": sum(str(event.get("error", "")).startswith("FormatError") for event in calls),
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
                   "sandbox": config.environment.sandbox, "max_triples": config.output.max_triples},
        "knowledge_base": {"index": str(index), "corpus_dir": str(kbc.corpus_dir), "fingerprint": fingerprint,
                           **counts},
        "trajectory": str(trajectory),
        "events": events,
    }
    record({"type": "finish", "status": status, "triples": len(checked.triples)})
    (folder / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result

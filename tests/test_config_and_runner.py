from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_harness import runner
from external_harness.config import expand, load

ANSWER = {
    "answer": "p1 detects hallucinations and evaluates on BenchY [p1-abstract].",
    "triples": [{"subject": "p1", "predicate": "usesDataset", "object": "BenchY", "source_id": "p1-abstract",
                 "evidence": "evaluate on BenchY"},
                {"subject": "p1", "predicate": "claims", "object": "x", "source_id": "p9-unknown"}],
}


def test_config_expands_variables_and_resolves_paths(config_file: Path, corpus: Path, monkeypatch) -> None:
    cfg = load(config_file)
    assert cfg.knowledge_base.corpus_dir == corpus
    assert cfg.knowledge_base.ontology == [(config_file.parent / "corpus" / "ontology.ttl").resolve()]
    assert cfg.environment.workdir_root == (config_file.parent / "runs").resolve()
    assert cfg.model.model_kwargs["api_key"] == "secret"
    digest = cfg.digest()
    monkeypatch.setenv("XH_TEST_KEY", "another")
    assert load(config_file).digest() == digest                     # the key is not part of the digest
    with pytest.raises(ValueError, match="NOT_SET_ANYWHERE"):
        expand("${NOT_SET_ANYWHERE}")
    assert expand({"a": ["${NOT_SET_ANYWHERE:-x}"]}) == {"a": ["x"]}


def _model(commands: list[str]):
    from minisweagent.models.test_models import DeterministicModel, make_output

    return DeterministicModel(outputs=[make_output(f"step {i}", [{"command": command}], cost=0.0)
                                       for i, command in enumerate(commands)])


def test_a_run_uses_the_commands_writes_the_answer_and_records_everything(config, tmp_path: Path) -> None:
    commands = [
        "kbsearch hallucination -k 3",
        "cat <<'EOF' > answer.json\n" + json.dumps(ANSWER) + "\nEOF",
        "kbcheck",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat answer.json",
    ]
    events: list[dict] = []
    result = runner.run(config, "Which benchmark does the hallucination paper use?", run_id="x-test",
                        model=_model(commands), emit=events.append)
    assert result["status"] == "answered" and result["exit_status"] == "Submitted"
    assert [t["object"] for t in result["triples"]] == ["BenchY"]
    assert result["dropped_triples"][0]["why"] == "unknown source_id 'p9-unknown'"
    assert result["usage"]["model_calls"] == 4 and result["usage"]["commands"] == 4
    # (the warning: the test config runs without the sandbox)
    assert [event["type"] for event in events][:4] == ["start", "warning", "model_call", "command"]
    assert events[-1] == {**events[-1], "type": "finish", "status": "answered", "triples": 1}
    folder = config.environment.workdir_root / "x-test"
    trajectory = json.loads((folder / "trajectory.json").read_text())
    observation = trajectory["messages"][3]["content"]
    assert "p1-abstract | p1 | 2025" in observation                # kbsearch ran inside the run folder
    assert json.loads((folder / "result.json").read_text())["status"] == "answered"
    assert "Which benchmark" in trajectory["messages"][1]["content"]
    assert "2 scientific papers (3 passages)" in trajectory["messages"][1]["content"]


def test_a_run_that_hits_the_step_limit_without_an_answer_says_so(config) -> None:
    config.agent.step_limit = 2
    result = runner.run(config, "q", run_id="x-limit", model=_model(["kbpapers --count"] * 3))
    assert result["status"] == "limits_exceeded" and result["exit_status"] == "LimitsExceeded"
    assert result["problem"] == "answer.json is missing or not a JSON object"


def test_an_invalid_answer_file_is_reported(config) -> None:
    result = runner.run(config, "q", run_id="x-bad", model=_model([
        "echo '{\"answer\": \"\"}' > answer.json", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    assert result["status"] == "invalid_answer" and result["problem"] == '"answer" must be a non-empty string'


def test_commands_see_only_the_listed_environment(config, monkeypatch) -> None:
    monkeypatch.setenv("SECRET_TOKEN", "do-not-leak")
    result = runner.run(config, "q", run_id="x-env", model=_model(["env", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    trajectory = json.loads(Path(result["trajectory"]).read_text())
    listing = trajectory["messages"][3]["content"]
    assert "XH_KB_DB=" in listing and "do-not-leak" not in listing and "SECRET_TOKEN" not in listing
    assert result["outside_paths"] == ["env"]                       # flagged for the audit


def test_an_earlier_answer_file_is_never_scored_again(config) -> None:
    first = runner.run(config, "q", run_id="x-again", model=_model([
        "cat <<'EOF' > answer.json\n" + json.dumps(ANSWER) + "\nEOF", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    assert first["status"] == "answered" and first["evidence_not_in_source"] == 0
    second = runner.run(config, "q", run_id="x-again", model=_model(["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    assert second["status"] == "no_answer"


def test_audit_flags_commands_that_look_outside_the_run_folder(tmp_path: Path) -> None:
    from external_harness.environment import outside_paths

    work = tmp_path / "work"
    commands = ["kbsearch x", "cat /etc/passwd", "cat ../other/answer.json", f"cat {work}/answer.json",
                "ls > /dev/null", "printenv"]
    assert outside_paths(commands, work) == ["cat /etc/passwd", "cat ../other/answer.json", "printenv"]


def test_a_stopped_run_kills_the_command_in_its_own_session(tmp_path: Path) -> None:
    import threading
    import time

    from external_harness.environment import KbEnvironment

    environment = KbEnvironment(cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin"}, timeout=60)
    outputs: list[dict] = []
    thread = threading.Thread(target=lambda: outputs.append(environment.execute({"command": "sleep 30"})))
    started = time.monotonic()
    thread.start()
    while environment.current is None:
        time.sleep(0.05)
    environment.kill_current()
    thread.join(timeout=10)
    assert not thread.is_alive() and time.monotonic() - started < 10 and outputs[0]["returncode"] != 0


def _bwrap_works() -> bool:
    import shutil
    import subprocess

    if not shutil.which("bwrap"):
        return False
    return subprocess.run(["bwrap", "--unshare-all", "--ro-bind", "/usr", "/usr", "--ro-bind-try", "/bin", "/bin",
                           "--ro-bind-try", "/lib", "/lib", "--ro-bind-try", "/lib64", "/lib64", "/bin/true"],
                          capture_output=True).returncode == 0


@pytest.mark.skipif(not _bwrap_works(), reason="bubblewrap is not available")
def test_the_bubblewrap_sandbox_shows_the_knowledge_base_and_nothing_else(config, corpus: Path) -> None:
    config.environment.sandbox = "bubblewrap"
    result = runner.run(config, "q", run_id="x-bwrap", model=_model([
        "kbsearch hallucination -k 1",
        f"cat {corpus}/papers.jsonl || echo NO-CORPUS-FILE",
        "ls /root /home 2>&1 | head -3; echo ok > note.txt && cat note.txt",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    messages = json.loads(Path(result["trajectory"]).read_text())["messages"]
    assert "p1-abstract | p1" in messages[3]["content"]                   # the index is readable
    assert "NO-CORPUS-FILE" in messages[5]["content"]                     # the raw corpus folder is not
    assert "ok" in messages[7]["content"] and "No such file" in messages[7]["content"]
    assert result["outside_paths"] == []


def test_a_reply_without_a_tool_call_is_a_format_error_and_its_text_is_not_an_answer(config) -> None:
    """mini-SWE-agent 2.4.6: LitellmModel.query raises FormatError before an assistant message is stored; the
    reply survives only in the format-error message's extra["response"]. Like mini-SWE-agent, the runner does
    not take an answer from it (litellm's mock_response plays the endpoint)."""
    config.model.model_kwargs["mock_response"] = json.dumps(ANSWER)
    events: list[dict] = []
    result = runner.run(config, "q", run_id="x-reply", emit=events.append)
    assert result["status"] == "no_answer" and result["exit_status"] == "RepeatedFormatError"
    assert result["answer_source"] is None and result["triples"] == []
    assert result["usage"]["format_errors"] == result["usage"]["model_calls"] == 5
    assert result["usage"]["commands"] == 0
    messages = json.loads(Path(result["trajectory"]).read_text())["messages"]
    assert [message["role"] for message in messages] == ["system", "user", *["user"] * 5, "exit"]
    rejected = messages[2]
    assert rejected["extra"]["interrupt_type"] == "FormatError" and "No tool calls" in rejected["content"]
    reply = rejected["extra"]["response"]["choices"][0]["message"]
    assert json.loads(reply["content"]) == ANSWER and not reply.get("tool_calls")


def test_an_answer_in_a_replys_text_is_not_taken_when_no_file_was_written(config) -> None:
    from minisweagent.models.test_models import DeterministicModel, make_output

    model = DeterministicModel(outputs=[make_output(json.dumps(ANSWER),
                                                    [{"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}], cost=0.0)])
    result = runner.run(config, "q", run_id="x-text", model=model)
    assert result["status"] == "no_answer" and result["answer_source"] is None


def test_an_answer_printed_after_the_completion_marker_is_the_submission(config) -> None:
    result = runner.run(config, "q", run_id="x-sub", model=_model(
        ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo '" + json.dumps(ANSWER) + "'"]))
    assert result["status"] == "answered" and result["answer_source"] == "submission"


def test_the_result_records_the_model_and_sandbox_used_and_warns_without_a_sandbox(config) -> None:
    events: list[dict] = []
    result = runner.run(config, "q", run_id="x-used", model=_model(["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]),
                        emit=events.append)
    assert result["model"] == "deterministic" and result["config"]["model"] == "openai/heavy-model"
    assert result["sandbox"] == "none" and result["warnings"] == [runner.SANDBOX_WARNING]
    assert events[0] == {**events[0], "type": "start", "model": "deterministic", "sandbox": "none"}
    assert events[1] == {**events[1], "type": "warning", "message": runner.SANDBOX_WARNING}
    assert result["max_triples"] == 500 and result["truncated_triples"] == 0


def test_the_prompt_states_the_limits_as_configured(config) -> None:
    config.knowledge_base.max_hits = 7
    result = runner.run(config, "q", run_id="x-prompt", model=_model(["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    task = json.loads(Path(result["trajectory"]).read_text())["messages"][1]["content"]
    assert '`kbsearch "words" [-k 7] [--paper ID]`' in task
    assert "The run ends after 8 replies (model calls)" in task and "at most 500, one fact each" in task
    config.output.max_triples = 0
    result = runner.run(config, "q", run_id="x-prompt0", model=_model(["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    task = json.loads(Path(result["trajectory"]).read_text())["messages"][1]["content"]
    assert "the facts behind the answer, one fact each" in task


def test_the_example_config_defaults_to_the_evaluation_setup(monkeypatch, tmp_path: Path) -> None:
    from external_harness.config import Config

    example = Path(__file__).parent.parent / "config.example.yaml"
    monkeypatch.setenv("CORPUS_ROOT", str(tmp_path))
    for name in ("XH_MODEL", "XH_SANDBOX"):
        monkeypatch.delenv(name, raising=False)
    cfg = load(example)
    defaults = Config(knowledge_base={"corpus_dir": tmp_path})
    assert cfg.model.model_name == defaults.model.model_name == "openai/heavy-model"
    assert cfg.environment.sandbox == defaults.environment.sandbox == "bubblewrap"
    assert cfg.output.max_triples == defaults.output.max_triples == 500
    assert cfg.agent.max_consecutive_format_errors == defaults.agent.max_consecutive_format_errors
    monkeypatch.setenv("XH_SANDBOX", "none")                          # development
    assert load(example).environment.sandbox == "none"


def test_xh_warns_loudly_on_stderr_when_the_sandbox_is_off(config_file: Path) -> None:
    from typer.testing import CliRunner

    from external_harness.cli import app

    result = CliRunner().invoke(app, ["check", "--config", str(config_file), "--no-model"])
    assert "WARNING: sandbox is none" in result.stderr and "WARNING" not in result.stdout


def test_show_prints_the_steps(config, capsys) -> None:
    from external_harness.cli import show

    result = runner.run(config, "q", run_id="x-show", model=_model(["kbpapers --count",
                                                                   "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    show(Path(result["trajectory"]), width=80)
    out = capsys.readouterr().out
    assert out.startswith("exit: Submitted") and "$ kbpapers --count" in out


DOC = [{"triple_id": "t00001", "subject": "p1", "predicate": "usesDataset", "object": "BenchY", "paper_id": "p1",
        "source_id": "p1-abstract", "evidence": "evaluate on BenchY", "interpretation": "i1"},
       {"triple_id": "t00002", "subject": "p1", "predicate": "usesOptimizer", "object": "AdamW", "paper_id": "p1",
        "source_id": "p1-abstract", "evidence": "trained with AdamW", "interpretation": "i1"}]


def test_with_an_evidence_document_the_agent_has_only_kbdoc_and_kbcheck(config, tmp_path: Path) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps(DOC))
    answer = {"answer": "p1 evaluates on BenchY [p1-abstract].", "triples": [
        {k: DOC[0][k] for k in ("subject", "predicate", "object", "source_id", "evidence")}]}
    commands = ["kbdoc --overview", "kbdoc --predicate usesDataset", "kbsearch BenchY",
                "cat <<'EOF' > answer.json\n" + json.dumps(answer) + "\nEOF", "kbcheck",
                "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat answer.json"]
    result = runner.run(config, "Which benchmark does p1 use?", run_id="x-doc", model=_model(commands),
                        evidence_doc=doc)
    assert result["status"] == "answered" and [t["object"] for t in result["triples"]] == ["BenchY"]
    assert result["input"] == {**result["input"], "kind": "evidence_doc", "triples": 2,
                               "corpus_commands": ["kbsearch BenchY"]}
    messages = json.loads(Path(result["trajectory"]).read_text())["messages"]
    assert "evidence document: 2 knowledge-graph triples" in messages[1]["content"]
    assert "2 triples" in messages[3]["content"] and "usesDataset (1)" in messages[3]["content"]
    assert "t00001 | p1 | usesDataset | BenchY | p1-abstract" in messages[5]["content"]
    assert "command not found" in messages[7]["content"] or "not found" in messages[7]["content"]
    assert result["input"]["doc_matches"] == {"spo_source": 1} and result["not_in_doc"] == 0


def test_with_an_evidence_document_triples_that_are_not_ds_are_dropped(config, tmp_path: Path) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps(DOC))
    answer = {"answer": "p1 uses BenchY [p1-abstract].", "triples": [
        {"triple_id": "t00002", **{k: DOC[1][k] for k in ("subject", "predicate", "object", "source_id")}},
        {"subject": "p1", "predicate": "usesDataset", "object": "BenchY", "source_id": "p1-abstract"},
        {"subject": "p1", "predicate": "scores", "object": "sentences", "source_id": "p1-s2-p1"}]}
    result = runner.run(config, "q", run_id="x-doc2", evidence_doc=doc, model=_model([
        "cat <<'EOF' > answer.json\n" + json.dumps(answer) + "\nEOF", "kbcheck",
        "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    assert result["status"] == "answered" and [t["triple_id"] for t in result["triples"]] == ["t00002", "t00001"]
    assert result["not_in_doc"] == 1 and result["dropped_triples"][0]["why"] == "not in the evidence document"
    assert result["input"]["doc_matches"] == {"triple_id": 1, "spo_source": 1}
    messages = json.loads(Path(result["trajectory"]).read_text())["messages"]
    assert "(1 not in the evidence document)" in messages[5]["content"]


def test_without_an_evidence_document_kbdoc_says_so(config) -> None:
    result = runner.run(config, "q", run_id="x-nodoc", model=_model(["kbdoc --overview",
                                                                      "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]))
    assert result["input"] == {"kind": "corpus"}
    assert "this run's input is the corpus" in json.loads(Path(result["trajectory"]).read_text())["messages"][3]["content"]

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
    assert [event["type"] for event in events][:3] == ["start", "model_call", "command"]
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

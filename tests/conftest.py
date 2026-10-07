from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from external_harness import kb
from external_harness.config import load

DATA = Path(__file__).parent / "data"


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    folder = tmp_path / "corpus"
    folder.mkdir()
    for name in ("papers.jsonl", "passages.jsonl", "facts.jsonl", "ontology.ttl"):
        shutil.copy(DATA / name, folder / name)
    return folder


@pytest.fixture
def index(corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = kb.ensure(corpus, [corpus / "ontology.ttl"], tmp_path / "index")
    monkeypatch.setenv("XH_KB_DB", str(path))
    return path


@pytest.fixture
def config_file(corpus: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CORPUS_ROOT", str(corpus))
    path = tmp_path / "config.yaml"
    path.write_text(
        "name: test\n"
        "model:\n  model_name: openai/heavy-model\n  model_kwargs:\n    api_base: http://127.0.0.1:9/v1\n"
        "    api_key: ${XH_TEST_KEY:-secret}\n"
        "agent:\n  step_limit: 8\n  command_timeout_seconds: 20\n"
        "knowledge_base:\n  corpus_dir: ${CORPUS_ROOT}\n  ontology: corpus/ontology.ttl\n  index_dir: index\n"
        # Without bubblewrap (the config default) so the tests run anywhere; one test switches it on.
        "environment:\n  sandbox: none\n  workdir_root: runs\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def config(config_file: Path):
    return load(config_file)

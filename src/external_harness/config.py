"""The one config file that fully describes an external-harness setup.

Every value the run depends on (model endpoint, limits, corpus, ontology, prompts, environment) lives
here, so a result can be reproduced from its config. ``${VAR}`` and ``${VAR:-default}`` are replaced
from the environment when the file is read; relative paths are resolved against the config file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ModelConfig(_Strict):
    """Passed to mini-SWE-agent's model factory (litellm underneath)."""

    model_name: str = "openai/heavy-model"
    # litellm: native tool calls (a bash tool); litellm_textbased: the command in a markdown block.
    model_class: Literal["litellm", "litellm_textbased"] = "litellm"
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    cost_tracking: Literal["default", "ignore_errors"] = "ignore_errors"   # local models have no price


class AgentConfig(_Strict):
    step_limit: int = Field(30, ge=1, description="Model calls before the run stops (every tool call of a reply runs).")
    wall_time_limit_seconds: int = Field(900, ge=0, description="0 = no limit.")
    command_timeout_seconds: int = Field(60, ge=1)
    max_consecutive_format_errors: int = Field(5, ge=1, description="Replies without a command in a row before the run ends.")


class KnowledgeBaseConfig(_Strict):
    corpus_dir: Path = Field(description="Folder with papers.jsonl, passages.jsonl, facts.jsonl, references.jsonl.")
    ontology: list[Path] = Field(default_factory=list, description="Turtle files with the classes and properties.")
    index_dir: Path = Path(".xh-index")
    max_hits: int = Field(10, ge=1, le=50)

    @field_validator("ontology", mode="before")
    @classmethod
    def _one_or_more(cls, value: Any) -> Any:
        return [value] if isinstance(value, str | Path) else value


class EnvironmentConfig(_Strict):
    # bubblewrap (the default, required for evaluation runs): every command in a fresh namespace (no
    # network; writable: the run folder only; readable: the system, this package's Python and the index).
    # none (development only): a clean environment in the run folder, but the user's files stay readable
    # (result.json flags commands that look outside, and `xh run` warns).
    sandbox: Literal["none", "bubblewrap"] = "bubblewrap"
    bwrap: str = "bwrap"
    extra_read_only: list[Path] = Field(default_factory=list, description="More folders visible in the sandbox.")
    workdir_root: Path = Path("runs")


class PromptConfig(_Strict):
    system: str = "default"      # "default" or a path to a Jinja template
    instance: str = "default"


class OutputConfig(_Strict):
    # High enough not to bind for realistic gold; result.json reports how many triples a cap cut (truncated).
    max_triples: int = Field(500, ge=0, description="Most triples kept from answer.json; 0 = no cap.")


class Config(_Strict):
    name: str = "mini-swe-agent-baseline"
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    knowledge_base: KnowledgeBaseConfig
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    prompts: PromptConfig = Field(default_factory=PromptConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    source: Path | None = Field(None, exclude=True)

    def digest(self) -> str:
        """Hash of the effective config (after variable expansion), recorded with every result.
        The API key is left out so a digest can be shared."""
        data = self.model_dump(mode="json")
        data["model"]["model_kwargs"] = {key: value for key, value in data["model"]["model_kwargs"].items()
                                         if key != "api_key"}
        for name in ("system", "instance"):
            path = Path(data["prompts"][name])
            if data["prompts"][name] != "default" and path.is_file():
                data["prompts"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def expand(value: Any) -> Any:
    """Replace ${VAR} and ${VAR:-default} in every string of a parsed YAML document."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            found = os.environ.get(name)
            if found is None or (found == "" and default is not None):
                if default is None:
                    raise ValueError(f"environment variable {name} is not set (use ${{{name}:-default}})")
                return default
            return found
        return _VAR.sub(replace, value)
    if isinstance(value, list):
        return [expand(item) for item in value]
    if isinstance(value, dict):
        return {key: expand(item) for key, item in value.items()}
    return value


def load(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = Config.model_validate({**expand(raw), "source": path})
    base = path.parent

    def resolve(value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (base / value).resolve()

    kb = config.knowledge_base
    kb.corpus_dir, kb.index_dir = resolve(kb.corpus_dir), resolve(kb.index_dir)
    kb.ontology = [resolve(item) for item in kb.ontology]
    config.environment.workdir_root = resolve(config.environment.workdir_root)
    config.environment.extra_read_only = [resolve(item) for item in config.environment.extra_read_only]
    for name in ("system", "instance"):
        value = getattr(config.prompts, name)
        if value != "default":
            setattr(config.prompts, name, str(resolve(Path(value))))
    return config

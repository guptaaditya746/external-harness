"""xh: run the external harness.

    xh index --config config.yaml                 # build the knowledge-base index (also done on demand)
    xh check --config config.yaml                 # config, index, ontology and model endpoint
    xh run --config config.yaml --question "…"    # one question; result.json in the run folder
    xh run ... --events                           # plus one JSON event per line on stdout (for the harness UI)
    xh batch --config config.yaml --questions questions.json [--ids nlp01,nlp02]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

from . import kb, runner
from .config import load

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__.split("\n\n")[0])
ConfigOption = Annotated[Path, typer.Option("--config", "-c", help="The YAML config file.", exists=True)]


@app.command()
def index(config: ConfigOption) -> None:
    """Build (or rebuild) the knowledge-base index from the corpus files and the ontology."""
    cfg = load(config)
    kbc = cfg.knowledge_base
    started = time.monotonic()
    counts = kb.build(kbc.corpus_dir, kbc.ontology, kbc.index_dir)
    typer.echo(f"{kbc.index_dir / kb.INDEX_NAME}: " + ", ".join(f"{value} {name}" for name, value in counts.items())
               + f" ({time.monotonic() - started:.1f}s)")


@app.command()
def check(config: ConfigOption,
          model: Annotated[bool, typer.Option(help="Also ask the model endpoint for its models.")] = True) -> None:
    """Check the config, the index, the ontology and (optionally) the model endpoint."""
    cfg = load(config)
    kbc = cfg.knowledge_base
    typer.echo(f"config {cfg.name} (digest {cfg.digest()})")
    path = kb.ensure(kbc.corpus_dir, kbc.ontology, kbc.index_dir)
    with kb.connect(path) as conn:
        counts = kb.stats(conn)
    typer.echo(f"index {path}: " + ", ".join(f"{value} {name}" for name, value in counts.items()))
    problems = []
    if not counts["terms"]:
        problems.append("the ontology has no classes or properties (knowledge_base.ontology)")
    if model:
        import httpx

        base = str(cfg.model.model_kwargs.get("api_base") or "").rstrip("/")
        if not base:
            problems.append("model.model_kwargs.api_base is not set")
        else:
            key = cfg.model.model_kwargs.get("api_key")
            try:
                response = httpx.get(f"{base}/models", timeout=10,
                                     headers={"Authorization": f"Bearer {key}"} if key else {})
                response.raise_for_status()
                served = [item.get("id") for item in response.json().get("data", [])]
                wanted = cfg.model.model_name.split("/", 1)[-1]
                typer.echo(f"endpoint {base}: {', '.join(map(str, served))}")
                if wanted not in served:
                    problems.append(f"model {wanted!r} is not served by {base}")
            except Exception as exc:
                problems.append(f"endpoint {base} unreachable: {type(exc).__name__}: {exc}")
    for problem in problems:
        typer.echo(f"problem: {problem}", err=True)
    raise typer.Exit(1 if problems else 0)


@app.command()
def run(
    config: ConfigOption,
    question: Annotated[str | None, typer.Option(help="The question.")] = None,
    question_file: Annotated[Path | None, typer.Option(help="A file holding the question.", exists=True)] = None,
    run_id: Annotated[str | None, typer.Option(help="Id for the run folder and the result.")] = None,
    out: Annotated[Path | None, typer.Option(help="Run folder (default: environment.workdir_root/<run id>).")] = None,
    events: Annotated[bool, typer.Option(help="Print one JSON event per line on stdout.")] = False,
) -> None:
    """Answer one question. The exit code is 0 whenever result.json was written, whatever its status."""
    text = question_file.read_text(encoding="utf-8").strip() if question_file else (question or "").strip()
    if not text:
        raise typer.BadParameter("give --question or --question-file")
    cfg = load(config)

    def emit(event: dict) -> None:
        if events:
            sys.stdout.write(json.dumps(event, ensure_ascii=False) + "\n")
            sys.stdout.flush()

    result = runner.run(cfg, text, run_id=run_id, out=out, emit=emit)
    if not events:
        typer.echo(f"{result['status']}: {len(result['triples'])} triples, {result['usage']['model_calls']} model calls, "
                   f"{result['duration_s']}s")
        typer.echo(result["answer"] or result.get("problem") or result.get("error") or "")


@app.command()
def batch(
    config: ConfigOption,
    questions: Annotated[Path, typer.Option(help="JSON list of {id, question}.", exists=True)],
    ids: Annotated[str | None, typer.Option(help="Only these ids, comma-separated.")] = None,
    out: Annotated[Path | None, typer.Option(help="results.jsonl (default: workdir_root/batch-<time>.jsonl).")] = None,
) -> None:
    """Run many questions one after the other (without the harness API); one result line each."""
    cfg = load(config)
    rows = json.loads(questions.read_text(encoding="utf-8"))
    wanted = {item.strip() for item in ids.split(",")} if ids else None
    rows = [row for row in rows if wanted is None or row["id"] in wanted]
    target = out or cfg.environment.workdir_root / f"batch-{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    for row in rows:
        result = runner.run(cfg, row["question"], run_id=f"{row['id']}-{time.strftime('%H%M%S')}")
        line = {key: result[key] for key in ("run_id", "status", "answer", "triples", "usage", "duration_s")}
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"question_id": row["id"], **line}, ensure_ascii=False) + "\n")
        typer.echo(f"[{row['id']}] {result['status']} {len(result['triples'])} triples {result['duration_s']}s")
    typer.echo(f"results in {target}")

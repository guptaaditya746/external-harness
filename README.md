# external-harness

A light **external baseline** for the Research Harness (rambodambo). An off-the-shelf agent,
[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, answers a research question
with nothing but a bash shell and six read-only commands over the same corpus and ontology the
harness uses. It uses none of the harness's pipeline: no planner, retrieval stages, schema
inference, extraction, evidence review or verifier. Its answer and triples are scored against the
same gold as the harness's Fast, Adaptive and Agentic paths.

mini-SWE-agent is the fixed harness of SWE-bench's bash-only leaderboard. It is used there because it
is minimal scaffolding (a ReAct loop with one bash tool), so differences come from the model, not the
harness. Here it plays the same role: "what does a capable general agent get from the same knowledge
base and model, without our pipeline?"

## What the agent gets

| Command | What it does |
|---|---|
| `kbsearch "words" [-k 10] [--paper ID]` | Full-text search over the passages (SQLite FTS5, BM25 ranking, Porter stemming; "quoted phrases" stay phrases) |
| `kbread ID [--section WORD]` | One passage, or a paper's metadata and passage list |
| `kbpapers [--venue V] [--year Y] [--title WORD] [--author NAME] [--count]` | Papers by DBLP metadata (venues as whole words, titles by word prefix) |
| `kbfacts PAPER_ID` / `--cited-by PAPER_ID` / `--author NAME` | DBLP facts and citations between corpus papers |
| `onto [WORD] [--kind class\|property]` | Classes and properties of the ontology |
| `kbcheck` | Checks `answer.json` against the output contract |

Everything else is plain bash in a per-run folder. The commands read `kb.sqlite`, an index built
once from `papers.jsonl`, `passages.jsonl`, `facts.jsonl` and the ontology Turtle files. It is
opened read-only and rebuilt when an input file changes.

## Isolation

A command never inherits the runner's environment. It sees only the index path, `PATH` and a home
inside the run folder, so there are no API keys and no paths to other data.

With `environment.sandbox: bubblewrap`, which the example config uses, each command also runs in a
fresh [bubblewrap](https://github.com/containers/bubblewrap) namespace:
- it has no network;
- the run folder is the only place it can write;
- it can read only the system (read-only), this package's Python runtime and the index.

So the agent cannot read the gold set, other runs, the raw corpus files or anything else on the
machine. `xh check` tests that bubblewrap works.

With `sandbox: none`, commands run in the run folder but can read what the user can read.
`result.json` then lists commands that name paths outside the run folder or read the environment
(`outside_paths`), so a run can be audited.

## What it must deliver

`answer.json` in the run folder:

```json
{"answer": "1-5 sentences, each claim followed by its source id [p12-s3-p2].",
 "triples": [{"subject": "p12", "predicate": "usesDataset", "object": "SQuAD",
              "source_id": "p12-s3-p2", "evidence": "we evaluate on SQuAD"}]}
```

A triple is kept only if every field is filled and its `source_id` is a real passage id or DBLP fact
source id. Dropped triples are listed in `result.json` with the reason.

## Setup (on the node)

```bash
uv sync
cp config.example.yaml config.yaml          # set corpus_dir and ontology (see below)
(cd ../rambodambo && make ontology)         # writes artifacts/ontology/harness_seed.ttl
uv run xh index --config config.yaml        # builds .xh-index/kb.sqlite
uv run xh check --config config.yaml        # config, index, ontology terms, model endpoint
uv run xh run --config config.yaml --question "Which datasets do the RAG papers use?"
```

Every value a run depends on is in `config.yaml`:
- the model (a litellm model string, e.g. `openai/heavy-model`, plus the LiteLLM proxy URL and key);
- the limits (steps, wall time, command timeout);
- the corpus folder and ontology files;
- the sandbox (`bubblewrap`, recommended, or `none`; see below);
- the prompts (the defaults in `src/external_harness/prompts/`, or your own Jinja files).

`${VAR}` and `${VAR:-default}` are read from the environment. Each result records the config's
digest; the API key is not part of it.

## Output of a run

`runs/<run id>/` holds:
- `work/answer.json`: what the agent wrote;
- `trajectory.json`: every message, command and output, from mini-SWE-agent;
- `result.json`: status, answer, kept and dropped triples, how many kept triples quote evidence that is
  not in their passage (`evidence_not_in_source`), `outside_paths`, token usage, the corpus folder and
  index fingerprint, the config digest and versions.

Statuses:
- `answered`
- `no_answer`: no answer file;
- `invalid_answer`: the file breaks the contract;
- `limits_exceeded`: hit the step or time limit;
- `error`: the model or agent failed.

With `--events`, `xh run` also prints one JSON event per line on stdout: start, each model call
(seconds, tokens), each command, finish. The harness UI shows them live.

## In the Research Harness

rambodambo runs this repo as its **External** path (one button on the Ask page). Set these in
rambodambo's `.env`:

```bash
HARNESS_EXTERNAL_COMMAND="uv run --project /path/to/external-harness xh"
HARNESS_EXTERNAL_CONFIG=/path/to/external-harness/config.yaml
```

The harness calls `xh run --events` for each question. It turns the events into its trace, and turns
`result.json` into a normal run: answer graph, claims, checks and artifacts. The Benchmark page,
`make evaluate` and `make run-questions ARGS="--path external"` then treat it like any other path.
You can also run questions without the harness:
`uv run xh batch --config config.yaml --questions questions.json`.

## Fairness notes

- Same model endpoint as the harness (choose the alias in the config), same corpus snapshot, same
  ontology file, same gold.
- The agent gets a search command because grepping 30k JSON lines is not a meaningful baseline. It
  gets no planner, no reranker, no schema inference and no verifier.
- The step limit and wall time are reported with every result. Compare paths at similar budgets.
- mini-SWE-agent is pinned (2.4.6). Its prompts are ours, in `prompts/`, and their text is part of the
  config digest when you replace them. Its global `.env` is not read: each run points it at an empty
  folder.
- Run evaluation batches with `sandbox: bubblewrap`, so the agent cannot see the gold or other runs.

## Development

```bash
uv run pytest -q      # includes an end-to-end run with mini-SWE-agent's deterministic model
uv run ruff check .
```

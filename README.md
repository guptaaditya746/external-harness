# external-harness

A light **external baseline** for the Research Harness (rambodambo). An off-the-shelf agent,
[mini-SWE-agent](https://github.com/SWE-agent/mini-swe-agent) 2.4.6, answers a research question
with nothing but a bash shell and six read-only commands over the same corpus and ontology the
harness uses. It uses none of the harness's pipeline: no planner, retrieval stages, schema
inference, extraction, evidence review or verifier. Its answer and triples are scored against the
same gold as the harness's Fast, Adaptive and Agentic paths.

What runs is **mini-swe-agent 2.4.6's `DefaultAgent`, with an unchanged loop** (linear message history,
the `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` completion marker, one subprocess per command), **with our
environment, prompts and limits**. mini-SWE-agent is the scaffold of SWE-bench's bash-only leaderboard because
it is minimal (a ReAct loop with one bash tool), so differences come from the model rather than the scaffold.
Here it plays the same role: "what does a capable general agent get from the same knowledge base and model,
without our pipeline?" It is not the leaderboard's configuration, though:

| | SWE-bench bash-only (mini-swe-agent `swebench.yaml`) | external-harness |
|---|---|---|
| Agent loop | `DefaultAgent` | `DefaultAgent`, unchanged (a subclass only times model calls and commands for the event stream) |
| Step limit | 250 model calls | 30 (`agent.step_limit`; counts model calls: every tool call of one reply runs) |
| Cost limit | $3 | none (`cost_limit: 0`; local models have no price) |
| Wall-time limit | none | 900 s (`agent.wall_time_limit_seconds`) |
| Format errors before the run ends | 3 in a row | 5 in a row (`agent.max_consecutive_format_errors`) |
| Long command output | over 10,000 characters: first 5,000 + last 5,000, with a warning to the model | 8,000 characters or more: first 5,000 + last 2,000 and the number of elided characters, no warning (`runner.OBSERVATION`) |
| Prompts | SWE-bench task prompts | ours (`src/external_harness/prompts/`), part of the config digest when replaced |
| Environment | Docker container per task | `KbEnvironment`, which replaces `LocalEnvironment.execute`: a clean environment (only the variables the runner lists), optional bubblewrap sandbox, stdin from `/dev/null`, each command in its own session so a stopped run kills it |
| Tools | bash | bash plus the read-only knowledge-base commands below |
| Output | a patch | `answer.json` (answer plus triples), checked against a contract |

### The answer comes from answer.json or the submission, nothing else

A run's answer is `work/answer.json`, or, when no file was written, mini-SWE-agent's own submission (what the
final command printed after the completion marker); `answer_source` in `result.json` says which. A reply without
a tool call is a format error inside mini-SWE-agent: in 2.4.6 `LitellmModel.query` raises `FormatError` before
any assistant message is stored, so the trajectory holds only a user-role rejection whose `extra.response` keeps
the reply. We deliberately do **not** read answers out of such replies (or out of any reply text): that keeps
mini-SWE-agent's behaviour, where an answer exists only once a command produced it. The prompt tells the
model so. (An earlier version tried to rescue answers from assistant messages; it never fired for plain replies,
because no assistant message exists for them, and was removed.)

## What the agent gets

| Command | What it does |
|---|---|
| `kbsearch "words" [-k 10] [--paper ID]` | Full-text search over the passages and their section titles (SQLite FTS5, BM25 ranking, Porter stemming; "quoted phrases" stay phrases); `-k` defaults to `knowledge_base.max_hits` |
| `kbread ID [--section WORD]` | One passage, or a paper's metadata and passage list |
| `kbpapers [--venue V] [--year Y] [--title WORD] [--author NAME] [--count]` | Papers by DBLP metadata (venues as whole words, titles by word prefix) |
| `kbfacts PAPER_ID` / `--cited-by PAPER_ID` / `--author NAME` | DBLP facts and citations between corpus papers |
| `onto [WORD] [--kind class\|property]` | Classes and properties of the ontology |
| `kbcheck` | Checks `answer.json` against the output contract |

Everything else is plain bash in a per-run folder. The commands read `kb.sqlite`, an index built
once from `papers.jsonl`, `passages.jsonl`, `facts.jsonl` and the ontology Turtle files. It is
opened read-only and rebuilt when an input file changes.

### Retrieval differs from the harness's by design

`kbsearch` is SQLite FTS5: the `porter unicode61` tokenizer (Porter stemming), FTS5's `bm25()` (k1 = 1.2,
b = 0.75, FTS5's IDF), our own short stopword list, and both the passage text and its section title are
indexed. The Research Harness's BM25 (rambodambo `harness_corpus`) is `bm25s`: the Lucene variant with
k1 = 1.5, b = 0.75, no stemming and bm25s's English stopwords, over the passage text (and, in hybrid mode,
fused with dense ranking). The same query can therefore rank passages differently on the two paths. This is
intended: the baseline gets a reasonable off-the-shelf search, not a copy of the harness's retrieval.

### With an evidence document as input

`xh run --triples-file doc.json` makes a JSON list of triples (`triple_id`, `subject`, `predicate`, `object`,
`paper_id`, `source_id`, `evidence`, `interpretation`) the run's only evidence. This is the Research Harness's run
input `evidence_doc`: the question's evidence document D from labelling, so every path is scored on choosing and
answering from the same triples, not on finding them. Then:

- the agent's PATH holds only `kbdoc` (overview, word search, one relation a page at a time, full evidence by id)
  and `kbcheck`; the corpus commands are not on it;
- the instance prompt is `prompts/instance_doc.j2` (unless the config names its own);
- `result.json` has `input: {kind: "evidence_doc", triples, sha256, corpus_commands}`, the last listing any
  command that still reached for a corpus command (they fail, but they are recorded for the audit);
- only the document's triples count: an answer triple is kept when it matches a triple of D by `triple_id`
  (the prompt asks the agent to copy it), else by its normalised (subject, predicate, object) and `source_id`,
  else by (subject, predicate, object) alone (it then takes that D triple's source). Matched by id or by
  (s, p, o) alone, it takes D's subject, predicate, object and source. Every other triple is dropped as
  "not in the evidence document", even when its `source_id` is a real passage; `result.json` counts them
  (`not_in_doc`) and says how the kept ones matched (`input.doc_matches`), and `kbcheck` reports them too.

## Isolation

A command never inherits the runner's environment. It sees only the index path, `PATH` and a home
inside the run folder, so there are no API keys and no paths to other data.

With `environment.sandbox: bubblewrap` (the default, and required for evaluation runs), each command also
runs in a fresh [bubblewrap](https://github.com/containers/bubblewrap) namespace:
- it has no network;
- the run folder is the only place it can write;
- it can read only the system (read-only), this package's Python runtime and the index.

So the agent cannot read the gold set, other runs, the raw corpus files or anything else on the
machine. `xh check` tests that bubblewrap works.

With `sandbox: none` (development only: `XH_SANDBOX=none` with the example config), commands run in the
run folder but can read what the user can read. `xh run`, `xh batch` and `xh check` then print a warning on
stderr, `result.json` has it in `warnings` and lists commands that name paths outside the run folder or read
the environment (`outside_paths`), so a run can be audited.

## What it must deliver

`answer.json` in the run folder:

```json
{"answer": "1-5 sentences, each claim followed by its source id [p12-s3-p2].",
 "triples": [{"subject": "p12", "predicate": "usesDataset", "object": "SQuAD",
              "source_id": "p12-s3-p2", "evidence": "we evaluate on SQuAD"}]}
```

A triple is kept only if every field is filled and its `source_id` is a real passage id or DBLP fact
source id (and, with an evidence document, it is one of the document's triples). Dropped triples are listed in
`result.json` with the reason. `output.max_triples` (default 500, 0 = no cap) caps the kept triples; it is
meant not to bind, and when it does `result.json` counts the cut triples (`truncated_triples`).

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
- the model (a litellm model string, plus the LiteLLM proxy URL and key): `openai/heavy-model` by default for
  evaluation (`XH_MODEL=lite-model` for a cheaper development run);
- the limits (steps, wall time, command timeout);
- the corpus folder and ontology files;
- the sandbox (`bubblewrap`, the default and required for evaluation, or `none` for development; see Isolation);
- the prompts (the defaults in `src/external_harness/prompts/`, or your own Jinja files).

`${VAR}` and `${VAR:-default}` are read from the environment. Each result records the config's
digest; the API key is not part of it.

## Output of a run

`runs/<run id>/` holds:
- `work/answer.json`: what the agent wrote;
- `trajectory.json`: every message, command and output, from mini-SWE-agent;
- `result.json`: status, answer, kept and dropped triples, how many kept triples quote evidence that is
  not in their passage (`evidence_not_in_source`), triples cut by the cap (`truncated_triples`) or, with an
  evidence document, not in it (`not_in_doc`), `answer_source`, `outside_paths`, token usage, the corpus
  folder and index fingerprint, the config digest and versions, and the model and sandbox the run actually
  used (`model`, `sandbox`; also in the `start` event) with any `warnings` (sandbox off).

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

- Same model endpoint (the LiteLLM proxy) as the harness, same corpus snapshot, same ontology file, same
  gold. The alias is a choice: the default is `heavy-model` (the harness uses it for planning, extraction and
  review; its Agentic tool loop runs on `lite-model`). Every result records the model it used, and the
  harness copies it into the External run's checks, so an evaluation can verify it.
- The agent gets a search command because grepping 30k JSON lines is not a meaningful baseline. It
  gets no planner, no reranker, no schema inference and no verifier.
- The step limit and wall time are reported with every result. Compare paths at similar budgets. The
  limits differ from SWE-bench's (see the table above).
- mini-SWE-agent is pinned (2.4.6). Its prompts are ours, in `prompts/`, and their text is part of the
  config digest when you replace them. Its global `.env` is not read: each run points it at an empty
  folder.
- Run evaluation batches with `sandbox: bubblewrap` (the default), so the agent cannot see the gold or other
  runs; the harness marks an External run made without it.

## Development

```bash
uv run pytest -q      # includes an end-to-end run with mini-SWE-agent's deterministic model
uv run ruff check .
```

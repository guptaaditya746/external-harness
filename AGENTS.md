# Working on external-harness

- This repo is a **baseline**: keep it minimal and general. Do not add planning, retrieval stages,
  reranking, schema inference, verification or anything else the Research Harness does; the point is to
  measure what an off-the-shelf agent gets without them.
- No question-specific logic: no code paths, keyword lists or prompt examples tied to the evaluation
  questions (`rambodambo/datasets/question_library_2025*.json`) or their papers. Prompt examples use
  made-up entities (p99, MethodX, BenchY).
- Every setting a run depends on belongs in the YAML config (`config.py`), and changes the config digest.
- mini-SWE-agent stays pinned; upgrade it only deliberately and note it in the results.
- The knowledge-base commands only read (`kb.connect` opens SQLite read-only).
- Before committing: `uv run ruff check .` and `uv run pytest -q`.

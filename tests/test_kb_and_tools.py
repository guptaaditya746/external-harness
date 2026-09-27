from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_harness import contract, kb, tools


def test_index_counts_and_search_uses_stemming(index: Path) -> None:
    with kb.connect(index) as conn:
        assert kb.stats(conn) == {"papers": 2, "passages": 3, "facts": 4, "terms": 6}
        hits = kb.search(conn, "hallucinations detectors", k=5)
        assert hits[0]["passage_id"] == "p1-abstract"
        assert kb.search(conn, "the of and") == []                   # nothing searchable left
        assert [row["passage_id"] for row in kb.search(conn, "detector", paper="p2")] == ["p2-abstract"]
        assert kb.known_source_ids(conn, ["p1-s2-p1", "p2-ref-1", "research_notes"]) == {"p1-s2-p1", "p2-ref-1"}


def test_match_expression_keeps_phrases_and_drops_stop_words() -> None:
    assert kb.match_expression('what does "graph coloring" use?') == '"graph coloring" OR "use"'
    assert kb.match_expression("?!") == ""


def test_the_index_is_rebuilt_when_an_input_changes(corpus: Path, tmp_path: Path) -> None:
    path = kb.ensure(corpus, [], tmp_path / "idx")
    with (corpus / "papers.jsonl").open("a") as handle:
        handle.write(json.dumps({"paper_id": "p3", "title": "New", "year": 2023, "venue": "X"}) + "\n")
    import os
    import time
    os.utime(corpus / "papers.jsonl", (time.time() + 5, time.time() + 5))
    path = kb.ensure(corpus, [], tmp_path / "idx")
    with kb.connect(path) as conn:
        assert kb.stats(conn)["papers"] == 3


def test_search_and_read_commands(index: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tools.kbsearch(["hallucination", "-k", "2"])
    out = capsys.readouterr().out
    assert out.startswith("p1-abstract | p1 | 2025 | ACL 2025 | Detecting Hallucinations")
    tools.kbread(["p1-s2-p1"])
    assert "Our detector scores" in capsys.readouterr().out
    tools.kbread(["p1", "--section", "method"])
    out = capsys.readouterr().out
    assert "authors: Ada Byron, Alan Turing" in out and "p1-s2-p1" in out and "p1-abstract" not in out


def test_paper_listing_matches_venues_by_whole_words_and_titles_by_prefix(index: Path, capsys) -> None:
    tools.kbpapers(["--venue", "ACL", "--count"])
    assert capsys.readouterr().out.strip() == "1"
    tools.kbpapers(["--venue", "EMNLP 2025", "--count"])
    assert capsys.readouterr().out.strip() == "0"
    tools.kbpapers(["--title", "hallucination"])
    assert "p1 | 2025" in capsys.readouterr().out


def test_facts_show_corpus_citations_only(index: Path, capsys) -> None:
    tools.kbfacts(["p2"])
    out = capsys.readouterr().out
    assert "p2-ref-1 | cites | p1 (Detecting" in out and "p2-ref-2" not in out
    tools.kbfacts(["--cited-by", "p1"])
    assert "(1 citing corpus papers)" in capsys.readouterr().out
    tools.kbfacts(["--author", "byron"])
    assert "(1 papers)" in capsys.readouterr().out


def test_ontology_listing(index: Path, capsys) -> None:
    tools.onto(["dataset", "--kind", "property"])
    out = capsys.readouterr().out
    assert "hrn:usesDataset | object property (sub of hrn:researchRelation) | hrn:Paper -> hrn:Dataset" in out
    assert "(1 terms)" in out


def test_contract_keeps_valid_triples_and_says_why_others_were_dropped(index: Path) -> None:
    text = """Here it is:
```json
{"answer": "p1 evaluates on BenchY [p1-abstract].",
 "triples": [
  {"subject": "p1", "predicate": "usesDataset", "object": "BenchY", "source_id": "[p1-abstract]",
   "evidence": "evaluate on BenchY"},
  {"subject": "p1", "predicate": "usesDataset", "object": "benchy", "source_id": "p1-abstract"},
  {"subject": "p1", "predicate": "claims", "object": "x", "source_id": "research_notes"},
  {"subject": "", "predicate": "claims", "object": "x", "source_id": "p1-abstract"}]}
```"""
    with kb.connect(index) as conn:
        checked = contract.check(contract.parse(text), conn)
    assert checked.ok and [t.object for t in checked.triples] == ["BenchY"]
    assert [item["why"] for item in checked.dropped] == ["empty subject", "duplicate",
                                                         "unknown source_id 'research_notes'"]
    assert contract.check(contract.parse('{"answer": ""}'), None).problem == '"answer" must be a non-empty string'
    assert contract.check(None, None).ok is False


def test_kbcheck_reports_invalid_files(index: Path, tmp_path: Path, capsys) -> None:
    path = tmp_path / "answer.json"
    path.write_text('{"answer": "It does [p1-abstract].", "triples": []}')
    tools.kbcheck([str(path)])
    assert capsys.readouterr().out.startswith("ok: 0 triples kept")
    path.write_text("not json")
    with pytest.raises(SystemExit, match="invalid"):
        tools.kbcheck([str(path)])


def test_concurrent_rebuilds_do_not_collide(corpus: Path, tmp_path: Path) -> None:
    import threading

    results: list[Path] = []
    errors: list[BaseException] = []

    def build() -> None:
        try:
            results.append(kb.ensure(corpus, [corpus / "ontology.ttl"], tmp_path / "shared"))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=build) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors and len(set(results)) == 1
    with kb.connect(results[0]) as conn:
        assert kb.stats(conn)["passages"] == 3 and "corpus_dir" not in kb.meta(conn)


def test_evidence_that_is_not_in_the_passage_is_counted(index: Path) -> None:
    document = {"answer": "x [p1-abstract].", "triples": [
        {"subject": "p1", "predicate": "usesDataset", "object": "BenchY", "source_id": "p1-abstract",
         "evidence": "evaluate on BenchY"},
        {"subject": "p1", "predicate": "claims", "object": "fast", "source_id": "p1-s2-p1", "evidence": "it is very fast"}]}
    with kb.connect(index) as conn:
        assert contract.check(document, conn).evidence_not_in_source == 1

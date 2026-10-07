from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_harness import contract, kb, tools


def test_index_counts_and_search_uses_stemming(index: Path) -> None:
    with kb.connect(index) as conn:
        assert kb.stats(conn) == {"papers": 2, "passages": 3, "facts": 4, "terms": 8}
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
    tools.onto(["employs"])
    out = capsys.readouterr().out
    assert "One research entity applies another. (same as ex:employs)" in out     # equivalences are named


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


DOC = [{"triple_id": "t00001", "subject": "p1", "predicate": "usesDataset", "object": "BenchY", "paper_id": "p1",
        "source_id": "p1-abstract", "evidence": "evaluate on BenchY", "interpretation": "i1"},
       {"triple_id": "t00002", "subject": "p1", "predicate": "detects", "object": "hallucination", "paper_id": "p1",
        "source_id": "p1-abstract", "evidence": "hallucination detection", "interpretation": "i1"},
       {"triple_id": "t00003", "subject": "p2", "predicate": "comparedWith", "object": "the detector", "paper_id": "p2",
        "source_id": "p2-abstract", "evidence": "compared with the detector", "interpretation": "i1"}]


def test_in_doc_mode_only_the_evidence_documents_triples_are_kept(index: Path) -> None:
    document = {"answer": "x [p1-abstract].", "triples": [
        # by triple id: D's fields are kept (the agent's evidence quote stays)
        {"triple_id": "t00002", "subject": "p1", "predicate": "detects", "object": "hallucinations",
         "source_id": "p1-abstract", "evidence": "hallucination detection for retrieval"},
        # by normalised (s, p, o) and source id
        {"subject": " P1 ", "predicate": "hrn:usesDataset", "object": "benchy", "source_id": "[p1-abstract]",
         "evidence": "evaluate on BenchY"},
        # by (s, p, o) alone: takes the D triple's source
        {"subject": "p2", "predicate": "comparedWith", "object": "The Detector", "source_id": "p1-s2-p1"},
        # a real passage of the corpus, but not a triple of D
        {"subject": "p1", "predicate": "scores", "object": "sentences", "source_id": "p1-s2-p1",
         "evidence": "scores each generated sentence"},
        {"triple_id": "t99999", "subject": "p1", "predicate": "claims", "object": "x", "source_id": "p1-abstract"}]}
    with kb.connect(index) as conn:
        checked = contract.check(document, conn, doc=DOC)
        corpus_mode = contract.check(document, conn)
    assert checked.ok and [t.triple_id for t in checked.triples] == ["t00002", "t00001", "t00003"]
    assert checked.triples[0].object == "hallucination" and checked.triples[0].evidence.endswith("retrieval")
    assert checked.triples[2].source_id == "p2-abstract" and checked.triples[2].evidence == "compared with the detector"
    assert checked.not_in_doc == 2 and [item["why"] for item in checked.dropped] == [contract.NOT_IN_DOC] * 2
    assert checked.doc_matches == {"triple_id": 1, "spo_source": 1, "spo": 1}
    # Without D (corpus mode) only the source ids are checked: the non-D triples stay.
    assert corpus_mode.not_in_doc == 0 and len(corpus_mode.triples) == 5


def test_the_triple_cap_is_reported_and_zero_means_no_cap(index: Path) -> None:
    document = {"answer": "x [p1-abstract].", "triples": [
        {"subject": "p1", "predicate": f"rel{i}", "object": "x", "source_id": "p1-abstract"} for i in range(5)]}
    with kb.connect(index) as conn:
        capped = contract.check(document, conn, 3)
        uncapped = contract.check(document, conn, 0)
        default = contract.check(document, conn)
    assert len(capped.triples) == 3 and capped.truncated == 2
    assert capped.dropped[0]["why"] == "more than 3 triples (output.max_triples)"
    assert len(uncapped.triples) == 5 and uncapped.truncated == 0
    assert default.truncated == 0                                  # the default (500) does not bind


def test_kbcheck_in_doc_mode_names_the_triples_that_are_not_ds(index: Path, tmp_path: Path, monkeypatch, capsys) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps(DOC))
    monkeypatch.setenv("XH_DOC", str(doc))
    path = tmp_path / "answer.json"
    path.write_text(json.dumps({"answer": "x [p1-abstract].", "triples": [
        {k: DOC[0][k] for k in ("subject", "predicate", "object", "source_id", "evidence")},
        {"subject": "p1", "predicate": "scores", "object": "sentences", "source_id": "p1-s2-p1"}]}))
    tools.kbcheck([str(path)])
    out = capsys.readouterr().out
    assert out.startswith("ok: 1 triples kept, 1 dropped (1 not in the evidence document)")
    assert "dropped (not in the evidence document)" in out


def test_kbdoc_pages_by_max_hits_as_its_help_says(tmp_path: Path, monkeypatch, capsys) -> None:
    doc = tmp_path / "doc.json"
    doc.write_text(json.dumps(DOC))
    monkeypatch.setenv("XH_DOC", str(doc))
    monkeypatch.setenv("XH_MAX_HITS", "2")
    tools.kbdoc([])
    assert "(3 triples; 1 more: --offset 2)" in capsys.readouterr().out
    monkeypatch.delenv("XH_MAX_HITS")
    with pytest.raises(SystemExit):
        tools.kbdoc(["--help"])
    assert "default: knowledge_base.max_hits" in capsys.readouterr().out

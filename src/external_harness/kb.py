"""The knowledge base the agent can read: the corpus files and the ontology in one SQLite file.

``build`` reads papers.jsonl, passages.jsonl, facts.jsonl and the ontology Turtle files once and
writes ``kb.sqlite`` (full-text search with SQLite FTS5 and BM25 ranking, no extra service). The
agent's commands (tools.py) open it read-only, so each command starts in milliseconds. The index
records a fingerprint of its inputs; ``ensure`` rebuilds it when a file changed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CORPUS_FILES = ("papers.jsonl", "passages.jsonl", "facts.jsonl")
OPTIONAL_FILES = ("references.jsonl",)
INDEX_NAME = "kb.sqlite"
SCHEMA_VERSION = "1"

_TABLES = """
create table meta(key text primary key, value text);
create table papers(paper_id text primary key, title text, year text, venue text, authors text, doi text,
                    dblp_iri text);
create table passages(passage_id text primary key, paper_id text, section text, para_idx integer, text text);
create virtual table passages_fts using fts5(passage_id unindexed, section, text, tokenize='porter unicode61');
create table facts(fact_id text, subject text, predicate text, object text, source_id text, in_corpus integer);
create index facts_subject on facts(subject);
create index facts_object on facts(object);
create index facts_source on facts(source_id);
create table terms(name text, prefixed text, iri text, kind text, domain text, range text, parent text,
                   label text, comment text);
"""
_STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of", "on",
         "or", "that", "the", "this", "to", "was", "what", "which", "who", "why", "with", "do", "does",
         "did", "papers", "paper"}


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def fingerprint(corpus_dir: Path, ontology: list[Path]) -> str:
    """Names, sizes and modification times of every input: cheap, and changes when a file does."""
    parts = [SCHEMA_VERSION]
    for path in [*(corpus_dir / name for name in (*CORPUS_FILES, *OPTIONAL_FILES)), *ontology]:
        if path.exists():
            stat = path.stat()
            parts.append(f"{path.resolve()}:{stat.st_size}:{int(stat.st_mtime)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _text(value: Any) -> str:
    if isinstance(value, list):
        return " > ".join(str(item) for item in value)
    return "" if value is None else str(value)


def _terms(ontology: list[Path]) -> list[tuple[str, ...]]:
    """Classes and properties of the ontology, one row each: name, prefixed name, kind, domain, range,
    parent, label and comment (the first rdfs:comment / skos:definition found)."""
    from rdflib import OWL, RDF, RDFS, Graph, URIRef
    from rdflib.namespace import SKOS

    graph = Graph()
    for path in ontology:
        graph.parse(path, format="turtle")

    def short(node: Any) -> str:
        if node is None:
            return ""
        try:
            return graph.namespace_manager.normalizeUri(node) if isinstance(node, URIRef) else str(node)
        except Exception:
            return str(node)

    kinds = {OWL.Class: "class", RDFS.Class: "class", OWL.ObjectProperty: "object property",
             OWL.DatatypeProperty: "datatype property", RDF.Property: "property"}
    rows: dict[str, tuple[str, ...]] = {}
    for rdf_type, kind in kinds.items():
        for iri in graph.subjects(RDF.type, rdf_type):
            if not isinstance(iri, URIRef):
                continue
            existing = rows.get(str(iri))
            if existing and existing[3] != "property":       # keep the most specific kind
                continue
            is_class = kind == "class"
            parent = graph.value(iri, RDFS.subClassOf if is_class else RDFS.subPropertyOf)
            name = re.split(r"[#/]", str(iri))[-1]
            comment = graph.value(iri, RDFS.comment) or graph.value(iri, SKOS.definition) or ""
            rows[str(iri)] = (name, short(iri), str(iri), kind, short(graph.value(iri, RDFS.domain)),
                              short(graph.value(iri, RDFS.range)), short(parent if isinstance(parent, URIRef) else None),
                              str(graph.value(iri, RDFS.label) or ""), " ".join(str(comment).split())[:300])
    return sorted(rows.values(), key=lambda row: (row[3] != "class", row[0].casefold()))


def build(corpus_dir: Path, ontology: list[Path], index_dir: Path) -> dict[str, int]:
    """Write ``index_dir/kb.sqlite`` from the corpus files and the ontology; returns row counts."""
    missing = [name for name in CORPUS_FILES if not (corpus_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{corpus_dir} lacks {', '.join(missing)}")
    for path in ontology:
        if not path.is_file():
            raise FileNotFoundError(f"ontology file {path} does not exist")
    index_dir.mkdir(parents=True, exist_ok=True)
    target = index_dir / INDEX_NAME
    partial = index_dir / f"{INDEX_NAME}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial"   # never shared
    conn = sqlite3.connect(partial)
    try:
        conn.executescript(_TABLES)
        papers = [(str(row["paper_id"]), _text(row.get("title")), _text(row.get("year")), _text(row.get("venue")),
                   json.dumps(row.get("authors") or [], ensure_ascii=False), _text(row.get("doi")),
                   _text(row.get("dblp_iri"))) for row in _jsonl(corpus_dir / "papers.jsonl")]
        conn.executemany("insert or replace into papers values (?,?,?,?,?,?,?)", papers)
        passages = [(str(row["passage_id"]), str(row["paper_id"]), _text(row.get("section_path")),
                     int(row.get("para_idx") or 0), _text(row.get("text"))) for row in _jsonl(corpus_dir / "passages.jsonl")]
        conn.executemany("insert or replace into passages values (?,?,?,?,?)", passages)
        conn.executemany("insert into passages_fts values (?,?,?)", [(row[0], row[2], row[4]) for row in passages])
        facts = [(_text(row.get("fact_id")), str(row.get("subject")), str(row.get("predicate")), _text(row.get("object")),
                  _text(row.get("source_id")), 1 if row.get("in_corpus") else 0)
                 for row in _jsonl(corpus_dir / "facts.jsonl")]
        conn.executemany("insert into facts values (?,?,?,?,?,?)", facts)
        terms = _terms(ontology) if ontology else []
        conn.executemany("insert into terms values (?,?,?,?,?,?,?,?,?)", terms)
        conn.executemany("insert into meta values (?,?)", [
            ("fingerprint", fingerprint(corpus_dir, ontology)), ("schema_version", SCHEMA_VERSION),
            ("ontology_files", json.dumps([path.name for path in ontology]))])
        conn.commit()
    finally:
        conn.close()
    partial.replace(target)
    return {"papers": len(papers), "passages": len(passages), "facts": len(facts), "terms": len(terms)}


def _current(target: Path, corpus_dir: Path, ontology: list[Path]) -> bool:
    if not target.is_file():
        return False
    try:
        with connect(target) as conn:
            return meta(conn).get("fingerprint") == fingerprint(corpus_dir, ontology)
    except sqlite3.Error:
        return False


def ensure(corpus_dir: Path, ontology: list[Path], index_dir: Path) -> Path:
    """The index path, rebuilt first when missing or out of date. Concurrent runs wait for one
    rebuild (a file lock) instead of building over each other."""
    target = index_dir / INDEX_NAME
    if _current(target, corpus_dir, ontology):
        return target
    index_dir.mkdir(parents=True, exist_ok=True)
    with (index_dir / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not _current(target, corpus_dir, ontology):
            build(corpus_dir, ontology, index_dir)
    return target


def connect(path: Path) -> sqlite3.Connection:
    """Read-only: the agent cannot change the knowledge base."""
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def meta(conn: sqlite3.Connection) -> dict[str, str]:
    return {row[0]: row[1] for row in conn.execute("select key, value from meta")}


def passage_texts(conn: sqlite3.Connection, ids: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        found |= {row[0]: row[1] for row in conn.execute(
            f"select passage_id, text from passages where passage_id in ({marks})", chunk)}
    return found


def match_expression(query: str) -> str:
    """An FTS5 query from free text: "quoted phrases" stay phrases, other words are OR-ed (BM25
    ranks passages with more and rarer matches first). Empty when nothing searchable is left."""
    phrases = re.findall(r'"([^"]+)"', query)
    rest = re.sub(r'"[^"]*"', " ", query)
    words = [word for word in re.findall(r"\w+", rest.lower()) if word not in _STOP and len(word) > 1]
    parts = []
    for phrase in phrases:
        tokens = re.findall(r"\w+", phrase)
        if tokens:
            parts.append('"' + " ".join(tokens) + '"')
    parts += [f'"{word}"' for word in dict.fromkeys(words)]
    return " OR ".join(parts)


def search(conn: sqlite3.Connection, query: str, k: int = 10, paper: str | None = None) -> list[sqlite3.Row]:
    expression = match_expression(query)
    if not expression:
        return []
    sql = ("select f.passage_id, p.paper_id, p.section, p.text, pa.title, pa.year, pa.venue, bm25(passages_fts) as score "
           "from passages_fts f join passages p on p.passage_id = f.passage_id "
           "left join papers pa on pa.paper_id = p.paper_id where passages_fts match ?")
    args: list[Any] = [expression]
    if paper:
        sql += " and p.paper_id = ?"
        args.append(paper)
    sql += " order by score limit ?"
    args.append(k)
    return conn.execute(sql, args).fetchall()


def known_source_ids(conn: sqlite3.Connection, ids: list[str]) -> set[str]:
    """Which of ``ids`` are passage ids or fact source ids."""
    found: set[str] = set()
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        marks = ",".join("?" * len(chunk))
        found |= {row[0] for row in conn.execute(f"select passage_id from passages where passage_id in ({marks})", chunk)}
        found |= {row[0] for row in conn.execute(f"select source_id from facts where source_id in ({marks})", chunk)}
    return found


def stats(conn: sqlite3.Connection) -> dict[str, int]:
    return {table: conn.execute(f"select count(*) from {table}").fetchone()[0]
            for table in ("papers", "passages", "facts", "terms")}

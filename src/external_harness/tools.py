"""The agent's commands. They are the only way into the knowledge base, and they only read.

    kbsearch "words or \"a phrase\"" [-k 10] [--paper ID]   full-text search over passages (BM25)
    kbread ID [--section WORD]                               a passage, or a paper's metadata and passages
    kbpapers [--venue V] [--year Y] [--title WORD] [--author NAME] [--count]
    kbfacts PAPER_ID | --cited-by PAPER_ID | --author NAME   DBLP facts and citations inside the corpus
    onto [WORD] [--kind class|property]                      classes and properties of the ontology
    kbcheck [answer.json]                                    checks the answer file against the contract
    kbdoc --overview | kbdoc "words" | kbdoc --predicate P [--paper ID] [--offset N] | kbdoc --id T...
                                                             the evidence document, when it is the input

The runner sets XH_KB_DB (the index built by ``xh index``) and, when an evidence document is the run's input,
XH_DOC. Output is plain text, one record per line,
cut to a readable length, so the agent's context stays small.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path

from . import contract, kb


def _conn() -> sqlite3.Connection:
    path = os.environ.get("XH_KB_DB", "")
    if not path or not Path(path).is_file():
        sys.exit("knowledge base not found: XH_KB_DB is not set or the index is missing (run `xh index`)")
    return kb.connect(Path(path))


def _cut(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _has_words(haystack: str, needle: str) -> bool:
    """Whole-word match that ignores case and punctuation ("ACL" matches "ACL 2025", not "NAACL")."""
    wanted = " ".join(_words(needle))
    return bool(wanted) and f" {wanted} " in f" {' '.join(_words(haystack))} "


def _title_has(title: str, term: str) -> bool:
    """Word-prefix match: "hallucination" also finds "Hallucinations"."""
    wanted = " ".join(_words(term))
    return bool(wanted) and f" {wanted}" in f" {' '.join(_words(title))}"


def kbsearch(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kbsearch", description="Full-text search (BM25) over the corpus passages.")
    parser.add_argument("query", help='words to search for; put exact phrases in double quotes')
    parser.add_argument("-k", type=int, default=int(os.environ.get("XH_MAX_HITS", "10")), help="number of hits")
    parser.add_argument("--paper", help="only passages of this paper id")
    args = parser.parse_args(argv)
    with _conn() as conn:
        rows = kb.search(conn, args.query, max(1, min(args.k, 50)), args.paper)
    if not rows:
        print("no passage matches")
        return
    for row in rows:
        print(f"{row['passage_id']} | {row['paper_id']} | {row['year'] or '?'} | {_cut(row['venue'] or '', 30)} | "
              f"{_cut(row['title'] or '', 90)} | {_cut(row['section'] or '', 40)}")
        print(f"    {_cut(row['text'], 320)}")


def kbread(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kbread", description="Read a passage (by passage id) or list a paper's "
                                                                "metadata and passages (by paper id).")
    parser.add_argument("id")
    parser.add_argument("--section", help="with a paper id: only passages whose section contains this word")
    args = parser.parse_args(argv)
    with _conn() as conn:
        passage = conn.execute("select p.*, pa.title from passages p left join papers pa using (paper_id) "
                               "where passage_id = ?", (args.id,)).fetchone()
        if passage:
            print(f"{passage['passage_id']} | paper {passage['paper_id']} | {passage['title'] or ''} | "
                  f"section: {passage['section'] or '?'}")
            print(passage["text"])
            return
        paper = conn.execute("select * from papers where paper_id = ?", (args.id,)).fetchone()
        if not paper:
            sys.exit(f"no passage or paper with id {args.id!r} (ids come from kbsearch or kbpapers)")
        authors = ", ".join(json.loads(paper["authors"] or "[]"))
        print(f"{paper['paper_id']} | {paper['title']} | {paper['year']} | {paper['venue']}")
        print(f"authors: {authors or '?'}")
        rows = conn.execute("select passage_id, section, text from passages where paper_id = ? order by rowid",
                            (args.id,)).fetchall()
        for row in rows:
            if args.section and not _has_words(row["section"] or "", args.section):
                continue
            print(f"  {row['passage_id']} | {_cut(row['section'] or '', 40)} | {_cut(row['text'], 110)}")


def kbpapers(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kbpapers", description="List corpus papers by DBLP metadata.")
    parser.add_argument("--venue", help='whole words of the venue, e.g. "ICML" or "NeurIPS 2023"')
    parser.add_argument("--year", help="publication year")
    parser.add_argument("--title", help="a word or phrase the title contains (word prefix)")
    parser.add_argument("--author", help="part of an author name")
    parser.add_argument("--count", action="store_true", help="print only the number of matching papers")
    args = parser.parse_args(argv)
    with _conn() as conn:
        rows = conn.execute("select * from papers order by year, paper_id").fetchall()
    chosen = []
    for row in rows:
        venue = str(row["venue"] or "")
        if args.venue:
            year_words = [word for word in _words(args.venue) if re.fullmatch(r"(19|20)\d\d", word)]
            name = " ".join(word for word in _words(args.venue) if word not in year_words)
            if (name and not _has_words(venue, name)) or any(
                    word not in _words(venue) and word != str(row["year"]) for word in year_words):
                continue
        if args.year and str(row["year"]) != args.year.strip():
            continue
        if args.title and not _title_has(row["title"] or "", args.title):
            continue
        if args.author and not any(args.author.casefold() in name.casefold()
                                   for name in json.loads(row["authors"] or "[]")):
            continue
        chosen.append(row)
    if args.count:
        print(len(chosen))
        return
    for row in chosen:
        print(f"{row['paper_id']} | {row['year']} | {_cut(row['venue'] or '', 40)} | {_cut(row['title'] or '', 120)}")
    print(f"({len(chosen)} papers)")


def kbfacts(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kbfacts", description="DBLP facts (title, author, year, venue) and "
                                                                 "citations between corpus papers.")
    parser.add_argument("paper", nargs="?", help="a paper id: its facts and the corpus papers it cites")
    parser.add_argument("--cited-by", dest="cited", help="a paper id: the corpus papers that cite it")
    parser.add_argument("--author", help="an author name: their corpus papers")
    args = parser.parse_args(argv)
    if not (args.paper or args.cited or args.author):
        parser.error("give a paper id, --cited-by or --author")
    with _conn() as conn:
        titles = {row[0]: row[1] for row in conn.execute("select paper_id, title from papers")}

        def label(value: str) -> str:
            return f"{value} ({_cut(titles[value], 70)})" if value in titles else value

        if args.paper:
            rows = conn.execute("select * from facts where subject = ? order by predicate, object", (args.paper,))
            printed = 0
            for row in rows:
                if row["predicate"] == "cites" and row["object"] not in titles:
                    continue                                   # references outside the corpus
                print(f"{row['source_id']} | {row['predicate']} | {label(row['object'])}")
                printed += 1
            if not printed:
                print(f"no facts for {args.paper!r}")
        if args.cited:
            rows = conn.execute("select * from facts where predicate = 'cites' and object = ? order by subject",
                                (args.cited,)).fetchall()
            for row in rows:
                print(f"{row['source_id']} | {label(row['subject'])} cites {args.cited}")
            print(f"({len({row['subject'] for row in rows})} citing corpus papers)")
        if args.author:
            rows = conn.execute("select * from facts where predicate = 'author' order by subject").fetchall()
            found = [row for row in rows if args.author.casefold() in str(row["object"]).casefold()]
            for row in found:
                print(f"{row['source_id']} | {label(row['subject'])} | author {row['object']}")
            print(f"({len({row['subject'] for row in found})} papers)")


def onto(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="onto", description="Classes and properties of the ontology the "
                                                              "knowledge graph uses.")
    parser.add_argument("word", nargs="?", help="only terms whose name, label or comment contains this")
    parser.add_argument("--kind", choices=["class", "property"])
    args = parser.parse_args(argv)
    with _conn() as conn:
        rows = conn.execute("select * from terms").fetchall()
    shown = 0
    for row in rows:
        if args.kind == "class" and row["kind"] != "class":
            continue
        if args.kind == "property" and row["kind"] == "class":
            continue
        if args.word and args.word.casefold() not in f"{row['name']} {row['label']} {row['comment']}".casefold():
            continue
        shape = f"{row['domain'] or '?'} -> {row['range'] or '?'}" if row["kind"] != "class" else ""
        parent = f" (sub of {row['parent']})" if row["parent"] else ""
        print(f"{row['prefixed']} | {row['kind']}{parent} | {shape} | {_cut(row['comment'], 120)}")
        shown += 1
    print(f"({shown} terms)")


def kbcheck(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kbcheck", description="Check answer.json against the output contract.")
    parser.add_argument("path", nargs="?", default="answer.json")
    args = parser.parse_args(argv)
    path = Path(args.path)
    document = contract.parse(path.read_text(encoding="utf-8")) if path.is_file() else None
    with _conn() as conn:
        checked = contract.check(document, conn, int(os.environ.get("XH_MAX_TRIPLES", "60")))
    if not checked.ok:
        sys.exit(f"invalid: {checked.problem}")
    print(f"ok: {len(checked.triples)} triples kept, {len(checked.dropped)} dropped")
    for item in checked.dropped[:10]:
        print(f"  dropped ({item['why']}): {json.dumps(item['triple'], ensure_ascii=False)[:160]}")


def _doc() -> list[dict]:
    path = os.environ.get("XH_DOC", "")
    if not path or not Path(path).is_file():
        sys.exit("no evidence document: this run's input is the corpus (use kbsearch, kbread, ...)")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _doc_line(row: dict) -> str:
    return (f"{row['triple_id']} | {row['subject']} | {row['predicate']} | {_cut(row['object'], 80)} | "
            f"{row['source_id']}\n    {_cut(row.get('evidence') or '', 240)}")


def kbdoc(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="kbdoc", description="The evidence document: the triples (each with "
                                                               "the passage that states it) that are this run's input.")
    parser.add_argument("query", nargs="?", help="words to rank the triples by (subject, relation, object, passage)")
    parser.add_argument("--overview", action="store_true", help="counts, relations, papers and interpretations")
    parser.add_argument("--predicate", help="only triples with this relation")
    parser.add_argument("--paper", help="only triples of this paper id")
    parser.add_argument("--interpretation", help="only triples of this interpretation id")
    parser.add_argument("--id", nargs="+", help="these triple ids, with their full evidence")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("-k", type=int, default=int(os.environ.get("XH_MAX_HITS", "25")), help="triples per page")
    args = parser.parse_args(argv)
    rows = _doc()
    if args.overview:
        from collections import Counter

        print(f"{len(rows)} triples")
        for name, count in Counter(row.get("interpretation") or "-" for row in rows).most_common():
            print(f"interpretation {name}: {count} triples")
        print("relations: " + ", ".join(f"{name} ({count})" for name, count in
                                        Counter(row["predicate"] for row in rows).most_common(40)))
        print("papers: " + ", ".join(f"{name} ({count})" for name, count in
                                     Counter(row.get("paper_id") or "?" for row in rows).most_common(60)))
        return
    if args.id:
        wanted = set(args.id)
        for row in rows:
            if row["triple_id"] in wanted:
                print(f"{row['triple_id']} | {row['subject']} | {row['predicate']} | {row['object']} | "
                      f"{row['source_id']}\n    {row.get('evidence') or ''}")
        return
    chosen = [row for row in rows
              if (not args.predicate or row["predicate"].casefold() == args.predicate.casefold())
              and (not args.paper or (row.get("paper_id") or "") == args.paper)
              and (not args.interpretation or (row.get("interpretation") or "") == args.interpretation)]
    if args.query:
        wanted = set(_words(args.query))
        scored = [(len(wanted & set(_words(f"{r['subject']} {r['predicate']} {r['object']} {r.get('evidence')}"))), i)
                  for i, r in enumerate(chosen)]
        chosen = [chosen[i] for score, i in sorted(scored, key=lambda item: (-item[0], item[1])) if score]
    page = chosen[max(0, args.offset): max(0, args.offset) + max(1, args.k)]
    for row in page:
        print(_doc_line(row))
    rest = len(chosen) - max(0, args.offset) - len(page)
    print(f"({len(chosen)} triples" + (f"; {rest} more: --offset {max(0, args.offset) + len(page)}" if rest > 0 else "")
          + ")")

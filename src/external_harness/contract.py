"""What the agent must deliver (answer.json) and how it is checked.

    {"answer": "Short prose, each claim followed by its source id [p12-s3-p2].",
     "triples": [{"subject": "p12", "predicate": "usesDataset", "object": "SQuAD",
                  "source_id": "p12-s3-p2", "evidence": "we evaluate on SQuAD"}]}

A triple is kept only when every field is filled and its source_id is a passage id or a DBLP fact
source id of the knowledge base; dropped triples are listed with the reason, never silently lost.

With an evidence document D as the run's input, a triple is kept only when it is one of D's triples:
matched by ``triple_id`` when the agent gives it, else by its normalised (subject, predicate, object) and
source_id, else by its (subject, predicate, object) alone (it then takes the source of that D triple).
The others are dropped as "not in the evidence document" and counted (``not_in_doc``).

``max_triples`` (0 = no cap) cuts the list after that many kept triples; ``truncated`` counts the cut ones.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any

from pydantic import BaseModel, Field

from . import kb

FIELDS = ("subject", "predicate", "object", "source_id", "evidence")
_EMPTY = {"", "null", "none", "n/a", "unknown"}


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    source_id: str
    evidence: str = ""
    triple_id: str | None = None          # the evidence document's triple, in doc mode


class Checked(BaseModel):
    ok: bool
    answer: str = ""
    triples: list[Triple] = Field(default_factory=list)
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    problem: str | None = None
    evidence_not_in_source: int = 0      # kept triples whose evidence quote is not in their passage
    not_in_doc: int = 0                  # doc mode: triples dropped because they are not D's
    truncated: int = 0                   # triples cut by max_triples
    doc_matches: dict[str, int] = Field(default_factory=dict)   # doc mode: kept triples by how they matched D


def parse(text: str) -> dict[str, Any] | None:
    """The JSON object in ``text``: the whole text, a ```json block, or the outermost {...}."""
    text = (text or "").strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        candidates.append(fenced.group(1))
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{"): text.rindex("}") + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def check(document: dict[str, Any] | None, conn: sqlite3.Connection | None, max_triples: int = 500,
          doc: list[dict[str, Any]] | None = None) -> Checked:
    """Validate a parsed answer.json against the contract and the knowledge base (and, with ``doc``, the
    evidence document D: only D's triples are kept)."""
    if document is None:
        return Checked(ok=False, problem="answer.json is missing or not a JSON object")
    answer = document.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        return Checked(ok=False, problem='"answer" must be a non-empty string')
    raw = document.get("triples", [])
    if not isinstance(raw, list):
        return Checked(ok=False, answer=answer.strip(), problem='"triples" must be a list')
    candidates: list[tuple[dict[str, Any], Triple]] = []
    dropped: list[dict[str, Any]] = []
    matcher = _DocMatcher(doc) if doc is not None else None
    for item in raw:
        if not isinstance(item, dict):
            dropped.append({"triple": item, "why": "not an object"})
            continue
        values = {name: " ".join(str(item.get(name) or "").split()) for name in FIELDS}
        values["source_id"] = values["source_id"].strip("[]")
        if matcher is not None:
            matched = matcher.match(item, values)
            if matched is None:
                dropped.append({"triple": item, "why": NOT_IN_DOC})
                continue
            values = matched
        empty = [name for name in FIELDS[:4] if values[name].casefold() in _EMPTY]
        if empty:
            dropped.append({"triple": item, "why": f"empty {', '.join(empty)}"})
            continue
        candidates.append((item, Triple(**values)))
    known = kb.known_source_ids(conn, [triple.source_id for _, triple in candidates]) if conn is not None else None
    kept: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    truncated = 0
    for item, triple in candidates:
        key = (triple.subject.casefold(), triple.predicate, triple.object.casefold())
        if known is not None and triple.source_id not in known:
            dropped.append({"triple": item, "why": f"unknown source_id {triple.source_id!r}"})
        elif key in seen:
            dropped.append({"triple": item, "why": "duplicate"})
        elif max_triples and len(kept) >= max_triples:
            dropped.append({"triple": item, "why": f"more than {max_triples} triples (output.max_triples)"})
            truncated += 1
        else:
            seen.add(key)
            kept.append(triple)
    texts = kb.passage_texts(conn, [triple.source_id for triple in kept]) if conn is not None else {}
    unmatched = sum(1 for triple in kept if triple.evidence and triple.source_id in texts
                    and _normal(triple.evidence) not in _normal(texts[triple.source_id]))
    return Checked(ok=True, answer=answer.strip(), triples=kept, dropped=dropped, evidence_not_in_source=unmatched,
                   not_in_doc=sum(1 for item in dropped if item["why"] == NOT_IN_DOC), truncated=truncated,
                   doc_matches=dict(matcher.counts) if matcher is not None else {})


NOT_IN_DOC = "not in the evidence document"


def _key(text: str) -> str:
    return " ".join(str(text or "").split()).casefold()


def _predicate_key(text: str) -> str:
    """``hrn:usesDataset``, ``https://…#usesDataset`` and ``usesDataset`` are one predicate."""
    return _key(re.split(r"[#/:]", str(text or "").strip())[-1])


class _DocMatcher:
    """Finds the evidence document's triple an answer triple stands for (doc mode)."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.by_id: dict[str, dict[str, Any]] = {}
        self.by_spo_source: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self.by_spo: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            ident = str(row.get("triple_id") or row.get("id") or "").strip()
            if ident:
                self.by_id.setdefault(ident, row)
            spo = self._spo(row)
            self.by_spo_source.setdefault((*spo, str(row.get("source_id") or "").strip().strip("[]")), row)
            self.by_spo.setdefault(spo, row)

    @staticmethod
    def _spo(row: dict[str, Any]) -> tuple[str, str, str]:
        return _key(row.get("subject")), _predicate_key(row.get("predicate")), _key(row.get("object"))

    def match(self, item: dict[str, Any], values: dict[str, str]) -> dict[str, Any] | None:
        """The answer triple's fields when it is one of D's (with D's triple id, and D's subject, predicate,
        object and source when it was matched by id or by (s, p, o) alone); None when it is not D's."""
        ident = str(item.get("triple_id") or item.get("id") or "").strip()
        spo = self._spo(values)
        if ident and ident in self.by_id:
            row, how = self.by_id[ident], "triple_id"
        elif (*spo, values["source_id"]) in self.by_spo_source:
            row, how = self.by_spo_source[(*spo, values["source_id"])], "spo_source"
        elif spo in self.by_spo:
            row, how = self.by_spo[spo], "spo"
        else:
            return None
        self.counts[how] = self.counts.get(how, 0) + 1
        canonical = {name: " ".join(str(row.get(name) or "").split()) for name in FIELDS}
        canonical["source_id"] = canonical["source_id"].strip("[]")
        if how == "spo_source":
            canonical = {**values}
        canonical["evidence"] = values["evidence"] or canonical["evidence"]
        canonical["triple_id"] = str(row.get("triple_id") or row.get("id") or "") or None
        return canonical


def _normal(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))

"""What the agent must deliver (answer.json) and how it is checked.

    {"answer": "Short prose, each claim followed by its source id [p12-s3-p2].",
     "triples": [{"subject": "p12", "predicate": "usesDataset", "object": "SQuAD",
                  "source_id": "p12-s3-p2", "evidence": "we evaluate on SQuAD"}]}

A triple is kept only when every field is filled and its source_id is a passage id or a DBLP fact
source id of the knowledge base; dropped triples are listed with the reason, never silently lost.
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


class Checked(BaseModel):
    ok: bool
    answer: str = ""
    triples: list[Triple] = Field(default_factory=list)
    dropped: list[dict[str, Any]] = Field(default_factory=list)
    problem: str | None = None


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


def check(document: dict[str, Any] | None, conn: sqlite3.Connection | None, max_triples: int = 60) -> Checked:
    """Validate a parsed answer.json against the contract and the knowledge base."""
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
    for item in raw:
        if not isinstance(item, dict):
            dropped.append({"triple": item, "why": "not an object"})
            continue
        values = {name: " ".join(str(item.get(name) or "").split()) for name in FIELDS}
        empty = [name for name in FIELDS[:4] if values[name].casefold() in _EMPTY]
        if empty:
            dropped.append({"triple": item, "why": f"empty {', '.join(empty)}"})
            continue
        values["source_id"] = values["source_id"].strip("[]")
        candidates.append((item, Triple(**values)))
    known = kb.known_source_ids(conn, [triple.source_id for _, triple in candidates]) if conn is not None else None
    kept: list[Triple] = []
    seen: set[tuple[str, str, str]] = set()
    for item, triple in candidates:
        key = (triple.subject.casefold(), triple.predicate, triple.object.casefold())
        if known is not None and triple.source_id not in known:
            dropped.append({"triple": item, "why": f"unknown source_id {triple.source_id!r}"})
        elif key in seen:
            dropped.append({"triple": item, "why": "duplicate"})
        elif len(kept) >= max_triples:
            dropped.append({"triple": item, "why": f"more than {max_triples} triples"})
        else:
            seen.add(key)
            kept.append(triple)
    return Checked(ok=True, answer=answer.strip(), triples=kept, dropped=dropped)

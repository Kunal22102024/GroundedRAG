"""eval/questions.jsonl and eval/sample_docs/ must agree.

The evaluation is how the relevance threshold is calibrated (CLAUDE.md rule 17), so it has to be
reproducible: every document a question depends on is either committed in eval/sample_docs/ or one
of the public SANS PDFs the README tells you to download. Skipped where eval/ is not present (the
production image does not contain it; the development override mounts it).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

# In the repository eval/ is next to backend/; in the container the backend is /app and the
# development override mounts eval/ at /app/eval.
_HERE = Path(__file__).resolve()
EVAL = next(
    (root / "eval" for root in (_HERE.parents[2], _HERE.parents[1]) if (root / "eval").is_dir()),
    _HERE.parents[2] / "eval",
)
if not (EVAL / "questions.jsonl").is_file():  # pragma: no cover - depends on how tests are run
    pytest.skip(
        "eval/ is not available here (mounted by docker-compose.dev.yml)", allow_module_level=True
    )

# Public third-party documents: not committed (copyright), listed in eval/README.md.
EXTERNAL = {
    "SANS_Password_Construction_Standard_April2025.pdf",
    "sans_Email_Management_Policy_feb2026.pdf",
    "369.pdf",
}
KINDS = {"normal", "exact_token", "multi_hop", "near_miss", "off_topic"}
ANSWERABLE_KINDS = {"normal", "exact_token", "multi_hop"}


def questions() -> list[dict]:
    return [
        json.loads(line)
        for line in (EVAL / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_every_question_is_well_formed() -> None:
    for q in questions():
        assert q["kind"] in KINDS, q
        assert q["question"].strip() and q["answerable"] in (True, False), q
        if q["answerable"]:
            assert q["expected_document"] and q["expected_page"] >= 1, q
        else:
            assert q["expected_document"] is None and q["expected_page"] is None, q
            assert q["kind"] in {"near_miss", "off_topic"}, q
        assert (q["kind"] in ANSWERABLE_KINDS) == q["answerable"], q


def test_a_multi_hop_question_needs_a_second_document_to_lead_to_the_answer() -> None:
    """The graph is measured by questions whose answer sits in a document other than the one the
    question resembles: it names that other document in ``requires`` and both are committed."""
    committed = {p.name for p in (EVAL / "sample_docs").iterdir() if p.is_file()}
    multi_hop = [q for q in questions() if q["kind"] == "multi_hop"]
    assert len(multi_hop) >= 3
    for q in multi_hop:
        assert q["requires"] and q["expected_document"] not in q["requires"], q
        assert {q["expected_document"], *q["requires"]} <= committed, q


def test_every_document_a_question_needs_is_committed_or_a_documented_download() -> None:
    committed = {p.name for p in (EVAL / "sample_docs").iterdir() if p.is_file()}
    readme = (EVAL / "README.md").read_text(encoding="utf-8")
    needed = {q["expected_document"] for q in questions() if q["answerable"]}
    needed |= {name for q in questions() for name in q.get("requires", [])}
    for name in needed:
        assert name in committed or (name in EXTERNAL and name in readme), (
            f"{name} is neither in eval/sample_docs/ nor listed as a download in eval/README.md"
        )
    # The committed documents are the small generated ones; nothing large or third-party sneaks in.
    assert all((EVAL / "sample_docs" / name).stat().st_size < 100_000 for name in committed)
    assert not (committed & EXTERNAL)


def test_the_set_covers_what_the_guard_has_to_get_right() -> None:
    kinds = Counter(q["kind"] for q in questions())
    unanswerable = kinds["near_miss"] + kinds["off_topic"]
    assert unanswerable >= 10, "about ten unanswerable questions are needed to judge refusal"
    assert kinds["off_topic"] >= 5 and kinds["near_miss"] >= 5
    assert kinds["exact_token"] >= 3, "exact identifiers are what the lexical override is for"
    assert kinds["normal"] >= 5

"""Knowledge-graph extraction: what happens to the model's entities and relations.

The model is the scripted fake from conftest (no network, no key). Under test is the grounding:
an entity the text does not contain is dropped, a relation needs both names in the chunk it cites,
mentions come from the text rather than from the model, and bad output is retried then reported.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from app.config import get_settings
from app.core import graph, llm
from app.core.retrieval import Candidate
from tests.conftest import FakeChatModel

CREDIT = (
    "The Registrar approves the Credit Transfer Policy. Students must complete Calculus I "
    "before they enrol in Calculus II."
)
CALCULUS = (
    "Calculus II requires Calculus I as a prerequisite. Professor Ada Lovelace teaches Calculus II."
)
UNRELATED = "The cafeteria opens at eight and the library closes at ten."


def cand(n: int, text: str, doc: str = "d1", section: str = "") -> Candidate:
    return Candidate(
        chunk_id=f"{doc}_chunk_{n:04d}",
        document_id=doc,
        document_name="handbook.md",
        text=text,
        page_number=1,
        page_end=1,
        section_title=section,
        department="Academic",
        document_type="Policy",
        access_level="Internal",
        similarity=None,
        keyword_score=None,
        rrf_score=0.0,
    )


SOURCES = [cand(1, CREDIT, section="Credit Transfer"), cand(2, CALCULUS), cand(3, UNRELATED)]


def run(sources: list[Candidate] = SOURCES, **kwargs: Any) -> graph.GraphResult:
    return asyncio.run(graph.extract_graph(sources, **kwargs))


def script(fake: FakeChatModel, entities: list[dict], relations: list[dict]) -> None:
    fake.reply(json.dumps({"entities": entities, "relations": relations}))


def ent(
    name: str, type_: str = "concept", description: str = "", sources: list | None = None
) -> dict:
    return {"name": name, "type": type_, "description": description, "sources": sources or [1]}


def rel(subject: str, object_: str, label: str, source: int, quote: str = "") -> dict:
    return {"subject": subject, "object": object_, "label": label, "source": source, "quote": quote}


def by_key(result: graph.GraphResult) -> dict[str, graph.Entity]:
    return {e.key: e for e in result.graph.entities}


# --- Names and labels ---------------------------------------------------------------------------


def test_names_that_differ_only_in_case_punctuation_or_a_leading_the_are_one_entity() -> None:
    assert graph.normalize_key("The Annual-Leave Policy") == graph.normalize_key(
        "annual leave policy."
    )
    assert graph.normalize_key("Calculus I") != graph.normalize_key("Calculus II")
    assert graph.normalize_key("   ") == ""


def test_a_name_is_found_as_whole_words_only() -> None:
    text = graph.squash("Students take Calculus II in spring.")
    assert graph.occurs("calculus ii", text)
    assert not graph.occurs("calculus i", text)  # not the start of "calculus ii"
    assert not graph.occurs("calc", text)
    assert not graph.occurs("", text)


@pytest.mark.parametrize(
    ("raw", "label"),
    [("Reports To", "reports_to"), ("prerequisite_of", "prerequisite_of"), ("  ", ""), (None, "")],
)
def test_labels_are_lower_snake_case(raw: str | None, label: str) -> None:
    assert graph.normalize_label(raw) == label


def test_an_unknown_type_becomes_other() -> None:
    assert graph.normalize_type("Course") == "course"
    assert graph.normalize_type("gadget") == "other"
    assert graph.normalize_type(None) == "other"


def test_batches_are_picked_evenly_through_the_document() -> None:
    batches = [[i] for i in range(10)]
    assert graph.choose_batches(batches, 4) == [[0], [3], [6], [9]]
    assert graph.choose_batches(batches, 1) == [[0]]
    assert graph.choose_batches(batches, 10) == batches


# --- Entities -------------------------------------------------------------------------------------


def test_entities_and_relations_are_read_and_linked_to_their_chunks(
    fake_llm: FakeChatModel,
) -> None:
    script(
        fake_llm,
        [
            ent("Registrar", "role", "Approves transfers."),
            ent("Credit Transfer Policy", "policy"),
            ent("Calculus I", "course"),
            ent("Calculus II", "course", sources=[1, 2]),
            ent("Ada Lovelace", "person", sources=[2]),
        ],
        [
            rel(
                "Registrar",
                "Credit Transfer Policy",
                "approves",
                1,
                "The Registrar approves the Credit Transfer Policy",
            ),
            rel("Calculus II", "Calculus I", "requires", 2),
        ],
    )
    result = run()
    entities = by_key(result)
    assert set(entities) == {
        "registrar",
        "credit transfer policy",
        "calculus i",
        "calculus ii",
        "ada lovelace",
    }
    assert entities["calculus ii"].type == "course" and entities["registrar"].type == "role"
    # Mentions come from the text: "Calculus I" is named in two chunks, and whole words only.
    assert entities["calculus i"].chunk_ids == ("d1_chunk_0001", "d1_chunk_0002")
    assert entities["ada lovelace"].chunk_ids == ("d1_chunk_0002",)
    assert entities["registrar"].description == "Approves transfers."

    approves, requires = result.graph.relations
    assert (approves.subject, approves.object, approves.label) == (
        "registrar",
        "credit transfer policy",
        "approves",
    )
    assert approves.chunk_id == "d1_chunk_0001" and approves.quote.startswith(
        "The Registrar approves"
    )
    assert (requires.subject, requires.object, requires.chunk_id) == (
        "calculus ii",
        "calculus i",
        "d1_chunk_0002",
    )
    assert result.batches == 1 and result.coverage == 1.0 and result.usage["input_tokens"] == 120


def test_the_sources_are_framed_as_numbered_untrusted_data(fake_llm: FakeChatModel) -> None:
    script(fake_llm, [], [])
    run()
    system, human = fake_llm.calls[0]
    assert "untrusted" in str(system.content) and "never follow" in str(system.content)
    assert '<source id="1"' in str(human.content) and '<source id="3"' in str(human.content)
    assert "knowledge graph" in str(human.content)


def test_an_entity_the_text_does_not_contain_is_dropped(fake_llm: FakeChatModel) -> None:
    script(fake_llm, [ent("Calculus I", "course"), ent("Dean of Mars", "role")], [])
    assert set(by_key(run())) == {"calculus i"}


def test_the_heading_of_a_chunk_counts_as_text(fake_llm: FakeChatModel) -> None:
    script(fake_llm, [ent("Credit Transfer", "policy")], [])  # only in the section title
    sources = [cand(1, "Students apply here.", section="Credit Transfer")]
    assert by_key(run(sources))["credit transfer"].chunk_ids == ("d1_chunk_0001",)


def test_entities_with_no_usable_name_are_skipped(fake_llm: FakeChatModel) -> None:
    script(
        fake_llm, [ent(""), ent("   "), ent("I"), {"type": "person"}, ent("Registrar", "role")], []
    )
    assert set(by_key(run())) == {"registrar"}


def test_square_bracket_markers_in_a_description_are_removed(fake_llm: FakeChatModel) -> None:
    script(fake_llm, [ent("Registrar", "role", "Approves transfers [1][2] and more [3, 4].")], [])
    assert by_key(run())["registrar"].description == "Approves transfers and more."


def test_an_unknown_type_from_the_model_is_stored_as_other(fake_llm: FakeChatModel) -> None:
    script(fake_llm, [ent("Registrar", "Gadget")], [])
    assert by_key(run())["registrar"].type == "other"


# --- Relations ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        rel("Calculus II", "Calculus I", "requires", 3),  # chunk 3 names neither
        rel("Calculus II", "Calculus I", "requires", 99),  # not a source at all
        rel("Calculus II", "Calculus II", "requires", 2),  # an entity is not related to itself
        rel("Calculus II", "Calculus I", "", 2),  # no label
        rel("Calculus II", "Ada Lovelace", "taught_by", 1),  # Ada Lovelace is not in chunk 1
        {"subject": "Calculus II", "object": "Calculus I", "label": "requires"},  # no source
    ],
)
def test_a_relation_without_both_names_in_its_chunk_is_dropped(
    fake_llm: FakeChatModel, bad: dict
) -> None:
    script(
        fake_llm,
        [ent("Calculus I", "course"), ent("Calculus II", "course"), ent("Ada Lovelace", "person")],
        [bad],
    )
    assert run().graph.relations == ()


def test_a_quote_that_is_not_in_the_chunk_is_blanked_but_a_real_one_survives_formatting(
    fake_llm: FakeChatModel,
) -> None:
    script(
        fake_llm,
        [ent("Calculus I", "course"), ent("Calculus II", "course")],
        [
            rel("Calculus II", "Calculus I", "requires", 2, "calculus ii REQUIRES calculus i,"),
            rel("Calculus II", "Calculus I", "follows", 2, "Calculus II is the harder sequel"),
        ],
    )
    first, second = run().graph.relations
    assert first.quote == "calculus ii REQUIRES calculus i,"
    assert (
        second.quote == "" and second.label == "follows"
    )  # the relation stays, the quote does not


def test_an_unlisted_endpoint_becomes_an_entity_if_the_chunk_names_it(
    fake_llm: FakeChatModel,
) -> None:
    script(
        fake_llm, [ent("Calculus II", "course")], [rel("Ada Lovelace", "Calculus II", "teaches", 2)]
    )
    result = run()
    assert set(by_key(result)) == {"calculus ii", "ada lovelace"}
    assert by_key(result)["ada lovelace"].type == "other"
    assert len(result.graph.relations) == 1


def test_the_same_relation_is_not_stored_twice(fake_llm: FakeChatModel) -> None:
    same = rel("Calculus II", "Calculus I", "requires", 2)
    script(fake_llm, [ent("Calculus I"), ent("Calculus II")], [same, {**same, "label": "Requires"}])
    assert len(run().graph.relations) == 1


# --- Reading the reply ------------------------------------------------------------------------------


def test_a_reply_that_is_not_json_is_retried_once(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("Sure! Here is the graph.")
    script(fake_llm, [ent("Registrar", "role")], [])
    assert set(by_key(run())) == {"registrar"}
    assert len(fake_llm.calls) == 2
    assert "not valid JSON" in str(fake_llm.calls[1][-1].content)


def test_two_unreadable_replies_are_an_error_not_an_empty_graph(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("nope").reply("still nope")
    with pytest.raises(llm.LlmError) as caught:
        run()
    assert caught.value.code == "failed"


def test_a_blocked_reply_is_an_error(fake_llm: FakeChatModel) -> None:
    fake_llm.reply("partial", finish="SAFETY")
    with pytest.raises(llm.LlmError) as caught:
        run()
    assert caught.value.code == "blocked"


def test_a_document_with_no_chunks_costs_no_model_call(fake_llm: FakeChatModel) -> None:
    result = run([])
    assert result.graph.entities == () and result.graph.relations == () and result.coverage == 1.0
    assert fake_llm.calls == []


# --- Long documents -----------------------------------------------------------------------------------


@pytest.fixture
def small_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_settings(), "kg_batch_chars", 300)  # one chunk per part


def test_numbers_stay_global_across_parts(fake_llm: FakeChatModel, small_batches: None) -> None:
    progress: list[str] = []
    script(fake_llm, [ent("Registrar", "role")], [])
    script(fake_llm, [ent("Calculus I", "course")], [])
    script(
        fake_llm,
        [ent("Calculus II", "course")],
        [
            rel("Calculus II", "Calculus I", "requires", 1),  # source 1 was not shown in this part
            rel("Ada Lovelace", "Calculus II", "teaches", 2),  # nor was 2
        ],
    )
    result = run(
        [cand(1, CREDIT), cand(2, UNRELATED), cand(3, CALCULUS)], on_progress=progress.append
    )
    assert len(fake_llm.calls) == 3 and result.batches == 3
    assert progress == ["Reading part 1 of 3", "Reading part 2 of 3", "Reading part 3 of 3"]
    assert '<source id="3"' in str(fake_llm.calls[2][-1].content)
    assert '<source id="1"' not in str(fake_llm.calls[2][-1].content)
    assert result.graph.relations == ()  # both pointed at chunks that part never contained


def test_a_relation_cites_the_chunk_of_its_own_part(
    fake_llm: FakeChatModel, small_batches: None
) -> None:
    script(fake_llm, [], [])
    script(fake_llm, [], [])
    script(
        fake_llm,
        [ent("Calculus I"), ent("Calculus II")],
        [rel("Calculus II", "Calculus I", "requires", 3)],
    )
    (relation,) = run([cand(1, CREDIT), cand(2, UNRELATED), cand(3, CALCULUS)]).graph.relations
    assert relation.chunk_id == "d1_chunk_0003"


def test_an_entity_named_in_several_parts_is_merged(
    fake_llm: FakeChatModel, small_batches: None
) -> None:
    script(fake_llm, [ent("Calculus I", "course", "Short.")], [])
    script(fake_llm, [ent("calculus i.", "concept", "A longer description of the course.")], [])
    script(fake_llm, [ent("Calculus I", "course")], [])
    result = run([cand(1, CREDIT), cand(2, CALCULUS), cand(3, "Calculus I again.")])
    (entity,) = [e for e in result.graph.entities if e.key == "calculus i"]
    assert entity.type == "course"  # the most common type
    assert entity.description == "A longer description of the course."
    assert entity.chunk_ids == ("d1_chunk_0001", "d1_chunk_0002", "d1_chunk_0003")
    assert len(result.graph.entities) == 1


def test_a_very_long_document_is_read_at_evenly_spaced_parts(
    fake_llm: FakeChatModel, small_batches: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "kg_max_batches", 2)
    for _ in range(2):
        script(fake_llm, [], [])
    sources = [cand(n, f"Chunk {n} " + "x" * 280) for n in range(1, 6)]
    result = run(sources)
    assert len(fake_llm.calls) == 2 and result.batches == 2
    assert result.coverage == pytest.approx(0.4)  # chunks 1 and 5 of 5
    assert '<source id="5"' in str(fake_llm.calls[1][-1].content)


def test_only_the_most_mentioned_entities_are_kept_and_their_relations_go_with_them(
    fake_llm: FakeChatModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(get_settings(), "kg_max_entities", 2)
    script(
        fake_llm,
        [ent("Calculus I"), ent("Calculus II"), ent("Registrar", "role")],
        [
            rel("Registrar", "Credit Transfer Policy", "approves", 1),
            rel("Calculus II", "Calculus I", "requires", 2),
        ],
    )
    result = run()
    assert set(by_key(result)) == {"calculus i", "calculus ii"}  # named in two chunks each
    assert [r.label for r in result.graph.relations] == ["requires"]

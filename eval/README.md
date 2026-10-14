# Evaluation

`run_eval.py` measures retrieval quality and the refusal guard over `questions.jsonl`. It is how
`RELEVANCE_THRESHOLD` was chosen, and CLAUDE.md requires running it (and comparing with the previous
result) after any change to retrieval, embeddings, the guard or the prompts. It calls the production
`retrieval.retrieve` and `guard.evaluate`, so what it reports is what a chat turn does. It makes no
LLM calls: the only cost is one embedding request per question.

## The corpus

| File | Where it comes from |
| --- | --- |
| `sample_docs/` | Five small generated documents, **committed**: `HR_Leave_Policy_2025-03-01.docx`, `Gateway_Runbook.md`, `Network_Spec.html`, `Operations_Notes.txt`, `Quarterly_Report_2026-01-15.pdf` (12 pages, one region per page). |
| three public PDFs | **Not committed** (third-party copyright). Download them and put them next to your own files in `sample_docs/` at the repository root (that folder is git-ignored): `369.pdf`, `SANS_Password_Construction_Standard_April2025.pdf`, `sans_Email_Management_Policy_feb2026.pdf`, all from the SANS security policy templates (<https://www.sans.org/information-security-policy/>). |

A question whose documents are not indexed is skipped and listed at the end of the run, so a partial
corpus still gives a meaningful (smaller) result. `369.pdf` carries four questions' worth of weight
for page-level citations, so index all three if you want numbers comparable with the README.

## Running it

The evaluation reads the real index, so upload the documents first (Library page, or
`POST /api/documents`) as an administrator and wait for `indexed`. Names must match the files above.

```bash
# `eval/` is only mounted in the container with the development override
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d
docker compose -f docker-compose.yml -f docker-compose.dev.yml exec backend python -m eval.run_eval --verbose
```

It reads the library and runs searches only; it never writes, drops or rebuilds anything. (Scripts
that do must set a scratch `CHROMA_COLLECTION` first.)

The exit status is 1 when an answerable question is wrongly stopped, an off-topic question gets
through the guard, or the hybrid hit-rate@5 is below `--min-hit-rate` (default 0.8).

## Question kinds

| `kind` | `answerable` | What it checks |
| --- | --- | --- |
| `normal` | yes | The expected document and page are in the top 5, and the guard lets it through. |
| `exact_token` | yes | An error code, id or time (`ERR_404_NET`, `EM-09`, `01:30`, `E2`) is found even when the dense score is modest. |
| `near_miss` | no | The topic is in the documents but the fact is not. The guard is **expected** to let it through: the model's own refusal handles it. Reported, not judged. |
| `off_topic` | no | Nothing in the documents is relevant. The guard must stop it before the LLM. |

`requires` (optional) lists documents a question depends on beyond `expected_document`; a near-miss
about the SANS password standard is meaningless without that PDF.

## Reference result

Gemini `gemini-embedding-001` at 768 dimensions, `RELEVANCE_THRESHOLD=0.60`, lexical override
margin 0.15, full corpus (24 questions):

```
Retrieval over 12 answerable questions (top 5)
dense      hit-rate@5 1.00   MRR 0.96
keyword    hit-rate@5 1.00   MRR 1.00
hybrid     hit-rate@5 1.00   MRR 1.00

Guard: off-topic stopped 5/5, answerable wrongly stopped 0/12, near-miss passed on 6/7

Best dense similarity by kind
  normal       min=0.674 mean=0.761 max=0.814
  exact_token  min=0.638 mean=0.703 max=0.799
  near_miss    min=0.571 mean=0.676 max=0.736
  off_topic    min=0.497 mean=0.519 max=0.544
```

The gap between the weakest answerable question (0.638) and the strongest off-topic one (0.544) is
what the 0.60 threshold sits in. It is about 0.1 wide: if a change narrows it, recalibrate before
shipping. Embedding p50 is about 0.5 s per question; search and fusion are a few milliseconds.

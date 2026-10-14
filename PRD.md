# Product Requirement Document (PRD)

**Product Name:** GroundedRAG — Enterprise Internal Knowledge Engine

**Document Version:** 1.2

**Target Release:** Q4 2026

**Product Status:** Proposed / In Development

---

## 1. Product Overview & Vision

**GroundedRAG** is an internal AI-powered search and document intelligence portal designed for enterprise teams. It transforms unstructured corporate knowledge bases (PDFs, Word documents, policy handbooks, and technical specifications) into an interactive, natural-language Q&A engine.

Instead of opening dozens of files and manually searching with `Ctrl + F`, employees ask natural-language questions and receive **concise, verified answers complete with exact inline source citations** (document name, section, and page numbers).

In one line: a "Google Search + ChatGPT" for a company's internal files, giving **answers with receipts**.

---

## 2. Problem Statement

* **Information Fragmentation:** Critical operational knowledge is buried inside hundred-page PDFs, SharePoint folders, and internal wikis across multiple departments (HR, Engineering, Legal).
* **Time Wasted:** Employees spend an average of 15–20 minutes navigating folders and skim-reading documents to find specific policy rules or technical error codes.
* **LLM Hallucinations:** Out-of-the-box public LLMs cannot answer questions on private company data and risk making up facts if ungrounded.
* **Lack of Trust:** Employees hesitate to rely on standard chatbot outputs unless they can immediately verify the source material.

---

## 3. Target Audience & Personas

| Persona | Role / Department | Primary Need |
| --- | --- | --- |
| **Sarah (Software Engineer)** | Engineering | Quick lookup of internal API specs, system architecture guidelines, and specific error codes. |
| **Mark (People Ops Specialist)** | Human Resources | Rapid response to employee policy inquiries (e.g., parental leave, expense limits) without manually checking 50-page handbooks. |
| **Elena (Compliance Analyst)** | Legal / Governance | Verifying policy compliance across legacy contracts filtered strictly by contract year and region. |

---

## 4. Key Product Features & Capabilities

### 4.1 Knowledge Base Management (Admin & Workspace)

* **Drag-and-Drop Document Upload:** Support for PDFs, `.docx`, `.md`, `.html`, and plain text files.
* **Automated Tagging & Metadata Extraction:** Auto-extracts document type, creation date, and department during indexing (user-supplied values at upload take priority).
* **Document Status Dashboard:** Real-time visibility into indexing progress (queued, processing, indexed, failed), document chunk counts, and total files stored.
* **Document Deletion:** Deleting a document removes the file, its registry entry, and all of its indexed content, so it never appears in answers again.

### 4.2 Interactive Search & Q&A Interface

* **Natural Language Query Bar:** Accepts conversational queries, exact error codes, or technical terms.
* **Conversational Follow-ups:** Follow-up questions (e.g., "what about for contractors?") are understood in the context of the conversation.
* **Pre-Retrieval Metadata Filters:** Side drawer or header dropdowns allowing users to narrow searches by:
  * **Department:** `[All, HR, Engineering, Legal, Operations, IT]`
  * **Document Type:** `[Policy, Manual, Technical Spec, Contract]`
  * **Date Range:** Filter documents created or updated within specific timeframes.
  * **Specific Documents:** Restrict a question to chosen files.
* **Grounded Answer Panel:** Clean, formatted natural language response synthesizing information across relevant files, streamed as it is generated.

### 4.3 Verified Source Citations & Transparency

* **Inline Citation Tags:** Every assertion made in the generated answer includes a clickable tag (e.g., `[IT_Security_Policy.pdf • Page 14]`).
* **Source Preview Drawer:** Clicking a citation opens a side panel displaying the exact text chunk extracted from the original file, and the original document opened at that page, for verification.
* **Out-of-Scope Graceful Handling:** If the retrieved documents do not contain enough information to answer with confidence (top-chunk relevance score below the configured threshold, default **0.60** on a 0–1 cosine-similarity scale, calibrated for `gemini-embedding-001`; see TDD 6.3), the system explicitly states that the information was not found rather than guessing.

### 4.4 Access Control

* **Roles:** `admin` and `user`.
* **Document Access Levels:** `Public`, `Internal`, `Restricted`. Users only retrieve content they are permitted to see; this is enforced on every query.

---

## 5. User Workflow & Experience

```
 [ User Selects Metadata Filters ]
 (e.g., Dept = "Engineering", Type = "Spec")
                │
                ▼
  [ User Types Natural Language Query ]
  ("What is the maximum retry limit for API timeouts?")
                │
                ▼
     [ System Searches Knowledge Base ]
                │
                ▼
  ┌──────────────────────────────────────────────────────────┐
  │                   Web UI Response View                   │
  │                                                          │
  │  1. Grounded Summary Answer                              │
  │     "The maximum API timeout retry limit is 3 attempts   │
  │     with exponential backoff [API_Spec_v2.pdf • Pg 8]."  │
  │                                                          │
  │  2. Interactive Citation Cards                           │
  │     [API_Spec_v2.pdf • Page 8, Section 4.2]              │
  │     └── Preview: "API gateways must enforce a maximum... │
  └──────────────────────────────────────────────────────────┘
```

### Demo Script (for a review panel)

1. **Show the problem:** open a library of 100+ messy PDFs and note that finding one rule means 20 minutes of opening files and using `Ctrl + F`.
2. **Ask a question** in the web UI and apply a quick metadata filter (e.g., `Engineering`).
3. **Show the result and the proof:** point to the instant answer, click a citation, and show the exact source chunk and page so the user knows the AI is not making things up.

---

## 6. Functional & Non-Functional Requirements

### Functional Requirements

1. **Document Ingestion:** The system must process and index uploaded files within 30 seconds for files up to 50 pages. *(Measured: not reachable on a free-tier Gemini key, which allows about 100 embedding requests per minute, so indexing runs at roughly 90 chunks a minute; a billing-enabled key meets it. See TDD 12.)*
2. **Citation Accuracy:** 100% of generated claims must map to at least one valid retrieved source chunk.
3. **Filtering:** Applying a metadata filter must strictly exclude any document outside the selected metadata criteria prior to answer generation.
4. **Access Enforcement:** A user must never receive content from a document above their access level.
5. **Deletion:** A deleted document's content must never appear in subsequent answers.

### Non-Functional Requirements

* **Response Latency:** End-to-end query response time must be under **3.5 seconds** (p95 latency). *(Measured with `gemini-3.8-flash`: 2.9 to 7.4 s; a refusal below the threshold takes about 0.6 s because no model is called. See TDD 12.)*
* **Usability:** Zero training required for end-users; the interface follows standard search engine / chat conventions.
* **Security & Privacy:** Documents are processed in isolated workspaces; raw files and embeddings are restricted based on user access roles.
* **Durability:** Documents, users, and conversations persist across restarts.

---

## 7. Success Metrics & KPIs

| Metric | Target Goal | Measure Method |
| --- | --- | --- |
| **Search Time Reduction** | $>70\%$ decrease in time-to-answer | Compared against manual document search baseline (15 mins $\rightarrow <1$ min). |
| **Answer Accuracy & Grounding** | $>95\%$ factually grounded | Evaluated via automated RAG evaluation frameworks and user feedback loops. |
| **User Trust / CSAT** | $>4.5 / 5.0$ rating | In-app thumbs-up / thumbs-down user ratings on answers and citations. |
| **Hallucination Rate** | $<2\%$ false positives | Strict thresholding on retrieval relevance scores before sending context to LLM. |

---

## 8. Out of Scope (Phase 1)

* Writing or modifying original source documents directly inside the portal (read-only search/Q&A).
* Multi-modal ingestion (processing audio, video, or complex image charts without textual tabular data).
* OCR for scanned PDFs (such files are rejected with a clear message).
* Integration with live external web search (strictly restricted to internal uploaded files).

---

## Changelog

* **1.2:** Brought the numbers in line with what was built and measured: the refusal threshold default is **0.60** (the 0.45 below was calibrated for no model and would never refuse with `gemini-embedding-001`; TDD 6.3), IT is a department, and the ingestion and latency requirements carry their measured status. Behavior added by the audit pass that users can see: signup can be closed, saved citations to deleted or no-longer-readable documents show as unavailable, and uploads and sign-in attempts are rate limited.
* **1.1:** Unified the refusal threshold at 0.45 on a cosine-similarity scale (previously 0.45 here and 0.35 in the TDD on an RRF score). Added access control, document deletion, follow-up questions, and the scanned-PDF out-of-scope note.

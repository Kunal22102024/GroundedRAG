export type Role = "admin" | "user";
export type AccessLevel = "Public" | "Internal" | "Restricted";
export type DocStatus = "queued" | "processing" | "indexed" | "failed";

export interface User {
  id: string;
  email: string;
  role: Role;
}

/** An account on the administrators' Users screen. */
export interface UserAdminItem extends User {
  created_at: string;
  documents: number;
  conversations: number;
}

export interface AuthResponse {
  access_token: string;
  user: User;
}

export type ExecutiveSummaryState = "none" | "pending" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  name: string;
  mime_type: string;
  size_bytes: number;
  status: DocStatus;
  error_message: string | null;
  chunk_count: number;
  department: string;
  document_type: string;
  creation_date: string | null;
  access_level: AccessLevel;
  created_at: string;
  /** The executive summary that is written automatically after indexing. */
  executive_summary: ExecutiveSummaryState;
}

export interface DocumentList {
  items: DocumentItem[];
  total: number;
  summary: { total_files: number; total_chunks: number; by_status: Record<string, number> };
}

export interface UploadResult {
  documents: DocumentItem[];
  rejected: { filename: string; reason: string }[];
}

export interface Citation {
  n: number;
  chunk_id: string;
  document_id: string;
  document_name: string;
  page_number: number;
  page_end: number;
  section_title: string;
  text: string;
  /** Set when the cited document was deleted or the reader may no longer see it. */
  unavailable?: boolean;
}

/** One side of a disagreement: what a document says, and the citation `n` it comes from. */
export interface ConflictClaim {
  n: number;
  document_id: string;
  document_name: string;
  claim: string;
}

/** A point on which two or more documents contradict each other. */
export interface Conflict {
  topic: string;
  claims: ConflictClaim[];
}

export type SuggestionType = "follow_up" | "sub_question" | "related";

/** A question offered after an answer, answerable from a passage of `document_id`. */
export interface FollowUp {
  type: SuggestionType;
  question: string;
  document_id: string | null;
}

/** How a chat question was asked (recorded for the dashboard). */
export type QuestionSource = "typed" | "suggestion" | "followup";

export interface Suggestions {
  documents: { question: string; document_id: string; document_name: string }[];
  popular: { question: string; asked_by: number }[];
  /** Documents whose suggested questions are still being written. */
  pending: string[];
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  conflicts: Conflict[] | null;
  suggestions: FollowUp[] | null;
  guard_score: number | null;
  flagged: boolean;
  /** Served from the saved-answer cache: no search and no new AI call. */
  cached?: boolean;
  created_at: string;
  feedback: "up" | "down" | null;
}

export type GraphStatus = "none" | "pending" | "ready" | "failed";

/** How one document's knowledge graph is doing. */
export interface GraphDocument {
  id: string;
  name: string;
  status: GraphStatus;
  entities: number;
  relations: number;
  /** Share of the document's chunks the model read (1 = all of them). */
  coverage: number | null;
  error: string | null;
}

/** An entity, merged by name across the documents shown. `id` is its normalised name. */
export interface GraphNode {
  id: string;
  name: string;
  type: string;
  description: string;
  document_ids: string[];
  mentions: number;
  degree: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  label: string;
  count: number;
}

export interface GraphView {
  documents: GraphDocument[];
  nodes: GraphNode[];
  edges: GraphEdge[];
  total_nodes: number;
  truncated: boolean;
}

export interface EntityRelation {
  label: string;
  /** `out`: this entity is the subject; `in`: it is the object. */
  direction: "out" | "in";
  quote: string;
  other: { id: string; name: string; type: string };
  evidence: Citation;
}

export interface EntityDetail {
  id: string;
  name: string;
  type: string;
  description: string;
  documents: { id: string; name: string }[];
  mentions: Citation[];
  relations: EntityRelation[];
}

export interface Conversation {
  id: string;
  title: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  messages: Message[];
}

export interface ChatFilters {
  department?: string;
  document_type?: string;
  date_from?: string;
  date_to?: string;
  document_ids?: string[];
}

export interface DonePayload {
  conversation_id: string;
  message_id: string;
  answer: string;
  refused: boolean;
  guard: { passed: boolean; reason: string; score: number | null; threshold: number };
  timings: Record<string, number | null>;
  usage: Record<string, number>;
  flags: { partial: boolean; citations_removed: number; uncited: boolean; retried: boolean; rewritten: boolean };
  /** True when the answer came from the saved-answer cache. */
  cached?: boolean;
}

export interface ChatHandlers {
  onStatus: (stage: string, data: Record<string, unknown>) => void;
  onToken: (text: string) => void;
  onCitations: (citations: Citation[]) => void;
  onConflicts: (conflicts: Conflict[]) => void;
  onSuggestions: (suggestions: FollowUp[]) => void;
  onDone: (done: DonePayload) => void;
  onError: (message: string, code: string) => void;
}

// --- Analysis (Compare, Extract, Summaries) ---------------------------------------------------------

export type SummaryKind = "executive" | "chapters" | "faq" | "glossary" | "key_changes";

/** Item lists keep their `[n]` markers in the text; `citations` (document-wide) resolves them. */
export type SummaryContent =
  | { kind: "executive"; text: string; citations: Citation[] }
  | { kind: "chapters"; items: { title: string; summary: string }[]; citations: Citation[] }
  | { kind: "faq"; items: { question: string; answer: string }[]; citations: Citation[] }
  | { kind: "glossary"; items: { term: string; definition: string }[]; citations: Citation[] }
  | { kind: "key_changes"; items: { change: string; impact: string }[]; citations: Citation[] };

export interface ExtractedValue {
  value: string;
  /** Number of the citation this value comes from. */
  n: number;
  quote: string;
}

export interface ExtractedField {
  field: string;
  values: ExtractedValue[];
}

export interface ExtractionContent {
  kind: "extraction";
  fields: ExtractedField[];
  citations: Citation[];
}

/** A saved analysis as `GET /documents/{id}/insights` lists it. */
export interface Insight {
  id: string;
  kind: SummaryKind | "extraction" | "questions";
  params_key: string;
  status: "pending" | "ready" | "failed";
  content: (SummaryContent | ExtractionContent) | null;
  error_message: string | null;
  updated_at: string;
}

/** What the `done` event of a summary or extraction carries. */
export interface AnalysisDone<T> {
  kind: string;
  content: T;
  /** True when a saved result was returned instead of asking the model again. */
  cached: boolean;
}

export interface CompareColumn {
  document_id: string;
  document_name: string;
  answer: string;
  /** False when the document does not contain the answer (no model call was made, or it declined). */
  covered: boolean;
  citations: Citation[];
  flags: { partial?: boolean; citations_removed?: number; uncited?: boolean };
}

// --- Evaluation dashboard (admins) ----------------------------------------------------------------

export interface DailyMetrics {
  day: string;
  queries: number;
  refused: number;
  failed: number;
  abstention_rate: number | null;
  failure_rate: number | null;
  p95_ms: number | null;
  input_tokens: number;
  output_tokens: number;
}

export interface MetricsSummary {
  days: number;
  kind: "all" | "chat" | "analysis";
  threshold: number;
  totals: { queries: number; answered: number; refused: number; failed: number; abstention_rate: number | null; failure_rate: number | null };
  relevance: {
    best_similarity_answered_mean: number | null;
    best_similarity_answered_p50: number | null;
    best_similarity_refused_mean: number | null;
    mean_similarity_answered: number | null;
    guard_reasons: Record<string, number>;
    lexical_override_rate: number | null;
    histogram: { from: number; count: number }[];
  };
  abstention: { guard_refusals: number; model_refusals: number; rate: number | null };
  groundedness: {
    answers: number;
    with_citations_rate: number | null;
    citation_precision: number | null;
    invalid_citations_removed: number;
    uncited_rate: number | null;
    retried_rate: number | null;
    truncated_rate: number | null;
    avg_citations: number | null;
    avg_documents: number | null;
    conflict_rate: number | null;
    feedback_up: number;
    feedback_down: number;
  };
  latency: { p50_ms: number | null; p95_ms: number | null; max_ms: number | null; stages: { stage: string; p50: number | null; p95: number | null }[] };
  cost: {
    input_tokens: number;
    output_tokens: number;
    usd: number | null;
    priced: boolean;
    by_kind: { kind: string; events: number; input_tokens: number; output_tokens: number; usd: number | null }[];
  };
  suggestions: {
    chat_questions: number;
    from_suggestion: number;
    from_followup: number;
    answers_with_followups: number;
    usage_rate: number | null;
    followup_click_rate: number | null;
  };
  cache: {
    enabled: boolean;
    hits: number;
    hit_rate: number | null;
    tokens_saved: number;
    hit_avg_ms: number | null;
    miss_avg_ms: number | null;
  };
  daily: DailyMetrics[];
}

export interface QueryEventItem {
  id: string;
  created_at: string;
  kind: string;
  outcome: "answered" | "refused" | "failed";
  source: QuestionSource;
  user_email: string | null;
  question: string | null;
  error_code: string | null;
  guard_reason: string | null;
  best_similarity: number | null;
  cited: number;
  citations_removed: number;
  uncited: boolean;
  retried: boolean;
  partial: boolean;
  total_ms: number | null;
  total_tokens: number;
  conversation_id: string | null;
  document_ids: string[] | null;
}

export const REFUSAL_MESSAGE = "Insufficient information available in the selected documents.";

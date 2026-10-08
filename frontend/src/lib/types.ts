export type Role = "admin" | "user";
export type AccessLevel = "Public" | "Internal" | "Restricted";
export type DocStatus = "queued" | "processing" | "indexed" | "failed";

export interface User {
  id: string;
  email: string;
  role: Role;
}

export interface AuthResponse {
  access_token: string;
  user: User;
}

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
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[] | null;
  guard_score: number | null;
  flagged: boolean;
  created_at: string;
  feedback: "up" | "down" | null;
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
}

export interface ChatHandlers {
  onStatus: (stage: string, data: Record<string, unknown>) => void;
  onToken: (text: string) => void;
  onCitations: (citations: Citation[]) => void;
  onDone: (done: DonePayload) => void;
  onError: (message: string, code: string) => void;
}

export const REFUSAL_MESSAGE = "Insufficient information available in the selected documents.";

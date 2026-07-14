/**
 * Mirrors the FastAPI backend's Pydantic schemas (backend/schemas/*.py) —
 * kept as plain interfaces (not generated) since there's no OpenAPI codegen
 * step in this project; if the backend schema changes, update here too.
 */

// --- Auth (backend/schemas/auth.py, backend/models/user.py) -----------------

export type UserRole = "admin" | "analyst" | "customer_support";

export interface UserPublic {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  created_at: string;
}

export interface TokenPair {
  access_token: string;
  refresh_token: string | null;
  token_type: string;
  expires_in: number;
  scope: string;
}

// --- Chat (backend/schemas/chat.py) -----------------------------------------

export type MessageRole = "user" | "assistant";
export type ConfidenceLevel = "high" | "medium" | "low";

export interface Citation {
  index: number;
  document_id: string;
  chunk_id: string;
  document_filename: string;
  excerpt: string;
  score: number;
  header_path: string;
}

export interface DocumentUsed {
  document_id: string;
  filename: string;
  chunk_count: number;
}

export interface ChatAnswer {
  conversation_id: string;
  answer: string;
  citations: Citation[];
  documents_used: DocumentUsed[];
  confidence_score: number;
  confidence_level: ConfidenceLevel;
  warnings: string[];
  model: string;
  input_tokens: number;
  output_tokens: number;
}

export interface MessagePublic {
  id: string;
  role: MessageRole;
  content: string;
  created_at: string;
  // Populated client-side once a streamed/non-streamed answer resolves —
  // the backend only returns citations on the ChatAnswer response, not on
  // GET /conversations/{id}'s persisted message history (see ChatPage).
  citations?: Citation[];
  documentsUsed?: DocumentUsed[];
  confidenceScore?: number;
  confidenceLevel?: ConfidenceLevel;
  warnings?: string[];
}

export interface ConversationPublic {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends ConversationPublic {
  messages: MessagePublic[];
}

/** One "done" event payload from the /chat/stream SSE endpoint. */
export interface ChatStreamDonePayload {
  citations: Citation[];
  documents_used: DocumentUsed[];
  confidence_score: number;
  confidence_level: ConfidenceLevel;
  warnings: string[];
  model: string;
  input_tokens: number;
  output_tokens: number;
}

// --- Documents (backend/schemas/document.py) --------------------------------

export type DocumentStatus = "pending" | "processing" | "indexed" | "failed";

export interface DocumentMetadataPublic {
  title: string | null;
  author: string | null;
  page_count: number | null;
  sheet_names: string[] | null;
  word_count: number | null;
  char_count: number | null;
  extra: Record<string, unknown>;
}

export interface DocumentPublic {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocumentStatus;
  error_message: string | null;
  document_key: string;
  content_hash: string;
  version: number;
  is_current: boolean;
  previous_version_id: string | null;
  near_duplicate_of_id: string | null;
  language: string | null;
  metadata: DocumentMetadataPublic | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentList {
  items: DocumentPublic[];
  total: number;
}

// --- Users / admin (backend/schemas/user.py) --------------------------------

export interface UserAdminView {
  id: string;
  email: string;
  full_name: string;
  role: UserRole;
  is_active: boolean;
  failed_login_attempts: number;
  locked_until: string | null;
  created_at: string;
  updated_at: string;
}

export interface UserList {
  items: UserAdminView[];
  total: number;
}

// --- Audit (backend/schemas/audit.py) ---------------------------------------

export interface AuditLogEntry {
  id: string;
  user_id: string | null;
  action: string;
  resource: string | null;
  detail: string | null;
  ip_address: string | null;
  created_at: string;
}

export interface AuditLogList {
  items: AuditLogEntry[];
  total: number;
}

export interface AuditChainVerification {
  intact: boolean;
  checked_entries: number;
  problems: string[];
}

// --- Evaluation (backend/schemas/evaluation.py) -----------------------------

export type EvalMetricName =
  | "faithfulness"
  | "context_precision"
  | "context_recall"
  | "answer_relevancy"
  | "hallucination_rate";

export interface MetricScorePublic {
  metric: EvalMetricName | string;
  score: number;
  provider: string;
}

export interface SampleEvaluationPublic {
  sample_id: string;
  question: string;
  answer: string;
  ground_truth: string;
  scores: MetricScorePublic[];
}

export interface EvalReportPublic {
  run_id: string;
  created_at: string;
  dataset_name: string;
  num_samples: number;
  aggregate_scores: Record<string, number>;
  providers_used: Record<string, string>;
  warnings: string[];
  sample_evaluations: SampleEvaluationPublic[];
}

export interface EvalRunSummary {
  id: string;
  dataset_name: string;
  num_samples: number;
  aggregate_scores: Record<string, number>;
  warnings: string[];
  created_at: string;
}

// --- Generic API error envelope (backend/core/exceptions.py) ---------------

export interface ApiErrorPayload {
  error: {
    code: string;
    message: string;
    request_id: string | null;
    details?: Record<string, unknown>;
  };
}

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';

export interface ClarificationRequest {
  session_id?: string;
  requirement_doc: string;
  target_repos?: string[];
}

export interface SSEEvent {
  type: string;
  agent?: string;
  timestamp?: string;
  content?: unknown;
}

export interface ClarificationResponse {
  session_id: string;
  status: string;
  questions: ClarificationQuestion[];
  tech_questions: Record<string, unknown>[];
  rounds: number;
}

export interface ClarificationQuestion {
  id: string;
  text: string;
  category: string;
  severity: string;
  source_code_refs?: SourceCodeRef[];
  requirement_ref?: string;
  suggested_options?: string[];
}

export interface SourceCodeRef {
  file: string;
  line?: number;
  snippet?: string;
}

export interface ClarificationRound {
  round: number;
  questions: ClarificationQuestion[];
  filtered_questions: ClarificationQuestion[];
  passed: boolean;
}

export interface AnalysisResult {
  coverage_score: number;
  covered_points: string[];
  uncovered_gaps: string[];
  needs_more_exploration: boolean;
  exploration_feedback: string;
}

export interface SessionState {
  sessionId: string;
  status: 'idle' | 'started' | 'running' | 'completed' | 'error';
  questions: ClarificationQuestion[];
  techQuestions: Record<string, unknown>[];
  events: SSEEvent[];
  analysis?: AnalysisResult;
  clarificationRounds: ClarificationRound[];
  error?: string;
}

export function createApiClient() {
  async function startSession(req: ClarificationRequest): Promise<{ session_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/clarification/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  function connectEvents(sessionId: string, onEvent: (event: SSEEvent) => void): EventSource {
    return new EventSource(`${API_BASE}/clarification/${sessionId}/events`);
  }

  async function getStatus(sessionId: string): Promise<{ session_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/clarification/${sessionId}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  return { startSession, connectEvents, getStatus };
}

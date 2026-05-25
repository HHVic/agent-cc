const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000/api/v1';

export interface ClarificationRequest {
  session_id?: string;
  requirement_doc: string;
  target_repos?: string[];
}

export interface SSEEvent {
  type: string;
  data?: unknown;
  content?: unknown;
  round?: number;
  timestamp?: number;
}

export interface ClarificationResponse {
  session_id: string;
  status: string;
  questions: Record<string, unknown>[];
  tech_questions: Record<string, unknown>[];
  rounds: number;
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

  function connectEvents(sessionId: string): EventSource {
    return new EventSource(`${API_BASE}/clarification/${sessionId}/events`);
  }

  async function getStatus(sessionId: string): Promise<{ session_id: string; status: string }> {
    const res = await fetch(`${API_BASE}/clarification/${sessionId}/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  }

  return { startSession, connectEvents, getStatus };
}

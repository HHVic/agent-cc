import { useState } from 'react';
import { createApiClient } from '../../api/client';
import { StreamDisplay } from '../../components/StreamDisplay';
import { StatusBadge } from '../../components/StatusBadge';

export function RequirementClarification() {
  const [doc, setDoc] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [questions, setQuestions] = useState<Record<string, unknown>[]>([]);

  async function handleStart() {
    const { startSession } = createApiClient();
    const { session_id } = await startSession({ requirement_doc: doc });
    setSessionId(session_id);
    setStatus('running');

    const evtSource = createApiClient().connectEvents(session_id);
    evtSource.onmessage = (event) => {
      if (event.data.startsWith(':')) return;
      try {
        const parsed = JSON.parse(event.data);
        if (parsed.type === 'done' || parsed.type === 'critic_pass') {
          setStatus('completed');
          setQuestions(parsed.content?.filtered_questions || []);
        }
        if (parsed.type === 'error') setStatus('error');
      } catch { /* skip */ }
    };
  }

  return (
    <div style={{ maxWidth: '900px', margin: '0 auto', padding: '24px' }}>
      <h1>Requirement Clarification</h1>

      <div style={{ marginBottom: '16px' }}>
        <textarea
          value={doc}
          onChange={e => setDoc(e.target.value)}
          placeholder="Paste requirement document here..."
          style={{ width: '100%', height: '200px', padding: '8px', fontFamily: 'monospace' }}
        />
        <button onClick={handleStart} disabled={!doc || !!sessionId}>
          {sessionId ? 'Processing...' : 'Start Clarification'}
        </button>
      </div>

      {sessionId && (
        <div style={{ marginBottom: '16px' }}>
          <StatusBadge status={status} />
          <StreamDisplay sessionId={sessionId} />
        </div>
      )}

      {questions.length > 0 && (
        <div>
          <h2>Clarification Questions</h2>
          {questions.map((q, i) => (
            <div key={i} style={{ padding: '12px', marginBottom: '8px', border: '1px solid #ddd', borderRadius: '8px' }}>
              <strong>[{(q as Record<string, unknown>).severity as string}]</strong> {(q as Record<string, unknown>).text as string}
              <div style={{ fontSize: '12px', color: '#888', marginTop: '4px' }}>
                Category: {(q as Record<string, unknown>).category as string} | Source: {Array.isArray((q as Record<string, unknown>).source_code_refs) ? (q as Record<string, unknown>).source_code_refs.map((r: Record<string, unknown>) => (r.file as string)).join(', ') : ''}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  createApiClient,
  type SSEEvent,
  type ClarificationQuestion,
} from '../../api/client';

// ─── Severity / Category styling ─────────────────────────────────────────────

const severityStyle: Record<string, { bg: string; color: string; label: string }> = {
  critical:  { bg: '#f8514920', color: '#f85149', label: '🔴 Critical' },
  important: { bg: '#d2992220', color: '#d29922', label: '🟡 Important' },
  normal:    { bg: '#58a6ff20', color: '#58a6ff', label: '🔵 Normal' },
  low:       { bg: '#3fb95020', color: '#3fb950', label: '🟢 Low' },
};

const categoryLabels: Record<string, string> = {
  business_flow:       'Business Flow',
  data_model:          'Data Model',
  edge_case:           'Edge Case',
  exception_handling:  'Exception Handling',
};

const agentColors: Record<string, string> = {
  Planner:       '#a371f7',
  Explorer:      '#58a6ff',
  Analyzer:      '#d29922',
  QuestionGen:   '#3fb950',
  Critic:        '#f85149',
};

// ─── Components ──────────────────────────────────────────────────────────────

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { bg: string; color: string; text: string; animate?: boolean }> = {
    started:   { bg: '#3b82f620', color: '#3b82f6', text: 'Started' },
    running:   { bg: '#d2992220', color: '#d29922', text: 'Running', animate: true },
    completed: { bg: '#3fb95020', color: '#3fb950', text: 'Completed' },
    error:     { bg: '#f8514920', color: '#f85149', text: 'Error' },
  };
  const style = map[status] || { bg: '#30363d30', color: '#8b949e', text: status };

  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '4px 12px', borderRadius: 12, fontSize: 12, fontWeight: 500,
      background: style.bg, color: style.color,
    }}>
      <span style={{
        width: 8, height: 8, borderRadius: '50%', background: style.color, display: 'inline-block',
        animation: style.animate ? 'pulse 1.5s infinite' : 'none',
      }} />
      {style.text}
    </span>
  );
}

function AgentIcon({ agent }: { agent?: string }) {
  if (!agent) return <span style={{ color: '#484f58' }}>●</span>;
  const color = agentColors[agent] || '#8b949e';
  return (
    <span style={{
      display: 'inline-block', width: 8, height: 8, borderRadius: '50%',
      background: color, flexShrink: 0, marginTop: 4,
    }} title={agent} />
  );
}

function StreamDisplay({ sessionId, onEvent }: { sessionId: string; onEvent?: (e: SSEEvent) => void }) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const evtSource = new EventSource(`/api/v1/clarification/${sessionId}/events`);

    evtSource.onmessage = (event) => {
      if (event.data.startsWith(':')) return;
      try {
        const parsed: SSEEvent = JSON.parse(event.data);
        setEvents(prev => [...prev, parsed]);
        onEvent?.(parsed);
      } catch { /* skip */ }
    };

    evtSource.onerror = () => evtSource.close();

    return () => evtSource.close();
  }, [sessionId, onEvent]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  return (
    <div style={{
      maxHeight: 360, overflowY: 'auto', padding: 12,
      background: '#010409', borderRadius: 8,
      border: '1px solid #30363d',
      fontFamily: 'ui-monospace, SFMono-Regular, SF Mono, monospace',
      fontSize: 12, lineHeight: 1.6,
    }}>
      {events.length === 0 && (
        <div style={{ color: '#484f58', padding: '20px 0', textAlign: 'center' }}>
          Waiting for events...
        </div>
      )}
      {events.map((event, i) => (
        <div key={i} style={{
          padding: '4px 0', borderBottom: '1px solid #21262d',
          display: 'flex', gap: 8, alignItems: 'flex-start',
        }}>
          <AgentIcon agent={event.agent} />
          <div style={{ flex: 1, minWidth: 0 }}>
            <span style={{ color: '#484f58' }}>{event.timestamp}</span>
            {' '}
            <span style={{ color: '#8b949e' }}>[{event.agent || 'system'}</span>
            <span style={{ color: '#484f58' }}>]</span>
            {' '}
            <span style={{ color: '#e6edf3', wordBreak: 'break-word' }}>
              {typeof event.content === 'string' ? event.content : JSON.stringify(event.content, null, 2)}
            </span>
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

function QuestionCard({ question, index }: { question: ClarificationQuestion; index: number }) {
  const sev = severityStyle[question.severity] || severityStyle.normal;
  const cat = categoryLabels[question.category] || question.category;

  return (
    <div style={{
      background: '#0d1117', border: '1px solid #30363d',
      borderRadius: 8, padding: 16, marginBottom: 12,
      transition: 'border-color 0.15s',
    }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = sev.color)}
      onMouseLeave={e => (e.currentTarget.style.borderColor = '#30363d')}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{
          display: 'inline-block', padding: '2px 8px', borderRadius: 4,
          fontSize: 11, fontWeight: 600, background: sev.bg, color: sev.color,
        }}>
          {sev.label}
        </span>
        <span style={{ color: '#484f58', fontSize: 12 }}>#{index + 1}</span>
      </div>

      <div style={{ fontSize: 14, lineHeight: 1.6, marginBottom: 12, color: '#e6edf3' }}>
        {question.text}
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, fontSize: 12 }}>
        <span style={{ color: '#8b949e' }}>
          📁 {cat}
        </span>
        {question.requirement_ref && (
          <span style={{ color: '#8b949e' }}>
            📄 {question.requirement_ref}
          </span>
        )}
        {question.source_code_refs && question.source_code_refs.length > 0 && (
          <span style={{ color: '#8b949e' }}>
            🔗 {question.source_code_refs.map(r => r.file).join(', ')}
          </span>
        )}
      </div>

      {question.suggested_options && question.suggested_options.length > 0 && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {question.suggested_options.map((opt, i) => (
            <span key={i} style={{
              background: '#161b22', border: '1px solid #30363d',
              borderRadius: 4, padding: '2px 8px', fontSize: 12,
              color: '#58a6ff',
            }}>
              {opt}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

const SAMPLE_DOCS = [
  {
    label: 'User Registration',
    doc: `User Registration Feature Requirements:

1. Users should be able to register with email and password
2. Password must meet complexity requirements (8+ chars, uppercase, lowercase, number, special char)
3. Email verification required before account activation
4. Users can reset password via email link (link expires in 24h)
5. Rate limiting: max 5 registration attempts per IP per hour
6. Support OAuth2 login with Google and GitHub
7. User profile includes: username, email, avatar, bio, created_at
8. Duplicate email check on registration
9. Account deletion should cascade to all associated data
10. GDPR compliance: users can export or delete their data`,
  },
  {
    label: 'E-commerce Order',
    doc: `E-commerce Order Management Requirements:

1. Users can browse products with search and category filtering
2. Shopping cart: add/remove items, update quantities
3. Checkout flow: shipping address → payment → order confirmation
4. Payment supports: credit card, PayPal, Alipay
5. Order status workflow: pending → confirmed → shipped → delivered
6. Inventory check before order confirmation
7. Order cancellation within 30 min of placement
8. Order history with filtering by status and date
9. Coupon/discount code support
10. Order notification via email and in-app notification
11. Refund processing within 7 business days`,
  },
  {
    label: 'Blank',
    doc: '',
  },
];

export function RequirementClarification() {
  const [doc, setDoc] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('');
  const [questions, setQuestions] = useState<ClarificationQuestion[]>([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleStart = useCallback(async () => {
    if (!doc.trim()) return;

    setLoading(true);
    setErrorMsg(null);
    setQuestions([]);
    setStatus('running');

    try {
      const { startSession } = createApiClient();
      const { session_id } = await startSession({ requirement_doc: doc });
      setSessionId(session_id);

      // Handle SSE events inline
      const evtSource = new EventSource(`/api/v1/clarification/${session_id}/events`);
      evtSource.onmessage = (event) => {
        if (event.data.startsWith(':')) return;
        try {
          const parsed: SSEEvent = JSON.parse(event.data);
          if (parsed.type === 'error') {
            setStatus('error');
            setErrorMsg(typeof parsed.content === 'string' ? parsed.content : 'Unknown error');
          }
          if (parsed.type === 'critic_pass' || parsed.type === 'done') {
            setStatus('completed');
          }
          if (parsed.type === 'questions_generated' && Array.isArray(parsed.content)) {
            setQuestions(parsed.content);
          }
        } catch { /* skip */ }
      };
      evtSource.onerror = () => evtSource.close();
    } catch (err) {
      setStatus('error');
      setErrorMsg(err instanceof Error ? err.message : 'Failed to start session');
    } finally {
      setLoading(false);
    }
  }, [doc]);

  const handleReset = () => {
    setDoc('');
    setSessionId(null);
    setStatus('');
    setQuestions([]);
    setErrorMsg(null);
  };

  const handleLoadSample = (sample: typeof SAMPLE_DOCS[number]) => {
    setDoc(sample.doc);
    if (sample.label === 'Blank') handleReset();
  };

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '32px 24px' }}>
      {/* Header */}
      <div style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 24, fontWeight: 600, marginBottom: 4 }}>
          📋 Requirement Clarification Agent
        </h1>
        <p style={{ color: '#8b949e', fontSize: 14 }}>
          Paste your requirement document below. The agent will explore your codebase, identify gaps, and generate clarification questions.
        </p>
      </div>

      {/* Input Section */}
      {!sessionId && (
        <div style={{ marginBottom: 20 }}>
          {/* Sample doc selector */}
          <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
            {SAMPLE_DOCS.map(s => (
              <button
                key={s.label}
                onClick={() => handleLoadSample(s)}
                style={{
                  background: doc.includes(s.label) ? '#1f6feb' : '#21262d',
                  color: doc.includes(s.label) ? '#fff' : '#8b949e',
                  padding: '4px 12px', borderRadius: 6, fontSize: 12,
                }}
              >
                {s.label}
              </button>
            ))}
          </div>

          <textarea
            value={doc}
            onChange={e => setDoc(e.target.value)}
            placeholder="Paste your requirement document here..."
            rows={10}
            style={{
              width: '100%', padding: 12, borderRadius: 8,
              background: '#0d1117', color: '#e6edf3',
              border: '1px solid #30363d',
              resize: 'vertical',
            }}
          />

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 12 }}>
            <span style={{ color: '#484f58', fontSize: 12 }}>
              {doc.length} characters
            </span>
            <button
              onClick={handleStart}
              disabled={!doc.trim() || loading}
              style={{
                background: !doc.trim() || loading ? '#30363d' : '#238636',
                color: !doc.trim() || loading ? '#484f58' : '#fff',
                padding: '8px 24px', borderRadius: 8, fontSize: 14, fontWeight: 600,
              }}
            >
              {loading ? (
                <><span className="spinner" style={{ marginRight: 8 }} />Processing...</>
              ) : (
                'Start Clarification'
              )}
            </button>
          </div>
        </div>
      )}

      {/* Active Session */}
      {sessionId && (
        <div>
          {/* Status bar */}
          <div style={{
            display: 'flex', justifyContent: 'space-between', alignItems: 'center',
            padding: '12px 16px', background: '#161b22',
            borderRadius: 8, marginBottom: 16,
            border: '1px solid #30363d',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <StatusBadge status={status} />
              <span style={{ color: '#484f58', fontSize: 12, fontFamily: 'monospace' }}>
                {sessionId.slice(0, 8)}...
              </span>
            </div>
            <button
              onClick={handleReset}
              style={{
                background: '#21262d', color: '#8b949e',
                padding: '4px 12px', borderRadius: 6, fontSize: 12,
              }}
            >
              New Session
            </button>
          </div>

          {/* Error message */}
          {errorMsg && (
            <div style={{
              padding: 12, background: '#f8514915', border: '1px solid #f8514940',
              borderRadius: 8, marginBottom: 16, color: '#f85149', fontSize: 13,
            }}>
              ⚠️ {errorMsg}
            </div>
          )}

          {/* Stream */}
          <StreamDisplay sessionId={sessionId} />

          {/* Results */}
          {questions.length > 0 && status === 'completed' && (
            <div style={{ marginTop: 24 }}>
              <h2 style={{ fontSize: 16, fontWeight: 600, marginBottom: 16, display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ color: '#3fb950' }}>✓</span>
                Clarification Questions ({questions.length})
              </h2>
              {questions.map((q, i) => (
                <QuestionCard key={q.id || i} question={q} index={i} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

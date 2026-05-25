import { useEffect, useState, useRef } from 'react';
import type { SSEEvent } from '../api/client';

interface StreamDisplayProps {
  sessionId: string;
  onEvent?: (event: SSEEvent) => void;
}

export function StreamDisplay({ sessionId, onEvent }: StreamDisplayProps) {
  const [events, setEvents] = useState<SSEEvent[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const evtSource = new EventSource(`/api/v1/clarification/${sessionId}/events`);

    evtSource.onmessage = (event) => {
      if (event.data.startsWith(':')) return; // ping
      try {
        const parsed: SSEEvent = JSON.parse(event.data);
        setEvents(prev => [...prev, parsed]);
        onEvent?.(parsed);
      } catch { /* skip non-JSON messages */ }
    };

    evtSource.onerror = () => evtSource.close();

    return () => evtSource.close();
  }, [sessionId, onEvent]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  return (
    <div style={{ maxHeight: '400px', overflowY: 'auto', padding: '8px', background: '#1e1e1e', borderRadius: '8px', color: '#ccc', fontFamily: 'monospace', fontSize: '12px' }}>
      {events.map((event, i) => (
        <div key={i} style={{ padding: '2px 0', borderBottom: '1px solid #333' }}>
          <span style={{ color: '#888' }}>[{event.type}]</span>
          {' '}{JSON.stringify(event.content ?? event.data)}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}

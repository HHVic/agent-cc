interface StatusBadgeProps {
  status: string;
}

const colors: Record<string, string> = {
  running: '#f59e0b',
  completed: '#22c55e',
  error: '#ef4444',
  started: '#3b82f6',
};

export function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: '6px',
      padding: '4px 12px', borderRadius: '12px', fontSize: '12px',
      background: (colors[status] || '#888') + '20', color: colors[status] || '#888',
    }}>
      <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: colors[status] || '#888', display: 'inline-block', animation: status === 'running' ? 'pulse 1.5s infinite' : 'none' }} />
      {status}
    </span>
  );
}

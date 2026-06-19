import React from 'react';

interface ProcessStatusBadgeProps {
  status: string;
  agentStatus?: string;
}

const ProcessStatusBadge: React.FC<ProcessStatusBadgeProps> = ({
  status,
  agentStatus,
}) => {
  const getDisplay = (): { label: string; color: string; pulse?: boolean } => {
    if (status === 'uploading' || status === 'converting') {
      return { label: 'アップロード中', color: 'bg-info', pulse: true };
    }
    if (status === 'pending') {
      return { label: '処理待ち', color: 'bg-neutral-400' };
    }
    if (status === 'processing') {
      return { label: '処理中', color: 'bg-info', pulse: true };
    }
    if (status === 'failed') {
      return { label: '失敗', color: 'bg-danger' };
    }

    if (agentStatus === 'processing') {
      return { label: '検証中', color: 'bg-info', pulse: true };
    }
    if (agentStatus === 'failed') {
      return { label: '検証失敗', color: 'bg-danger' };
    }

    return { label: '完了', color: 'bg-success' };
  };

  const { label, color, pulse } = getDisplay();

  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap text-xs text-neutral-700">
      <span className={`w-2 h-2 rounded-full ${color} ${pulse ? 'animate-pulse' : ''}`} />
      {label}
    </span>
  );
};

export default ProcessStatusBadge;

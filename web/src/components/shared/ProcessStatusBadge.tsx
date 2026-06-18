import React from 'react';
import { Loader2 } from 'lucide-react';

interface ProcessStatusBadgeProps {
  status: string;
  agentStatus?: string;
}

const ProcessStatusBadge: React.FC<ProcessStatusBadgeProps> = ({
  status,
  agentStatus,
}) => {
  const getDisplay = (): { label: string; style: string; spinning?: boolean } => {
    // OCR未完了
    if (status === 'uploading' || status === 'converting') {
      return { label: 'アップロード中', style: 'bg-info-light text-info-text', spinning: true };
    }
    if (status === 'pending') {
      return { label: '処理待ち', style: 'bg-neutral-100 text-neutral-600' };
    }
    if (status === 'processing') {
      return { label: '処理中', style: 'bg-info-light text-info-text', spinning: true };
    }
    if (status === 'failed') {
      return { label: '失敗', style: 'bg-danger-light text-danger-text' };
    }

    // status=completed: agentStatus で分岐
    if (agentStatus === 'processing') {
      return { label: '検証中', style: 'bg-info-light text-info-text', spinning: true };
    }
    if (agentStatus === 'failed') {
      return { label: '検証失敗', style: 'bg-danger-light text-danger-text' };
    }

    // completed + (agent=completed or skipped or null)
    return { label: '完了', style: 'bg-success-light text-success-text' };
  };

  const { label, style, spinning } = getDisplay();

  return (
    <span className={`px-2 inline-flex items-center text-xs leading-5 font-semibold rounded-full ${style}`}>
      {label}
      {spinning && <Loader2 size={14} className="animate-spin ml-1" />}
    </span>
  );
};

export default ProcessStatusBadge;

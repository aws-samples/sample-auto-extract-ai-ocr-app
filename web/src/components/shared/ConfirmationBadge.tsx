import React from 'react';
import { Loader2 } from 'lucide-react';

interface ConfirmationBadgeProps {
  status: string;
  agentStatus?: string;
  agentSuggestionsCount?: number;
  verificationCompleted?: boolean;
}

const ConfirmationBadge: React.FC<ConfirmationBadgeProps> = ({
  status,
  agentStatus,
  agentSuggestionsCount,
  verificationCompleted,
}) => {
  // OCR未完了 → 表示なし
  if (status !== 'completed' && status !== 'failed') {
    return <span className="text-neutral-300">-</span>;
  }

  // 人間確認済み → 最優先
  if (verificationCompleted) {
    return (
      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-success-light text-success-text">
        確認済み
      </span>
    );
  }

  // AI検証失敗
  if (agentStatus === 'failed') {
    return (
      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-danger-light text-danger-text">
        検証失敗
      </span>
    );
  }

  // AI検証中
  if (agentStatus === 'processing') {
    return (
      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-info-light text-info-text">
        AI検証中
        <Loader2 size={14} className="animate-spin ml-1" />
      </span>
    );
  }

  // AI指摘あり
  if (agentStatus === 'completed' && (agentSuggestionsCount ?? 0) > 0) {
    return (
      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-warning-light text-warning-text">
        要確認({agentSuggestionsCount}件)
      </span>
    );
  }

  // それ以外（AI問題なし / スキップ / null）→ 確認待ち
  return (
    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-info-light text-info-text">
      確認待ち
    </span>
  );
};

export default ConfirmationBadge;

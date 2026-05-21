import React, { useState } from 'react';
import { Suggestion } from '../../types/agent';

interface AgentValidationPanelProps {
  suggestions: Suggestion[];
  onAccept: (suggestion: Suggestion) => void;
  onReject: (suggestion: Suggestion) => void;
  jobStatus?: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  imageId?: string;
  onRetry?: (imageId: string) => Promise<void>;
}

const AgentValidationPanel: React.FC<AgentValidationPanelProps> = ({
  suggestions,
  onAccept,
  onReject,
  jobStatus,
  imageId,
  onRetry,
}) => {
  const [isRetrying, setIsRetrying] = useState(false);

  const handleRetry = async () => {
    if (!imageId || !onRetry) return;
    setIsRetrying(true);
    try {
      await onRetry(imageId);
    } finally {
      setIsRetrying(false);
    }
  };

  if (jobStatus === 'failed') {
    return (
      <div className="mt-4 p-4 bg-error-light border border-error-border rounded">
        <p className="text-error-text mb-3">検証に失敗しました</p>
        {onRetry && imageId && (
          <button
            onClick={handleRetry}
            disabled={isRetrying}
            className="px-4 py-2 bg-primary text-on-primary rounded hover:bg-primary-hover disabled:opacity-50"
          >
            {isRetrying ? '再実行中...' : '検証を再実行'}
          </button>
        )}
      </div>
    );
  }

  if (jobStatus === 'running' || jobStatus === 'pending') {
    return (
      <div className="mt-4 p-4 bg-info-light border border-info-border rounded">
        <p className="text-info-text">検証中...</p>
      </div>
    );
  }

  if (jobStatus === 'skipped') {
    return (
      <div className="mt-4 p-4 bg-neutral-100 border border-neutral-300 rounded">
        <p className="text-neutral-600">ツールが設定されていないため、エージェント検証はスキップされました</p>
      </div>
    );
  }

  if (suggestions.length === 0) {
    return (
      <div className="mt-4 p-4 bg-success-light border border-success-border rounded">
        <p className="text-success-text">問題は検出されませんでした</p>
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4">
      <h3 className="text-lg font-semibold">エージェント検証結果</h3>
      {suggestions.map((suggestion, index) => (
        <div
          key={index}
          className="p-4 bg-warning-light border border-warning-border rounded"
        >
          <div className="mb-2">
            <span className="font-semibold">{suggestion.field}</span>
          </div>
          <div className="mb-2 space-y-1">
            <div className="text-sm">
              <span className="text-neutral-600">現在: </span>
              <span className="font-mono">{suggestion.original_value}</span>
            </div>
            <div className="text-sm">
              <span className="text-neutral-600">提案: </span>
              <span className="font-mono text-info">
                {suggestion.suggested_value}
              </span>
            </div>
          </div>
          <div className="mb-3 text-sm text-neutral-700">
            <span className="font-semibold">理由: </span>
            {suggestion.reason}
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => onAccept(suggestion)}
              className="px-4 py-2 bg-primary text-on-primary rounded hover:bg-primary-hover"
            >
              採用
            </button>
            <button
              onClick={() => onReject(suggestion)}
              className="px-4 py-2 bg-neutral-300 text-neutral-700 rounded hover:bg-neutral-400"
            >
              却下
            </button>
          </div>
        </div>
      ))}
    </div>
  );
};

export default AgentValidationPanel;

import React from 'react';
import { Suggestion } from '../../types/agent';

interface AgentValidationPanelProps {
  suggestions: Suggestion[];
  onAccept: (suggestion: Suggestion) => void;
  onReject: (suggestion: Suggestion) => void;
}

const AgentValidationPanel: React.FC<AgentValidationPanelProps> = ({
  suggestions,
  onAccept,
  onReject,
}) => {
  if (suggestions.length === 0) {
    return (
      <div className="mt-4 p-4 bg-success-light border border-success-border rounded">
        <p className="text-success-text">✓ 問題は検出されませんでした</p>
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

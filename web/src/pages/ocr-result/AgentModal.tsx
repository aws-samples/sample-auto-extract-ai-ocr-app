import { useState, useEffect } from 'react';
import { Modal, Button } from '../../components/ui';
import { Tool, Suggestion } from '../../types/agent';

interface AgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  tools: Tool[];
  onRunAgent: () => Promise<Suggestion[]>;
  agentStatus: 'idle' | 'running' | 'completed';
}

export default function AgentModal({ isOpen, onClose, tools, onRunAgent, agentStatus }: AgentModalProps) {
  const [running, setRunning] = useState(false);

  useEffect(() => {
    if (agentStatus === 'running') {
      setRunning(true);
    } else {
      setRunning(false);
    }
  }, [agentStatus]);

  const handleExecute = async () => {
    setRunning(true);
    try {
      await onRunAgent();
      onClose();
    } catch {
      setRunning(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="max-w-2xl w-full mx-4 p-6 max-h-[80vh] overflow-y-auto">
      <h3 className="text-lg font-semibold mb-4">エージェント検証</h3>

      {/* ツール一覧 */}
      <div className="mb-6">
        <h4 className="text-sm font-medium text-muted mb-2">登録ツール ({tools.length}件)</h4>
        {tools.length === 0 ? (
          <p className="text-sm text-muted">ツールが登録されていません</p>
        ) : (
          <div className="space-y-2">
            {tools.map((tool, i) => (
              <div key={i} className="p-3 border border-neutral-200 rounded">
                <div className="font-medium text-sm text-neutral-800">{tool.name}</div>
                {tool.description && (
                  <div className="text-xs text-muted mt-1">{tool.description}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 実行ボタン */}
      <div className="flex justify-end gap-2">
        <Button variant="outline" size="sm" onClick={onClose}>閉じる</Button>
        <Button
          variant="primary"
          size="sm"
          onClick={handleExecute}
          disabled={running}
        >
          {running ? '検証中...' : '検証実行'}
        </Button>
      </div>
    </Modal>
  );
}

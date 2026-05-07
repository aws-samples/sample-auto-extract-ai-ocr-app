import { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import { Modal, Button, Alert } from '../../components/ui';
import api from '../../services/api';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onExecute: () => void;
  appName: string;
  loading?: boolean;
}

export default function ReExtractModal({ isOpen, onClose, onExecute, appName, loading }: Props) {
  const [showPrompt, setShowPrompt] = useState(false);
  const [customPrompt, setCustomPrompt] = useState('');
  const [originalPrompt, setOriginalPrompt] = useState('');
  const [loadingPrompt, setLoadingPrompt] = useState(false);
  const [savingPrompt, setSavingPrompt] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setShowPrompt(false);
      setError(null);
    }
  }, [isOpen]);

  const loadPrompt = async () => {
    if (!appName) return;
    setLoadingPrompt(true);
    setError(null);
    try {
      const res = await api.get(`/apps/${appName}/custom-prompt`);
      const prompt = res.data.custom_prompt || '';
      setCustomPrompt(prompt);
      setOriginalPrompt(prompt);
    } catch (err: any) {
      setError('カスタムプロンプトの読み込みに失敗しました');
    } finally {
      setLoadingPrompt(false);
    }
  };

  const handleTogglePrompt = () => {
    if (!showPrompt) {
      loadPrompt();
    }
    setShowPrompt(!showPrompt);
  };

  const promptDirty = customPrompt !== originalPrompt;

  const handleExecute = async () => {
    // プロンプトが変更されていたら先に保存
    if (promptDirty) {
      setSavingPrompt(true);
      try {
        await api.put(`/apps/${appName}/custom-prompt`, { custom_prompt: customPrompt });
        setOriginalPrompt(customPrompt);
      } catch {
        setError('カスタムプロンプトの保存に失敗しました');
        setSavingPrompt(false);
        return;
      }
      setSavingPrompt(false);
    }
    onExecute();
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="max-w-xl w-full mx-4 p-6">
      <h3 className="text-lg font-semibold mb-2">再度抽出の確認</h3>
      <p className="text-sm text-muted mb-4">
        情報抽出を最初からやり直します。現在の抽出結果は上書きされます。
      </p>

      {/* カスタムプロンプト展開 */}
      <button
        type="button"
        onClick={handleTogglePrompt}
        className="flex items-center gap-1 text-sm text-primary hover:text-primary-hover mb-3"
      >
        {showPrompt ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        カスタムプロンプトを修正
      </button>

      {showPrompt && (
        <div className="mb-4">
          {loadingPrompt ? (
            <div className="flex justify-center py-4">
              <div className="animate-spin rounded-full h-6 w-6 border-t-2 border-b-2 border-info"></div>
            </div>
          ) : (
            <textarea
              value={customPrompt}
              onChange={(e) => setCustomPrompt(e.target.value)}
              className="w-full px-3 py-2 border border-default rounded-lg text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
              rows={6}
              placeholder="抽出ルールや注意点を指定..."
            />
          )}
          {promptDirty && (
            <p className="text-xs text-info mt-1">プロンプトが変更されています。実行時に自動保存されます。</p>
          )}
        </div>
      )}

      {error && <Alert type="error" className="mb-4">{error}</Alert>}

      <div className="flex justify-end gap-2">
        <Button variant="secondary" onClick={onClose}>キャンセル</Button>
        <Button variant="primary" onClick={handleExecute} disabled={loading || savingPrompt}>
          {savingPrompt ? '保存中...' : loading ? '実行中...' : '実行'}
        </Button>
      </div>
    </Modal>
  );
}

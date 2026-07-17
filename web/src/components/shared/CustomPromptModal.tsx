import React, { useState, useEffect } from "react";
import { AlertTriangle } from "lucide-react";
import api from "../../services/api";
import { Alert, Button, Modal } from "../ui";

interface CustomPromptModalProps {
  isOpen: boolean;
  onClose: () => void;
  appName: string;
}

const CustomPromptModal: React.FC<CustomPromptModalProps> = ({
  isOpen,
  onClose,
  appName,
}) => {
  const [customPrompt, setCustomPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // モーダルが開かれたときにカスタムプロンプトを読み込む
  useEffect(() => {
    if (isOpen && appName) {
      loadCustomPrompt();
    }
  }, [isOpen, appName]);

  const loadCustomPrompt = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.get(`/apps/${appName}/custom-prompt`);
      setCustomPrompt(response.data.custom_prompt || "");
    } catch (err: any) {
      setError(`カスタムプロンプトの読み込みに失敗しました: ${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
    } finally {
      setIsLoading(false);
    }
  };

  const saveCustomPrompt = async () => {
    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);
    try {
      await api.put(`/apps/${appName}/custom-prompt`, {
        custom_prompt: customPrompt
      });
      setSuccessMessage("カスタムプロンプトを保存しました");
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err: any) {
      setError(`カスタムプロンプトの保存に失敗しました: ${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="w-full max-w-3xl max-h-[90vh] overflow-hidden">
        <div className="flex justify-between items-center border-b p-4">
          <h2 className="text-xl font-semibold">カスタムプロンプト設定</h2>
          <button
            onClick={onClose}
            className="text-neutral-500 hover:text-neutral-700"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-6 w-6"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        <div className="p-6 overflow-y-auto" style={{ maxHeight: "calc(90vh - 120px)" }}>
          {isLoading ? (
            <div className="flex justify-center items-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-info"></div>
            </div>
          ) : (
            <>
              <div className="mb-6">
                <p className="text-neutral-700 mb-4">
                  OCR処理後の情報抽出時に使用するカスタムプロンプトを設定できます。
                  特定の抽出ルールや注意点などを指定してください。
                </p>
                <div className="bg-warning-light border-l-4 border-warning-border p-4 mb-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <AlertTriangle size={20} className="text-warning" />
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-warning-text">
                        このプロンプトはOCR処理後の情報抽出時に使用されます。
                        特定のフィールドの抽出方法や、特殊なフォーマットの解釈方法などを指定できます。
                      </p>
                    </div>
                  </div>
                </div>
                <textarea
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  className="w-full px-3 py-2 border border-neutral-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
                  rows={12}
                  placeholder="例: 請求書番号は「No.」や「請求書番号:」などの後に続く数字を抽出してください。日付は「yyyy年mm月dd日」形式に統一してください。"
                ></textarea>
              </div>

              {error && (
                <Alert type="error" className="mb-4">
                  <span className="block sm:inline whitespace-pre-line">{error}</span>
                </Alert>
              )}

              {successMessage && (
                <Alert type="success" className="mb-4">
                  <span className="block sm:inline">{successMessage}</span>
                </Alert>
              )}
            </>
          )}
        </div>

        <div className="border-t p-4 flex justify-end space-x-2">
          <Button
            variant="secondary"
            onClick={onClose}
          >
            キャンセル
          </Button>
          <Button
            variant="primary"
            onClick={saveCustomPrompt}
            disabled={isSaving}
          >
            {isSaving ? "保存中..." : "保存"}
          </Button>
        </div>
    </Modal>
  );
};

export default CustomPromptModal;

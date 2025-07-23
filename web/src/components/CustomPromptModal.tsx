import React, { useState, useEffect } from "react";
import api from "../utils/api";

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
      setError(`カスタムプロンプトの読み込みに失敗しました: ${err.message}`);
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
      setError(`カスタムプロンプトの保存に失敗しました: ${err.message}`);
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl max-h-[90vh] overflow-hidden">
        <div className="flex justify-between items-center border-b p-4">
          <h2 className="text-xl font-semibold">カスタムプロンプト設定</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
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
              <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-blue-500"></div>
            </div>
          ) : (
            <>
              <div className="mb-6">
                <p className="text-gray-700 mb-4">
                  OCR処理後の情報抽出時に使用するカスタムプロンプトを設定できます。
                  特定の抽出ルールや注意点などを指定してください。
                </p>
                <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-4">
                  <div className="flex">
                    <div className="flex-shrink-0">
                      <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="ml-3">
                      <p className="text-sm text-yellow-700">
                        このプロンプトはOCR処理後の情報抽出時に使用されます。
                        特定のフィールドの抽出方法や、特殊なフォーマットの解釈方法などを指定できます。
                      </p>
                    </div>
                  </div>
                </div>
                <textarea
                  value={customPrompt}
                  onChange={(e) => setCustomPrompt(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={12}
                  placeholder="例: 請求書番号は「No.」や「請求書番号:」などの後に続く数字を抽出してください。日付は「yyyy年mm月dd日」形式に統一してください。"
                ></textarea>
              </div>

              {error && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4" role="alert">
                  <span className="block sm:inline">{error}</span>
                </div>
              )}

              {successMessage && (
                <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-4" role="alert">
                  <span className="block sm:inline">{successMessage}</span>
                </div>
              )}
            </>
          )}
        </div>

        <div className="border-t p-4 flex justify-end space-x-2">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-gray-200 text-gray-800 rounded-md hover:bg-gray-300"
          >
            キャンセル
          </button>
          <button
            onClick={saveCustomPrompt}
            className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            disabled={isSaving}
          >
            {isSaving ? "保存中..." : "保存"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default CustomPromptModal;

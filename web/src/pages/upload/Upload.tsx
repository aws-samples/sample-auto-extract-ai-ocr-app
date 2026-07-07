import { useState, useEffect, useRef, FormEvent } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import api from "../../services/api";
import { ImageFile } from "../../types/ocr";
import { useAppContext } from "../../contexts/AppContext";
import { usePolling } from "../../hooks/usePolling";
import { usePresence, PRESENCE_LIST_MODE } from "../../hooks/usePresence";
import FileList from "./FileList";
import OcrActionBar from "./OcrActionBar";
import S3SyncModal from "./S3SyncModal";
import CustomPromptModal from "../../components/shared/CustomPromptModal";
import ConfirmModal from "../../components/shared/ConfirmModal";
import LoadingToast from "./LoadingToast";

import SharingModal from "./SharingModal";

import { Alert, Button } from "../../components/ui";

import { MoreVertical, Eye, Pencil, RefreshCw, Trash2, Share2 } from "lucide-react";

function Upload() {
  const { appName } = useParams<{ appName: string }>();
  const navigate = useNavigate();
  const { apps, refreshApps, isAuthorOrAbove, isAdmin, currentUser } = useAppContext();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{
    [key: string]: number;
  }>({});
  const [s3SyncModalOpen, setS3SyncModalOpen] = useState(false);
  const [customPromptModalOpen, setCustomPromptModalOpen] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [showSharing, setShowSharing] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);
  const [pageProcessingMode, setPageProcessingMode] = useState<'combined' | 'individual'>('combined');

  // 現在選択されているアプリの情報
  const selectedApp = apps.find(app => app.name === appName);
  const appDisplayName = selectedApp?.display_name || appName;
  const s3SyncEnabled = selectedApp?.input_methods?.s3_sync || false;

  // ファイル一覧関連の状態
  const [files, setFiles] = useState<ImageFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [isEndpointWarming, setIsEndpointWarming] = useState(false);
  const warmingPollRef = useRef<NodeJS.Timeout | null>(null);
  // pollingEnabledは使用されているので削除しない
  const [pollingEnabled] = useState(true);

  // メニュー外クリックで閉じる
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) setShowMenu(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  useEffect(() => {
    return () => {
      if (warmingPollRef.current) {
        clearInterval(warmingPollRef.current);
        warmingPollRef.current = null;
      }
    };
  }, []);

  // ファイル一覧を取得
  const fetchFiles = async () => {
    try {
      const response = await api.get(`/images?app_name=${appName || ""}`);
      if (response.data && Array.isArray(response.data.images)) {
        setFiles(response.data.images);
      }
    } catch (error) {
      console.error("ファイル一覧の取得に失敗しました:", error);
    }
  };

  // OCR処理を開始
  const startOcr = async () => {
    try {
      setIsProcessing(true);
      const response = await api.post(`/apps/${appName}/jobs`);

      if (response.data && response.data.jobId) {
        // 成功したら即座に一覧を更新
        fetchFiles();
      }
    } catch (error: any) {
      console.error("OCR処理の開始に失敗しました:", error);
      
      // エンドポイント起動中エラーの場合
      if (error.response?.status === 503 && error.response?.data?.detail?.error === 'endpoint_not_ready') {
        setIsEndpointWarming(true);
        setIsProcessing(false);

        if (warmingPollRef.current) clearInterval(warmingPollRef.current);
        warmingPollRef.current = setInterval(async () => {
          try {
            const statusResponse = await api.get('/system/ocr-endpoint-status');

            if (statusResponse.data.ready) {
              if (warmingPollRef.current) {
                clearInterval(warmingPollRef.current);
                warmingPollRef.current = null;
              }
              setIsEndpointWarming(false);

              const retryResponse = await api.post(`/apps/${appName}/jobs`);
              if (retryResponse.data?.jobId) {
                fetchFiles();
              }
            }
          } catch (pollError) {
            console.error('ポーリングエラー:', pollError);
          }
        }, 10000);

        return;
      }
    } finally {
      if (!isEndpointWarming) {
        setIsProcessing(false);
      }
    }
  };

  // 一覧を更新
  const refreshFiles = () => {
    setRefreshTrigger((prev) => prev + 1);
  };

  // S3同期モーダルを開く
  const openS3SyncModal = () => {
    setS3SyncModalOpen(true);
  };

  // S3同期モーダルを閉じる
  const closeS3SyncModal = () => {
    setS3SyncModalOpen(false);
  };
  
  // カスタムプロンプトモーダルを開く
  const openCustomPromptModal = () => {
    setCustomPromptModalOpen(true);
  };

  // カスタムプロンプトモーダルを閉じる
  const closeCustomPromptModal = () => {
    setCustomPromptModalOpen(false);
  };

  // アプリ削除を実行
  const executeDelete = async () => {
    try {
      await api.delete(`/apps/${appName}`);
      await refreshApps();
      navigate('/');
    } catch (err: any) {
      setError(`削除に失敗しました: ${err.message}`);
    }
  };

  // S3ファイルインポート完了時の処理
  const handleImportComplete = () => {
    // ファイル一覧を更新
    fetchFiles();
  };

  // 未処理のファイルがあるかチェック
  const hasPendingFiles = files.some((file) => file.status === "pending");

  // 選択されたファイルにPDFが含まれているかチェック
  const hasPdfFiles = selectedFiles.some(file => file.type === "application/pdf");

  // ファイル選択時の処理
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const fileList = Array.from(e.target.files);

      // PDF・画像ファイルをフィルタリング
      const validFiles = fileList.filter(
        (file) => file.type === "application/pdf" || file.type.startsWith("image/")
      );

      if (validFiles.length !== fileList.length) {
        setError("PDF・画像ファイル（JPG、PNG）のみアップロード可能です");
      }

      if (validFiles.length > 0) {
        setSelectedFiles(validFiles);
        setError(null);
      } else {
        setError("PDF・画像ファイルを選択してください");
      }
    }
  };

  // ドラッグオーバー時の処理
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  // ドロップ時の処理
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const fileList = Array.from(e.dataTransfer.files);

      // PDF・画像ファイルをフィルタリング
      const validFiles = fileList.filter(
        (file) => file.type === "application/pdf" || file.type.startsWith("image/")
      );

      if (validFiles.length !== fileList.length) {
        setError("PDF・画像ファイル（JPG、PNG）のみアップロード可能です");
      }

      if (validFiles.length > 0) {
        setSelectedFiles(validFiles);
        setError(null);
      } else {
        setError("PDF・画像ファイルを選択してください");
      }
    }
  };

  // 選択したファイルを削除
  const removeSelectedFile = (index: number) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  // 署名付きURLを使用したアップロード処理
  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    if (selectedFiles.length === 0) {
      setError("ファイルを選択してください");
      return;
    }

    try {
      setUploading(true);
      setError(null);

      // 各ファイルを順番にアップロード
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        setUploadProgress({ ...uploadProgress, [file.name]: 0 });

        // 1. 署名付きURLを取得
        const presignedUrlResponse = await api.post(`/images/upload-url`, {
          filename: file.name,
          content_type: file.type,
          app_name: appName || undefined,
          page_processing_mode: pageProcessingMode,
        });

        const { presigned_url, s3_key, image_id } = presignedUrlResponse.data;

        // 2. 署名付きURLを使用してS3に直接アップロード
        await fetch(presigned_url, {
          method: "PUT",
          body: file,
          headers: {
            "Content-Type": file.type,
          },
        });

        // アップロード進捗を更新
        setUploadProgress((prev) => ({ ...prev, [file.name]: 50 }));

        // 3. アップロード完了を通知
        await api.post(`/images/${image_id}/upload-complete`, {
          filename: file.name,
          s3_key,
          app_name: appName || undefined,
          page_processing_mode: pageProcessingMode,
        });

        // アップロード進捗を完了に更新
        setUploadProgress((prev) => ({ ...prev, [file.name]: 100 }));
      }

      // 成功したらフォームをリセット
      setSelectedFiles([]);
      setUploadProgress({});
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
      
      // ファイル一覧を更新
      fetchFiles();
    } catch (err) {
      console.error("Upload failed:", err);
      setError("アップロードに失敗しました。もう一度お試しください。");
    } finally {
      setUploading(false);
    }
  };

  // コンポーネントマウント時にファイル一覧を取得
  useEffect(() => {
    fetchFiles();
  }, [appName, refreshTrigger]);

  // 定期的なポーリング（2秒ごと）
  usePolling(fetchFiles, { interval: 2000, enabled: pollingEnabled });

  // プレゼンス機能（一覧モード）: 各行の「見ている人」バッジ表示用
  const { byImageId: presenceByImageId } = usePresence({
    imageId: PRESENCE_LIST_MODE,
  });

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto bg-bg rounded-lg shadow-md">
        {/* アップロードフォーム */}
        <div className="p-6 border-b border-neutral-200">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold">{appDisplayName || appName}</h1>
            
            <div className="flex items-center gap-1">
              {isAuthorOrAbove && (isAdmin || selectedApp?.permission === 'owner') && (
                <button
                  onClick={() => setShowSharing(true)}
                  className="p-2 rounded-lg hover:bg-neutral-100 text-neutral-600 transition"
                  title="共有設定"
                >
                  <Share2 size={20} />
                </button>
              )}
              <div className="relative" ref={menuRef}>
                <button
                  onClick={() => setShowMenu(!showMenu)}
                  className="p-2 rounded-lg hover:bg-neutral-100 text-neutral-600 transition"
                  title="その他"
                >
                  <MoreVertical size={24} />
                </button>
                {showMenu && (
                  <div className="absolute right-0 mt-1 w-56 bg-bg rounded-lg shadow-lg border border-neutral-200 z-20 py-1">
                    <Link
                      to={`/schema-generator/${appName}`}
                      className="flex items-center px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-50"
                      onClick={() => setShowMenu(false)}
                    >
                      <Eye size={16} className="mr-3 text-neutral-500" />
                      スキーマ確認・編集
                    </Link>
                    <button
                      onClick={() => { openCustomPromptModal(); setShowMenu(false); }}
                      className="flex items-center w-full px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-50"
                    >
                      <Pencil size={16} className="mr-3 text-neutral-500" />
                      カスタムプロンプト
                    </button>
                    {s3SyncEnabled && (
                      <button
                        onClick={() => { openS3SyncModal(); setShowMenu(false); }}
                        className="flex items-center w-full px-4 py-2 text-sm text-neutral-700 hover:bg-neutral-50"
                      >
                        <RefreshCw size={16} className="mr-3 text-neutral-500" />
                        S3ファイル同期
                      </button>
                    )}
                    <div className="border-t border-neutral-200 my-1"></div>
                    <button
                      onClick={() => { setShowDeleteConfirm(true); setShowMenu(false); }}
                      className="flex items-center w-full px-4 py-2 text-sm text-danger hover:bg-danger-light"
                    >
                      <Trash2 size={16} className="mr-3" />
                      削除
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>

          {error && (
            <Alert type="error" className="mb-4">
              <span className="block sm:inline">{error}</span>
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <div
              className="border-2 border-dashed border-neutral-300 rounded-lg p-8 text-center cursor-pointer mb-4"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              {selectedFiles.length > 0 ? (
                <div>
                  <p className="text-success font-medium">
                    {selectedFiles.length}ファイルが選択されています
                  </p>
                  <ul className="mt-2 text-left max-h-40 overflow-auto">
                    {selectedFiles.map((file, index) => (
                      <li
                        key={index}
                        className="flex justify-between items-center py-1 border-b"
                      >
                        <span className="truncate max-w-xs">{file.name}</span>
                        <span className="text-sm text-neutral-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeSelectedFile(index);
                          }}
                          className="text-danger hover:text-danger-hover"
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            className="h-5 w-5"
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
                      </li>
                    ))}
                  </ul>
                </div>
              ) : (
                <div>
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="mx-auto h-12 w-12 text-neutral-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <p className="mt-2 text-sm text-neutral-600">
                    クリックしてファイルを選択
                    <br />
                    または
                    <br />
                    ファイルをドラッグ＆ドロップ
                  </p>
                  <p className="mt-1 text-xs text-neutral-500">
                    PDF・画像ファイル（JPG、PNG）のみ (最大10MB)
                  </p>
                </div>
              )}
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf,image/*"
              multiple
              onChange={handleFileChange}
              className="hidden"
            />

            {/* ページ処理モード選択 - PDFファイルが選択されている場合のみ表示 */}
            {hasPdfFiles && (
              <div className="mb-4 p-4 bg-neutral-50 rounded-lg border">
                <h3 className="text-sm font-medium text-neutral-700 mb-3">
                  複数ページPDFの処理方法
                </h3>
                <div className="space-y-3">
                  <label className="flex items-start space-x-3 cursor-pointer">
                    <input
                      type="radio"
                      name="pageProcessingMode"
                      value="combined"
                      checked={pageProcessingMode === 'combined'}
                      onChange={(e) => setPageProcessingMode(e.target.value as 'combined' | 'individual')}
                      className="mt-1 h-4 w-4 text-info focus:ring-primary border-neutral-300"
                    />
                    <div>
                      <div className="text-sm font-medium text-neutral-900">
                        全ページ統合処理
                      </div>
                      <div className="text-xs text-neutral-500">
                        複数ページを1つの画像として結合し、まとめて1つの抽出結果を生成します
                      </div>
                    </div>
                  </label>
                  
                  <label className="flex items-start space-x-3 cursor-pointer">
                    <input
                      type="radio"
                      name="pageProcessingMode"
                      value="individual"
                      checked={pageProcessingMode === 'individual'}
                      onChange={(e) => setPageProcessingMode(e.target.value as 'combined' | 'individual')}
                      className="mt-1 h-4 w-4 text-info focus:ring-primary border-neutral-300"
                    />
                    <div>
                      <div className="text-sm font-medium text-neutral-900">
                        ページ別個別処理
                      </div>
                      <div className="text-xs text-neutral-500">
                        各ページを個別に処理し、ページごとに抽出結果を生成します
                      </div>
                    </div>
                  </label>
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <Button
                type="submit"
                variant="primary"
                disabled={selectedFiles.length === 0 || uploading}
              >
                {uploading ? "アップロード中..." : "アップロード"}
              </Button>
            </div>
          </form>
        </div>

        {/* OCRアクションバー */}
        <OcrActionBar
          hasFiles={files.length > 0}
          hasPending={hasPendingFiles}
          isProcessing={isProcessing}
          onStartOcr={startOcr}
        />

        {/* ファイル一覧 */}
        <FileList files={files} onRefresh={refreshFiles} presenceByImageId={presenceByImageId} />
      </div>

      {/* S3同期モーダル */}
      <S3SyncModal
        isOpen={s3SyncModalOpen}
        onClose={closeS3SyncModal}
        appName={appName || ""}
        onImportComplete={handleImportComplete}
      />
      
      {/* カスタムプロンプトモーダル */}
      <CustomPromptModal
        isOpen={customPromptModalOpen}
        onClose={closeCustomPromptModal}
        appName={appName || ""}
      />

      {/* 削除確認モーダル */}
      <ConfirmModal
        isOpen={showDeleteConfirm}
        onClose={() => setShowDeleteConfirm(false)}
        onConfirm={executeDelete}
        title="アプリの削除"
        message={`アプリ「${appDisplayName || appName}」を削除してもよろしいですか？`}
        confirmText="削除"
        cancelText="キャンセル"
      />

      <SharingModal
        isOpen={showSharing}
        onClose={() => setShowSharing(false)}
        appName={appName || ''}
        appDisplayName={appDisplayName || appName || ''}
        currentUserId={currentUser?.id}
        onPermissionLost={() => { setShowSharing(false); refreshApps(); }}
      />

      {/* エンドポイント起動中表示 */}
      <LoadingToast
        show={isEndpointWarming}
        message={`OCRエンドポイント起動中（約10分）\n\nこの画面を開いたままにすると起動後に自動でOCR処理を開始します。\n画面を閉じてもバックグラウンドで起動処理は継続されます。`}
      />
    </div>
  );
}

export default Upload;

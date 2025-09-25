import { useState, useEffect, useRef, FormEvent } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import api from "../utils/api";
import { ImageFile } from "../types/ocr";
import { useAppContext } from "../components/AppContext";
import FileList from "../components/FileList";
import OcrActionBar from "../components/OcrActionBar";
import S3SyncModal from "../components/S3SyncModal";
import CustomPromptModal from "../components/CustomPromptModal"; // 追加

function Upload() {
  const { appName } = useParams<{ appName: string }>();
  const navigate = useNavigate();
  const { apps, refreshApps } = useAppContext();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<{
    [key: string]: number;
  }>({});
  const [s3SyncModalOpen, setS3SyncModalOpen] = useState(false);
  const [customPromptModalOpen, setCustomPromptModalOpen] = useState(false); // 追加
  const [pageProcessingMode, setPageProcessingMode] = useState<'combined' | 'individual'>('combined'); // 追加

  // 現在選択されているアプリの情報
  const selectedApp = apps.find(app => app.name === appName);
  const appDisplayName = selectedApp?.display_name || appName;
  const s3SyncEnabled = selectedApp?.input_methods?.s3_sync || false;

  // ファイル一覧関連の状態
  const [files, setFiles] = useState<ImageFile[]>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  // pollingEnabledは使用されているので削除しない
  const [pollingEnabled] = useState(true);

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
      const response = await api.post("/ocr/start", {
        app_name: appName,
      });

      if (response.data && response.data.jobId) {
        // 成功したら即座に一覧を更新
        fetchFiles();
      }
    } catch (error) {
      console.error("OCR処理の開始に失敗しました:", error);
    } finally {
      setIsProcessing(false);
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
        const presignedUrlResponse = await api.post("/generate-presigned-url", {
          filename: file.name,
          content_type: file.type,
          app_name: appName || undefined,
          page_processing_mode: pageProcessingMode, // 追加
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
        await api.post("/upload-complete", {
          image_id,
          filename: file.name,
          s3_key,
          app_name: appName || undefined,
          page_processing_mode: pageProcessingMode, // 追加
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

  // コンポーネントマウント時とrefreshTrigger変更時にファイル一覧を取得
  useEffect(() => {
    // 初回読み込み
    fetchFiles();
    
    // 定期的なポーリングを設定（2秒ごと）
    const interval = setInterval(() => {
      if (pollingEnabled) {
        fetchFiles();
      }
    }, 2000);
    
    // コンポーネントのアンマウント時にポーリングを停止
    return () => {
      clearInterval(interval);
    };
  }, [appName, refreshTrigger, pollingEnabled]);

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-4xl mx-auto bg-white rounded-lg shadow-md">
        {/* アップロードフォーム */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-2xl font-bold">{appDisplayName || appName}</h1>
            
            <div className="flex space-x-2">
              {/* スキーマ確認・編集ボタン */}
              <Link 
                to={`/schema-generator/${appName}`} 
                className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-lg flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                </svg>
                スキーマ確認・編集
              </Link>
              
              {/* カスタムプロンプト編集ボタン */}
              <button
                onClick={openCustomPromptModal}
                className="bg-indigo-500 hover:bg-indigo-600 text-white px-4 py-2 rounded-lg flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
                カスタムプロンプト
              </button>
              
              {/* 削除ボタン */}
              <button
                onClick={() => {
                  if (window.confirm(`アプリ「${appDisplayName || appName}」を削除してもよろしいですか？`)) {
                    // 削除APIを呼び出す
                    api.delete(`/apps/${appName}`)
                      .then(() => {
                        // AppContextのアプリ一覧を更新
                        refreshApps().then(() => {
                          // 成功したらトップページに戻る
                          navigate('/');
                        });
                      })
                      .catch(err => {
                        setError(`削除に失敗しました: ${err.message}`);
                      });
                  }
                }}
                className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg flex items-center"
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
                削除
              </button>
              
              {/* S3同期ボタン - s3_syncがtrueの場合のみ表示 */}
              {s3SyncEnabled && (
                <button
                  onClick={openS3SyncModal}
                  className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-200 flex items-center"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  S3ファイル同期
                </button>
              )}
            </div>
          </div>

          {error && (
            <div
              className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4"
              role="alert"
            >
              <span className="block sm:inline">{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit}>
            <div
              className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer mb-4"
              onClick={() => fileInputRef.current?.click()}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              {selectedFiles.length > 0 ? (
                <div>
                  <p className="text-green-600 font-medium">
                    {selectedFiles.length}ファイルが選択されています
                  </p>
                  <ul className="mt-2 text-left max-h-40 overflow-auto">
                    {selectedFiles.map((file, index) => (
                      <li
                        key={index}
                        className="flex justify-between items-center py-1 border-b"
                      >
                        <span className="truncate max-w-xs">{file.name}</span>
                        <span className="text-sm text-gray-500">
                          {(file.size / 1024 / 1024).toFixed(2)} MB
                        </span>
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            removeSelectedFile(index);
                          }}
                          className="text-red-500 hover:text-red-700"
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
                    className="mx-auto h-12 w-12 text-gray-400"
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
                  <p className="mt-2 text-sm text-gray-600">
                    クリックしてファイルを選択
                    <br />
                    または
                    <br />
                    ファイルをドラッグ＆ドロップ
                  </p>
                  <p className="mt-1 text-xs text-gray-500">
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
              <div className="mb-4 p-4 bg-gray-50 rounded-lg border">
                <h3 className="text-sm font-medium text-gray-700 mb-3">
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
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                    />
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        全ページ統合処理
                      </div>
                      <div className="text-xs text-gray-500">
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
                      className="mt-1 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300"
                    />
                    <div>
                      <div className="text-sm font-medium text-gray-900">
                        ページ別個別処理
                      </div>
                      <div className="text-xs text-gray-500">
                        各ページを個別に処理し、ページごとに抽出結果を生成します
                      </div>
                    </div>
                  </label>
                </div>
              </div>
            )}

            <div className="flex justify-end">
              <button
                type="submit"
                className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition duration-200 disabled:bg-gray-300 disabled:cursor-not-allowed"
                disabled={selectedFiles.length === 0 || uploading}
              >
                {uploading ? (
                  <span className="flex items-center">
                    <svg
                      className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    アップロード中...
                  </span>
                ) : (
                  "アップロード"
                )}
              </button>
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
        <FileList files={files} onRefresh={refreshFiles} />
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
    </div>
  );
}

export default Upload;

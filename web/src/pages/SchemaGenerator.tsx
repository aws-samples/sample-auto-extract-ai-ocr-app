import React, { useState, useRef, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import SchemaPreview from "../components/SchemaPreview";
import { Field } from "../types/app-schema";
import api from "../utils/api";
import { useAppContext } from "../components/AppContext";

interface SchemaData {
  name: string;
  display_name: string;
  description?: string;
  fields: Field[];
  input_methods?: {
    file_upload: boolean;
    s3_sync: boolean;
    s3_uri?: string;
  };
}

interface SchemaGeneratorProps {
  mode?: 'create' | 'view' | 'edit';
}

const SchemaGenerator: React.FC<SchemaGeneratorProps> = ({ mode = 'create' }) => {
  const navigate = useNavigate();
  const { appName: urlAppName } = useParams<{ appName: string }>();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { refreshApps } = useAppContext();
  
  // モード関連の状態
  const [isViewMode] = useState(mode === 'view');
  const [isEditMode] = useState(mode === 'edit');
  const [isCreateMode] = useState(mode === 'create');
  const [isLoading, setIsLoading] = useState(false);

  const [appName, setAppName] = useState("");
  const [appDisplayName, setAppDisplayName] = useState("");
  const [appDescription, setAppDescription] = useState("");
  const [extractionInstructions, setExtractionInstructions] = useState("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [filePreviewUrl, setFilePreviewUrl] = useState<string | null>(null);
  const [generatedSchema, setGeneratedSchema] = useState<SchemaData | null>(
    null
  );
  const [fieldsJson, setFieldsJson] = useState<string>("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);

  // 入力方法の設定
  const [fileUploadEnabled, setFileUploadEnabled] = useState(true);
  const [s3SyncEnabled, setS3SyncEnabled] = useState(false);
  const [s3Uri, setS3Uri] = useState("");

  // 既存のスキーマを読み込む（編集・閲覧モード）
  useEffect(() => {
    if ((isViewMode || isEditMode) && urlAppName) {
      setIsLoading(true);
      api
        .get(`/apps/${urlAppName}`)
        .then((response) => {
          const appData = response.data;
          setAppName(appData.name);
          setAppDisplayName(appData.display_name);
          setAppDescription(appData.description || "");
          setGeneratedSchema(appData);
          
          // fieldsのみのJSONを設定
          if (appData.fields) {
            setFieldsJson(JSON.stringify(appData.fields, null, 2));
          }
          
          // 入力方法の設定を復元
          if (appData.input_methods) {
            setFileUploadEnabled(appData.input_methods.file_upload);
            setS3SyncEnabled(appData.input_methods.s3_sync);
            setS3Uri(appData.input_methods.s3_uri || "");
          }
        })
        .catch((err) => {
          setError(`スキーマの読み込みに失敗しました: ${err.message}`);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [isViewMode, isEditMode, urlAppName]);

  // ファイル選択ダイアログを開く
  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // ファイル選択時の処理
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      processFile(file);
    }
  };

  // ドラッグ&ドロップ時の処理
  const handleFileDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      processFile(file);
    }
  };

  // ファイル処理共通関数
  const processFile = (file: File) => {
    // ファイルサイズチェック (10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError("ファイルサイズは10MB以下にしてください");
      return;
    }

    // ファイル形式チェック
    if (!isPdfFile(file) && !isImageFile(file)) {
      setError("PDF、JPG、PNGファイルのみアップロード可能です");
      return;
    }

    setUploadedFile(file);
    setError(null);

    // プレビュー用URL生成
    const fileUrl = URL.createObjectURL(file);
    setFilePreviewUrl(fileUrl);
  };

  // ファイル削除
  const removeFile = () => {
    setUploadedFile(null);
    if (filePreviewUrl) {
      URL.revokeObjectURL(filePreviewUrl);
      setFilePreviewUrl(null);
    }
  };

  // ファイル形式判定
  const isPdfFile = (file: File): boolean => {
    return file.type === "application/pdf";
  };

  const isImageFile = (file: File): boolean => {
    return file.type.startsWith("image/");
  };

  // スキーマ生成
  const generateSchema = async () => {
    if (!uploadedFile) return;

    setIsGenerating(true);
    setError(null);

    try {
      // 1. まず署名付きURLを取得
      const presignedUrlResponse = await api.post("/schema/generate-presigned-url", {
        filename: uploadedFile.name,
        content_type: uploadedFile.type
      });
      
      const { presigned_url, s3_key } = presignedUrlResponse.data;
      
      // 2. 署名付きURLを使ってS3に直接アップロード
      await fetch(presigned_url, {
        method: 'PUT',
        body: uploadedFile,
        headers: {
          'Content-Type': uploadedFile.type
        }
      });
      
      // 3. スキーマ生成APIを呼び出し
      const schemaResponse = await api.post("/schema/generate", {
        s3_key: s3_key,
        filename: uploadedFile.name,
        instructions: extractionInstructions || ""
      });

      const schema = schemaResponse.data;
      setGeneratedSchema(schema);
      
      // fieldsのみのJSONを設定
      if (schema.fields) {
        setFieldsJson(JSON.stringify(schema.fields, null, 2));
      }

      // 生成されたスキーマ名を設定
      if (schema.name && !appName) {
        setAppName(schema.name);
      }
      if (schema.display_name && !appDisplayName) {
        setAppDisplayName(schema.display_name);
      }
    } catch (err: any) {
      console.error("スキーマ生成エラー:", err);
      setError(`スキーマ生成に失敗しました: ${err.response?.data?.detail || err.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // スキーマ再生成
  const regenerateSchema = async () => {
    if (!uploadedFile) return;
    await generateSchema();
  };

  // スキーマ保存
  const saveSchema = async () => {
    if (!appName || !appDisplayName) {
      setError("アプリ名と表示名は必須です");
      setSuccessMessage(null);
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      let schemaToSave = generatedSchema;
      
      if (!schemaToSave) {
        try {
          schemaToSave = { fields: JSON.parse(fieldsJson) } as SchemaData;
        } catch (err) {
          throw new Error("JSONの形式が正しくありません");
        }
      }

      // スキーマにアプリ情報を設定
      const inputMethods: any = {
        file_upload: fileUploadEnabled,
        s3_sync: s3SyncEnabled,
      };
      
      // S3同期が有効な場合のみs3_uriを追加
      if (s3SyncEnabled && s3Uri) {
        inputMethods.s3_uri = s3Uri;
      }

      const finalSchema = {
        ...schemaToSave,
        name: appName,
        display_name: appDisplayName,
        description: appDescription,
        input_methods: inputMethods,
      };

      console.log("送信するスキーマデータ:", finalSchema);

      // 新規作成か更新かで処理を分ける
      if (isEditMode && urlAppName) {
        await api.put(`/schema/update/${urlAppName}`, finalSchema);
        setSuccessMessage("ユースケース情報を更新しました");
      } else {
        await api.post("/schema/save", finalSchema);
        setSuccessMessage("ユースケースを作成しました");
      }

      // AppContextのアプリ一覧を更新
      await refreshApps();

      // エラーメッセージをクリア
      setError(null);
      
      // 3秒後に成功メッセージを消す
      setTimeout(() => {
        setSuccessMessage(null);
      }, 3000);
    } catch (err: any) {
      console.error("スキーマ保存エラー:", err);
      const errorMessage = err.response?.data?.detail || err.message || "不明なエラーが発生しました";
      setError(`スキーマの保存に失敗しました: ${errorMessage}`);
      setSuccessMessage(null); // 成功メッセージをクリア
    } finally {
      setIsSaving(false);
    }
  };

  // JSONエディタの変更ハンドラ
  const handleFieldsJsonChange = (
    e: React.ChangeEvent<HTMLTextAreaElement>
  ) => {
    setFieldsJson(e.target.value);
    try {
      const parsedFields = JSON.parse(e.target.value);
      if (generatedSchema) {
        const updatedSchema = {
          ...generatedSchema,
          fields: parsedFields
        };
        setGeneratedSchema(updatedSchema);
      }
    } catch (err) {
      // JSONのパースエラーは無視（編集中の可能性があるため）
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">
            {isCreateMode ? '新規ユースケース作成' : 
             isViewMode ? 'ユースケース確認' : 'ユースケース編集'}
          </h1>
          
          {isViewMode ? (
            <div className="flex space-x-2">
              <button
                onClick={() => navigate(`/schema-generator/${urlAppName}`)}
                className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md"
              >
                編集
              </button>
              <button
                onClick={() => {
                  // 確認モードの場合は元のアップロード画面に戻る
                  if (urlAppName) {
                    navigate(`/app/${urlAppName}`);
                  } else {
                    // 万が一urlAppNameがない場合はトップページに戻る
                    navigate("/");
                  }
                }}
                className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
              >
                戻る
              </button>
            </div>
          ) : (
            <div className="flex space-x-2">
              <button
                onClick={saveSchema}
                className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-md disabled:bg-gray-300 disabled:cursor-not-allowed"
                disabled={isSaving}
              >
                {isSaving ? "保存中..." : "保存"}
              </button>
              <button
                onClick={() => {
                  // 編集モードの場合は元のアップロード画面に戻る
                  if (urlAppName) {
                    navigate(`/app/${urlAppName}`);
                  } else {
                    // 新規作成モードの場合はトップページに戻る
                    navigate("/");
                  }
                }}
                className="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded-md"
              >
                キャンセル
              </button>
            </div>
          )}
        </div>

        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
          </div>
        ) : (
          <div>
            {/* 成功メッセージ */}
            {successMessage && (
              <div
                className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative mb-6"
                role="alert"
              >
                <span className="block sm:inline">{successMessage}</span>
              </div>
            )}

            {/* 基本情報入力フォーム */}
            <div className="bg-white p-6 rounded-lg shadow-md mb-6">
              <h2 className="text-xl font-semibold mb-4">基本情報</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    アプリ名（英数字）
                  </label>
                  <input
                    type="text"
                    value={appName}
                    onChange={(e) => setAppName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="invoice_processor"
                    disabled={isViewMode}
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    表示名
                  </label>
                  <input
                    type="text"
                    value={appDisplayName}
                    onChange={(e) => setAppDisplayName(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="請求書処理"
                    disabled={isViewMode}
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    説明（オプション）
                  </label>
                  <textarea
                    value={appDescription}
                    onChange={(e) => setAppDescription(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows={2}
                    placeholder="このアプリケーションの説明..."
                    disabled={isViewMode}
                  ></textarea>
                </div>
              </div>

              {/* 入力方法設定 */}
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">入力方法</h3>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="fileUpload"
                      checked={fileUploadEnabled}
                      onChange={(e) => setFileUploadEnabled(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      disabled={isViewMode}
                    />
                    <label
                      htmlFor="fileUpload"
                      className="ml-2 block text-sm text-gray-900"
                    >
                      ファイルアップロード
                    </label>
                  </div>
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="s3Sync"
                      checked={s3SyncEnabled}
                      onChange={(e) => setS3SyncEnabled(e.target.checked)}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                      disabled={isViewMode}
                    />
                    <label
                      htmlFor="s3Sync"
                      className="ml-2 block text-sm text-gray-900"
                    >
                      S3同期
                    </label>
                  </div>
                  {s3SyncEnabled && (
                    <div className="pl-6">
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        S3 URI
                      </label>
                      <input
                        type="text"
                        value={s3Uri}
                        onChange={(e) => setS3Uri(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="s3://bucket-name/prefix/"
                        disabled={isViewMode}
                      />
                      <div className="mt-2 p-3 bg-yellow-50 border border-yellow-200 rounded-md">
                        <div className="flex">
                          <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                          </div>
                          <div className="ml-3">
                            <h3 className="text-sm font-medium text-yellow-800">注意事項</h3>
                            <div className="mt-2 text-sm text-yellow-700">
                              <p>API用のLambda関数からこのバケットにアクセスできるように、バケットポリシーを適切に設定してください。</p>
                              <p className="mt-1">バケットポリシーには、Lambda実行ロールに対する <code className="bg-yellow-100 px-1 py-0.5 rounded">s3:GetObject</code> 権限を付与する必要があります。</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <div
            className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4"
            role="alert"
          >
            <span className="block sm:inline">{error}</span>
          </div>
        )}

        <div className="flex flex-col lg:flex-row gap-6">
          {/* 左側: PDFアップロード領域 - 確認モードでは非表示 */}
          {!isViewMode && (
            <div className="w-full lg:w-1/2 bg-white p-6 rounded-lg shadow-md">
              <h2 className="text-xl font-semibold mb-4">
                サンプル画像アップロード
              </h2>

              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  スキーマ生成の指示（オプション）
                </label>
                <textarea
                  value={extractionInstructions}
                  onChange={(e) => setExtractionInstructions(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  rows={3}
                  placeholder="例: この請求書から、請求日、請求番号、品目、金額などの情報を抽出できるスキーマを生成してください。"
                  disabled={isViewMode}
                ></textarea>
              </div>

              {/* ファイル入力要素を追加 */}
              <input
                ref={fileInputRef}
                type="file"
                className="hidden"
                accept=".pdf,.jpg,.jpeg,.png"
                onChange={handleFileSelect}
                disabled={isViewMode}
              />

              <div
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:bg-gray-50"
                onClick={triggerFileInput}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
              >
                {!uploadedFile ? (
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
                      PDF・画像ファイル (最大10MB)
                    </p>
                  </div>
                ) : (
                  <div>
                    {isImageFile(uploadedFile) ? (
                      <img
                        src={filePreviewUrl || undefined}
                        alt="プレビュー"
                        className="mx-auto h-32 object-contain"
                      />
                    ) : (
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
                          d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                        />
                      </svg>
                    )}
                    <p className="mt-2 text-sm font-medium text-gray-900">
                      {uploadedFile.name}
                    </p>
                    <p className="text-xs text-gray-500">
                      {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile();
                      }}
                      className="mt-2 text-sm text-red-600 hover:text-red-800"
                    >
                      削除
                    </button>
                  </div>
                )}
              </div>

              <div className="mt-4 flex justify-between">
                <button
                  onClick={generateSchema}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-md disabled:bg-gray-300 disabled:cursor-not-allowed"
                  disabled={!uploadedFile || isGenerating}
                >
                  {isGenerating ? (
                    <span className="flex items-center">
                      <svg
                        className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
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
                      生成中...
                    </span>
                  ) : (
                    "スキーマを生成"
                  )}
                </button>
              </div>

              {uploadedFile && filePreviewUrl && isImageFile(uploadedFile) && (
                <div className="mt-6">
                  <h3 className="text-lg font-medium mb-2">プレビュー</h3>
                  <div className="border rounded-md overflow-hidden">
                    <img
                      src={filePreviewUrl || undefined}
                      className="w-full h-auto"
                      alt="アップロードされた画像"
                    />
                  </div>
                </div>
              )}

              {uploadedFile && filePreviewUrl && isPdfFile(uploadedFile) && (
                <div className="mt-6">
                  <h3 className="text-lg font-medium mb-2">プレビュー</h3>
                  <div className="border rounded-md p-4 bg-gray-100 text-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="mx-auto h-12 w-12 text-red-500"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                    <p className="mt-2 text-sm text-gray-600">
                      {uploadedFile.name}
                    </p>
                    <a
                      href={filePreviewUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2 inline-block text-blue-500 hover:text-blue-700"
                    >
                      PDFを開く
                    </a>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 右側: スキーマ表示・編集領域 */}
          <div className={`w-full ${!isViewMode ? 'lg:w-1/2' : ''} bg-white p-6 rounded-lg shadow-md`}>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold">スキーマ定義</h2>
              {generatedSchema && !isViewMode && (
                <button
                  onClick={regenerateSchema}
                  className="text-blue-500 hover:text-blue-700 flex items-center"
                  disabled={!uploadedFile || isGenerating}
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5 mr-1"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                    />
                  </svg>
                  再生成
                </button>
              )}
            </div>

            {generatedSchema ? (
              <div>
                {/* JSONエディタ - fieldsのみ表示 */}
                {!isViewMode && (
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      フィールド定義 (JSON)
                    </label>
                    <textarea
                      value={fieldsJson}
                      onChange={handleFieldsJsonChange}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md font-mono text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                      rows={15}
                    ></textarea>
                  </div>
                )}

                {/* スキーマプレビュー */}
                <div>
                  <h3 className="text-lg font-medium mb-2">プレビュー</h3>
                  <SchemaPreview schema={generatedSchema} />
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                {isViewMode ? (
                  <p>スキーマ情報を読み込み中...</p>
                ) : (
                  <p>
                    サンプル画像をアップロードして「スキーマを生成」ボタンをクリックしてください。
                    <br />
                    または手動でJSONを入力することもできます。
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SchemaGenerator;

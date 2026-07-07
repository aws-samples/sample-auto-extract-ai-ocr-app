import React, { useState, useRef, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Info } from "lucide-react";
import SchemaPreview from "./SchemaPreview";
import { Field } from "../../types/app-schema";
import api from "../../services/api";
import { useAppContext } from "../../contexts/AppContext";
import { Alert, Button, Skeleton } from "../../components/ui";

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
  const { refreshApps, apps, appsLoaded, isAdmin } = useAppContext();
  
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
  const [appNameError, setAppNameError] = useState<string | null>(null);

  // 入力方法の設定
  const [fileUploadEnabled, setFileUploadEnabled] = useState(true);
  const [s3SyncEnabled, setS3SyncEnabled] = useState(false);
  const [s3Uri, setS3Uri] = useState("");

  // エージェント設定
  const [agentEnabled, setAgentEnabled] = useState(false);
  const [usecaseTools, setUsecaseTools] = useState<any[]>([]);
  const [availableTools, setAvailableTools] = useState<any[]>([]);

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

          // エージェント設定を復元
          setAgentEnabled(appData.agent_enabled || false);
        })
        .catch((err) => {
          setError(`スキーマの読み込みに失敗しました: ${err.message}`);
        })
        .finally(() => {
          setIsLoading(false);
        });

      // ユースケースのツール設定と利用可能ツールを取得
      if (urlAppName) {
        api.get(`/apps/${urlAppName}/tools`).then((res) => {
          setUsecaseTools(res.data.tools || []);
        }).catch(() => {});

        api.get(`/apps/${urlAppName}/available-tools`).then((res) => {
          setAvailableTools(res.data.tools || []);
        }).catch(() => {});
      }
    }
  }, [isViewMode, isEditMode, urlAppName]);

  // edit モードの権限ガード: apps 取得完了後に判定し、編集権限（admin / editor / owner）が
  // 無ければ view モードへリダイレクトする。appsLoaded を待たないとロード途中（apps=[]）に
  // 正当な editor/owner まで誤って view へ飛ばしてしまうため、必ず取得完了を待つ。
  useEffect(() => {
    if (!isEditMode || !appsLoaded || !urlAppName) return;
    const app = apps.find((a) => a.name === urlAppName);
    const canEdit =
      isAdmin || app?.permission === "editor" || app?.permission === "owner";
    if (!canEdit) {
      navigate(`/apps/${urlAppName}/view`, { replace: true });
    }
  }, [isEditMode, appsLoaded, urlAppName, apps, isAdmin, navigate]);

  // ファイル選択ダイアログを開く
  const triggerFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // アプリ名のバリデーション
  const validateAppName = (name: string): boolean => {
    if (!name) {
      setAppNameError("アプリ名は必須です");
      return false;
    }
    // 新規作成モードのみ重複チェック
    if (isCreateMode && apps.find(app => app.name === name)) {
      setAppNameError("このアプリ名は既に使用されています");
      return false;
    }
    setAppNameError(null);
    return true;
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

  // スキーマ生成ジョブの結果をポーリング (最大 3 分)
  // Bedrock 呼び出しが 40-50 秒かかり API Gateway 29 秒制限を超えるため、
  // バックエンドは job_id を即返却し、フロントがここで完了までポーリングする。
  const pollSchemaGenerationResult = async (
    jobId: string,
    maxAttempts = 60,
    intervalMs = 3000
  ): Promise<SchemaData> => {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const res = await api.get(`/apps/schema/generate/${jobId}`);
      const { status, result, error: jobError } = res.data;

      if (status === "completed") {
        return result as SchemaData;
      }
      if (status === "failed") {
        throw new Error(jobError || "スキーマ生成に失敗しました");
      }
      // processing → wait
      await new Promise((r) => setTimeout(r, intervalMs));
    }
    throw new Error("スキーマ生成がタイムアウトしました (3 分)");
  };

  // スキーマ生成
  const generateSchema = async () => {
    if (!uploadedFile) return;

    setIsGenerating(true);
    setError(null);

    try {
      // 1. まず署名付きURLを取得
      const presignedUrlResponse = await api.post("/apps/schema/upload-url", {
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
      
      // 3. スキーマ生成ジョブを起動 (即返却)
      // appName が空でも URL パス上のみで実際には使わないので "_new" を仮置き
      const startResponse = await api.post(
        `/apps/${appName || "_new"}/schema/generate`,
        {
          s3_key: s3_key,
          filename: uploadedFile.name,
          instructions: extractionInstructions || "",
        }
      );

      const { job_id } = startResponse.data;

      // 4. ジョブ完了までポーリング (最大 3 分)
      const schema = await pollSchemaGenerationResult(job_id);

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

    // アプリ名の検証
    if (!validateAppName(appName)) {
      setError(appNameError || "アプリ名が無効です");
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
        agent_enabled: agentEnabled,
      };

      console.log("送信するスキーマデータ:", finalSchema);

      // 新規作成か更新かで処理を分ける
      if (isEditMode && urlAppName) {
        await api.put(`/apps/${urlAppName}`, finalSchema);
        setSuccessMessage("ユースケース情報を更新しました");
      } else {
        await api.post("/apps", finalSchema);
        setSuccessMessage("ユースケースを作成しました");
        
        // 新規作成時はホーム画面に遷移
        await refreshApps();
        setTimeout(() => {
          navigate("/");
        }, 500);
        return;
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
    <div className="p-6 max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <div>
            <h1 className="text-2xl font-semibold text-default">
              {isCreateMode ? '新規ユースケース作成' : 
               isViewMode ? 'ユースケース確認' : 'ユースケース編集'}
            </h1>
            <p className="text-sm text-muted mt-1">
              {isCreateMode ? 'サンプル画像からスキーマを自動生成できます' :
               isViewMode ? 'ユースケースの設定内容を確認' : 'ユースケースの設定を編集'}
            </p>
          </div>
          
          {isViewMode ? (
            <div className="flex gap-2">
              <Button variant="primary" onClick={() => navigate(`/schema-generator/${urlAppName}`)}>
                編集
              </Button>
              <Button variant="secondary" onClick={() => urlAppName ? navigate(`/app/${urlAppName}`) : navigate("/")}>
                戻る
              </Button>
            </div>
          ) : (
            <div className="flex gap-2">
              <Button variant="success" onClick={saveSchema} disabled={isSaving || !!appNameError}>
                {isSaving ? (isCreateMode ? "作成中..." : "保存中...") : (isCreateMode ? "作成" : "保存")}
              </Button>
              <Button variant="secondary" onClick={() => urlAppName ? navigate(`/app/${urlAppName}`) : navigate("/")}>
                キャンセル
              </Button>
            </div>
          )}
        </div>

        {isLoading ? (
          <div className="space-y-6">
            <div className="rounded-xl border border-default shadow-sm p-6 bg-bg">
              <Skeleton className="h-6 w-24 mb-4" />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Skeleton className="h-10" />
                <Skeleton className="h-10" />
                <div className="md:col-span-2"><Skeleton className="h-16" /></div>
              </div>
            </div>
          </div>
        ) : (
          <div>
            {/* 成功メッセージ */}
            {successMessage && (
              <Alert type="success" className="mb-6">
                <span className="block sm:inline">{successMessage}</span>
              </Alert>
            )}

            {/* 基本情報入力フォーム */}
            <div className="rounded-xl border border-default shadow-sm p-6 bg-bg mb-6">
              <h2 className="text-lg font-semibold mb-4">基本情報</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-muted mb-1">
                    アプリ名（英数字）
                  </label>
                  <input
                    type="text"
                    value={appName}
                    onChange={(e) => setAppName(e.target.value)}
                    onBlur={(e) => validateAppName(e.target.value)}
                    className="w-full px-3 py-2 border border-default rounded-lg text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary disabled:bg-surface disabled:cursor-not-allowed"
                    placeholder="invoice_processor"
                    disabled={isViewMode || isEditMode}
                  />
                  {appNameError && (
                    <p className="mt-1 text-sm text-danger">{appNameError}</p>
                  )}
                </div>
                <div>
                  <label className="block text-sm font-medium text-muted mb-1">
                    表示名
                  </label>
                  <input
                    type="text"
                    value={appDisplayName}
                    onChange={(e) => setAppDisplayName(e.target.value)}
                    className="w-full px-3 py-2 border border-default rounded-lg text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
                    placeholder="請求書処理"
                    disabled={isViewMode}
                  />
                </div>
                <div className="md:col-span-2">
                  <label className="block text-sm font-medium text-muted mb-1">
                    説明（オプション）
                  </label>
                  <textarea
                    value={appDescription}
                    onChange={(e) => setAppDescription(e.target.value)}
                    className="w-full px-3 py-2 border border-default rounded-lg text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
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
                      className="h-4 w-4 text-info focus:ring-primary border-neutral-300 rounded"
                      disabled={isViewMode}
                    />
                    <label
                      htmlFor="fileUpload"
                      className="ml-2 block text-sm text-neutral-900"
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
                      className="h-4 w-4 text-info focus:ring-primary border-neutral-300 rounded"
                      disabled={isViewMode}
                    />
                    <label
                      htmlFor="s3Sync"
                      className="ml-2 block text-sm text-neutral-900"
                    >
                      S3同期
                    </label>
                  </div>
                  {s3SyncEnabled && (
                    <div className="pl-6">
                      <div className="bg-info-light border border-info-border rounded-md p-3">
                        <div className="flex">
                          <Info size={20} className="text-info mr-2 shrink-0" />
                          <div>
                            <h4 className="text-sm font-medium text-info-text">S3同期バケット</h4>
                            <div className="mt-1 text-sm text-info-text">
                              <p>バケット: <code className="bg-info-light px-1 py-0.5 rounded font-mono">{import.meta.env.VITE_SYNC_BUCKET_NAME || 'Loading...'}</code></p>
                              <p className="mt-1">パス: <code className="bg-info-light px-1 py-0.5 rounded font-mono">{appName || 'app-name'}/</code></p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* エージェント検証設定 */}
              <div className="mt-4">
                <h3 className="text-lg font-medium mb-2">エージェント検証</h3>
                <div className="space-y-2">
                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="agentEnabled"
                      checked={agentEnabled}
                      onChange={(e) => setAgentEnabled(e.target.checked)}
                      className="h-4 w-4 text-info focus:ring-primary border-neutral-300 rounded"
                      disabled={isViewMode}
                    />
                    <label
                      htmlFor="agentEnabled"
                      className="ml-2 block text-sm text-neutral-900"
                    >
                      抽出後にエージェント検証を自動実行
                    </label>
                  </div>
                  {agentEnabled && !isEditMode && !isViewMode && (
                    <p className="pl-6 mt-2 text-sm text-neutral-500">
                      ツールの設定は保存後に編集画面で行えます
                    </p>
                  )}
                  {agentEnabled && (isEditMode || isViewMode) && (
                    <div className="pl-6 mt-2">
                      {isViewMode ? (
                        <>
                          <p className="text-sm text-muted mb-2">割当済みツール:</p>
                          {usecaseTools.length > 0 ? (
                            <div className="space-y-1">
                              {usecaseTools.map((tool: any) => (
                                <div key={tool.id} className="text-sm px-2 py-1 bg-surface rounded">
                                  <span className="font-medium">{tool.name}</span>
                                  {tool.description && (
                                    <span className="text-muted ml-2">- {tool.description}</span>
                                  )}
                                </div>
                              ))}
                            </div>
                          ) : (
                            <p className="text-sm text-muted">ツールが設定されていません</p>
                          )}
                        </>
                      ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {/* 左パネル: 割当済みツール */}
                          <div>
                            <p className="text-sm font-medium text-muted mb-2">割当済みツール</p>
                            <div className="border border-default rounded-lg p-2 min-h-[100px] space-y-1">
                              {usecaseTools.length > 0 ? usecaseTools.map((tool: any) => (
                                <div key={tool.id} className="text-sm px-2 py-1.5 bg-surface rounded">
                                  <div className="flex justify-between items-center">
                                    <span className="font-medium">{tool.name}</span>
                                    <button
                                      onClick={() => {
                                        const prev = usecaseTools;
                                        const updated = usecaseTools.filter((t: any) => t.id !== tool.id);
                                        setUsecaseTools(updated);
                                        api.put(`/apps/${urlAppName}/tools`, {
                                          tool_ids: updated.map((t: any) => t.id),
                                        }).catch((err: any) => {
                                          setUsecaseTools(prev);
                                          setError(`ツール解除に失敗しました: ${err.response?.data?.detail || err.message}`);
                                        });
                                      }}
                                      className="w-5 h-5 flex items-center justify-center rounded text-danger hover:bg-danger-light text-sm font-bold"
                                      title="解除"
                                    >
                                      −
                                    </button>
                                  </div>
                                  {tool.description && (
                                    <p className="text-xs text-muted mt-0.5">{tool.description}</p>
                                  )}
                                </div>
                              )) : (
                                <p className="text-xs text-muted p-2">ツール未設定</p>
                              )}
                            </div>
                          </div>
                          {/* 右パネル: 追加可能ツール */}
                          <div>
                            <p className="text-sm font-medium text-muted mb-2">追加可能なツール</p>
                            <div className="border border-default rounded-lg p-2 min-h-[100px] space-y-1">
                              {availableTools
                                .filter((t: any) => !usecaseTools.find((ut: any) => ut.id === t.id))
                                .map((tool: any) => (
                                  <div key={tool.id} className="text-sm px-2 py-1.5 bg-surface rounded">
                                    <div className="flex justify-between items-center">
                                      <span className="font-medium">{tool.name}</span>
                                      <button
                                        onClick={() => {
                                          const prev = usecaseTools;
                                          const updated = [...usecaseTools, tool];
                                          setUsecaseTools(updated);
                                          api.put(`/apps/${urlAppName}/tools`, {
                                            tool_ids: updated.map((t: any) => t.id),
                                          }).catch((err: any) => {
                                            setUsecaseTools(prev);
                                            setError(`ツール追加に失敗しました: ${err.response?.data?.detail || err.message}`);
                                          });
                                        }}
                                        className="w-5 h-5 flex items-center justify-center rounded text-primary hover:bg-primary/10 text-sm font-bold"
                                        title="追加"
                                      >
                                        +
                                      </button>
                                    </div>
                                    {tool.description && (
                                      <p className="text-xs text-muted mt-0.5">{tool.description}</p>
                                    )}
                                  </div>
                                ))}
                              {availableTools.filter((t: any) => !usecaseTools.find((ut: any) => ut.id === t.id)).length === 0 && (
                                <p className="text-xs text-muted p-2">追加可能なツールはありません</p>
                              )}
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {error && (
          <Alert type="error" className="mb-4">
            <span className="block sm:inline">{error}</span>
          </Alert>
        )}

        <div className="flex flex-col lg:flex-row gap-6">
          {/* 左側: PDFアップロード領域 - 確認モードでは非表示 */}
          {!isViewMode && (
            <div className="w-full lg:w-1/2 rounded-xl border border-default shadow-sm p-6 bg-bg">
              <h2 className="text-lg font-semibold mb-4">
                サンプル画像アップロード
              </h2>

              <div className="mb-4">
                <label className="block text-sm font-medium text-neutral-700 mb-1">
                  スキーマ生成の指示（オプション）
                </label>
                <textarea
                  value={extractionInstructions}
                  onChange={(e) => setExtractionInstructions(e.target.value)}
                  className="w-full px-3 py-2 border border-default rounded-lg text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
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
                className="border-2 border-dashed border-neutral-300 rounded-lg p-8 text-center cursor-pointer hover:bg-neutral-50"
                onClick={triggerFileInput}
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleFileDrop}
              >
                {!uploadedFile ? (
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
                        className="mx-auto h-12 w-12 text-neutral-400"
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
                    <p className="mt-2 text-sm font-medium text-neutral-900">
                      {uploadedFile.name}
                    </p>
                    <p className="text-xs text-neutral-500">
                      {(uploadedFile.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        removeFile();
                      }}
                      className="mt-2 text-sm text-danger hover:text-danger-text"
                    >
                      削除
                    </button>
                  </div>
                )}
              </div>

              <div className="mt-4 flex justify-between">
                <Button
                  variant="primary"
                  onClick={generateSchema}
                  disabled={!uploadedFile || isGenerating}
                >
                  {isGenerating ? "生成中... (最大3分)" : "スキーマを生成"}
                </Button>
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
                  <div className="border rounded-md p-4 bg-neutral-100 text-center">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="mx-auto h-12 w-12 text-danger"
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
                    <p className="mt-2 text-sm text-neutral-600">
                      {uploadedFile.name}
                    </p>
                    <a
                      href={filePreviewUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="mt-2 inline-block text-primary hover:text-primary-hover"
                    >
                      PDFを開く
                    </a>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* 右側: スキーマ表示・編集領域 */}
          <div className={`w-full ${!isViewMode ? 'lg:w-1/2' : ''} rounded-xl border border-default shadow-sm p-6 bg-bg`}>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-lg font-semibold">スキーマ定義</h2>
              {generatedSchema && !isViewMode && (
                <button
                  onClick={regenerateSchema}
                  className="text-primary hover:text-primary-hover flex items-center"
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
                    <label className="block text-sm font-medium text-muted mb-1">
                      フィールド定義 (JSON)
                    </label>
                    <textarea
                      value={fieldsJson}
                      onChange={handleFieldsJsonChange}
                      className="w-full px-3 py-2 border border-default rounded-lg font-mono text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
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
              <div className="text-center py-12 text-neutral-500">
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
  );
};

export default SchemaGenerator;

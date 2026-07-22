import React, { useState, useRef, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Info } from "lucide-react";
import SchemaPreview from "./SchemaPreview";
import SchemaFieldsEditor from "./SchemaFieldsEditor";
import { Field } from "../../types/app-schema";
import api from "../../services/api";
import { validateSchemaFields, isValidAppName } from "../../utils/schemaValidation";
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
  
  // モードは prop から導出する（不変なので state に持たない）
  const isViewMode = mode === 'view';
  const isEditMode = mode === 'edit';
  const isCreateMode = mode === 'create';
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
  // スキーマ生成に使ったサンプル画像の S3 キー (今セッションでアップロードした場合のみセット)
  const [uploadedS3Key, setUploadedS3Key] = useState<string | null>(null);
  // 保存済みサンプル画像 (view / edit モードで取得)
  const [savedSampleImage, setSavedSampleImage] = useState<{
    url: string;
    filename?: string;
    contentType?: string;
    s3Key?: string;
  } | null>(null);
  // スキーマ定義の編集モード (デフォルトは表示のみ)
  const [isFieldsEditing, setIsFieldsEditing] = useState(false);
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
  const [agentAutoRun, setAgentAutoRun] = useState(false);
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

          // 保存済みの生成指示プロンプトを復元 (プロンプトだけ変えて再生成できるように)
          if (appData.schema_instructions) {
            setExtractionInstructions(appData.schema_instructions);
          }

          // 入力方法の設定を復元
          if (appData.input_methods) {
            setFileUploadEnabled(appData.input_methods.file_upload);
            setS3SyncEnabled(appData.input_methods.s3_sync);
            setS3Uri(appData.input_methods.s3_uri || "");
          }

          // エージェント設定を復元
          setAgentEnabled(appData.agent_enabled || false);
          setAgentAutoRun(appData.agent_auto_run || false);
        })
        .catch((err) => {
          setError(`スキーマの読み込みに失敗しました: ${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
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

        // 保存済みサンプル画像の presigned URL を取得 (未紐付けなら url=null)
        api.get(`/apps/${urlAppName}/sample-image-url`).then((res) => {
          if (res.data.url) {
            setSavedSampleImage({
              url: res.data.url,
              filename: res.data.filename,
              contentType: res.data.content_type,
              s3Key: res.data.s3_key,
            });
          }
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
    // 形式チェック（バックの NAME_PATTERN と同期。従来はバック 400 でしか弾けなかった）
    if (!isValidAppName(name)) {
      setAppNameError("アプリ名は英数字とアンダースコアのみ使用できます");
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
    // 別ファイルに差し替えたので、以前のアップロード済み s3_key は無効化
    // (次回生成時に新しいファイルがアップロードされる)
    setUploadedS3Key(null);

    // プレビュー用URL生成
    const fileUrl = URL.createObjectURL(file);
    setFilePreviewUrl(fileUrl);
  };

  // ファイル削除
  const removeFile = () => {
    setUploadedFile(null);
    setUploadedS3Key(null);
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
  // 新規アップロードファイルがあればアップロードして生成、
  // なければ保存済みサンプル画像の s3_key を使って再生成する (プロンプトを変えたリトライ用)
  const generateSchema = async () => {
    if (!uploadedFile && !savedSampleImage?.s3Key) return;

    setIsGenerating(true);
    setError(null);

    try {
      let s3Key: string;
      let filename: string;

      if (uploadedFile && uploadedS3Key) {
        // 同じファイルで再生成 (プロンプトだけ変えたリトライ): アップロード済み s3_key を使い回す
        s3Key = uploadedS3Key;
        filename = uploadedFile.name;
      } else if (uploadedFile) {
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

        // 保存時にスキーマへ紐づけるため s3_key を保持 (再生成時の使い回しにも使う)
        setUploadedS3Key(s3_key);
        s3Key = s3_key;
        filename = uploadedFile.name;
      } else {
        // 保存済みサンプル画像で再生成 (アップロードはスキップ)
        s3Key = savedSampleImage!.s3Key!;
        filename = savedSampleImage!.filename || "sample";
      }

      // 3. スキーマ生成ジョブを起動 (即返却)
      // appName が空でも URL パス上のみで実際には使わないので "_new" を仮置き
      const startResponse = await api.post(
        `/apps/${appName || "_new"}/schema/generate`,
        {
          s3_key: s3Key,
          filename,
          instructions: extractionInstructions || "",
        }
      );

      const { job_id } = startResponse.data;

      // 4. ジョブ完了までポーリング (最大 3 分)
      const schema = await pollSchemaGenerationResult(job_id);

      setGeneratedSchema(schema);

      // 生成されたスキーマ名を設定
      if (schema.name && !appName) {
        setAppName(schema.name);
      }
      if (schema.display_name && !appDisplayName) {
        setAppDisplayName(schema.display_name);
      }
    } catch (err: any) {
      console.error("スキーマ生成エラー:", err);
      setError(`スキーマ生成に失敗しました:\n${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // スキーマ再生成 (新規アップロード or 保存済みサンプル画像のどちらかがあれば可)
  const regenerateSchema = async () => {
    if (!uploadedFile && !savedSampleImage?.s3Key) return;
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

    // フィールドの意味的検証（バックの pydantic ルールをミラー。保存前に問題を洗い出す）
    const fieldErrors = validateSchemaFields(generatedSchema?.fields || []);
    if (fieldErrors.length > 0) {
      setError(`スキーマに問題があります:\n${fieldErrors.join("\n")}`);
      setSuccessMessage(null);
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      // 未生成の場合は空フィールドで保存 (エディタで手動追加された場合は generatedSchema に反映済み)
      const schemaToSave = generatedSchema || ({ fields: [] } as unknown as SchemaData);

      // スキーマにアプリ情報を設定
      const inputMethods: any = {
        file_upload: fileUploadEnabled,
        s3_sync: s3SyncEnabled,
      };
      
      // S3同期が有効な場合のみs3_uriを追加
      if (s3SyncEnabled && s3Uri) {
        inputMethods.s3_uri = s3Uri;
      }

      const finalSchema: any = {
        ...schemaToSave,
        name: appName,
        display_name: appDisplayName,
        description: appDescription,
        input_methods: inputMethods,
        agent_enabled: agentEnabled,
        agent_auto_run: agentEnabled && agentAutoRun,
        // 生成指示プロンプトも保存し、編集画面で復元できるようにする
        schema_instructions: extractionInstructions || "",
      };

      // 今セッションで画像をアップロードした場合のみ紐付けを送信
      // (未送信なら BE 側で既存の紐付けを保持する)
      if (uploadedS3Key) {
        finalSchema.sample_image_s3_key = uploadedS3Key;
        finalSchema.sample_image_filename = uploadedFile?.name || "";
      }

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
      setError(`スキーマの保存に失敗しました:\n${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
      setSuccessMessage(null); // 成功メッセージをクリア
    } finally {
      setIsSaving(false);
    }
  };

  // フィールドエディタの変更ハンドラ
  const handleFieldsChange = (fields: Field[]) => {
    setGeneratedSchema((prev) =>
      prev ? { ...prev, fields } : ({ fields } as SchemaData)
    );
  };

  // プレビューペインを表示するか (閲覧モードではサンプル画像があるときのみ)
  const showPreviewPane = !isViewMode || !!savedSampleImage;

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
                      エージェント検証を有効にする
                    </label>
                  </div>
                  {agentEnabled && (
                    <div className="flex items-center pl-6">
                      <input
                        type="checkbox"
                        id="agentAutoRun"
                        checked={agentAutoRun}
                        onChange={(e) => setAgentAutoRun(e.target.checked)}
                        className="h-4 w-4 text-info focus:ring-primary border-neutral-300 rounded"
                        disabled={isViewMode}
                      />
                      <label
                        htmlFor="agentAutoRun"
                        className="ml-2 block text-sm text-neutral-900"
                      >
                        抽出後に自動実行する
                      </label>
                    </div>
                  )}
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
                                          setError(`ツール解除に失敗しました: ${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
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
                                            setError(`ツール追加に失敗しました: ${err?.userMessage ?? err?.message ?? "不明なエラー"}`);
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

            {/* サンプル画像アップロード (作成・編集モード) */}
            {!isViewMode && (
              <div className="rounded-xl border border-default shadow-sm p-6 bg-bg mb-6">
                <h2 className="text-lg font-semibold mb-4">サンプル画像アップロード</h2>
                  <div className="space-y-4">
                    <div>
                      <label className="block text-sm font-medium text-muted mb-1">
                        スキーマ生成の指示（オプション）
                      </label>
                      <textarea
                        value={extractionInstructions}
                        onChange={(e) => setExtractionInstructions(e.target.value)}
                        className="w-full px-3 py-2 border border-default rounded-lg text-sm bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
                        rows={3}
                        placeholder="例: この請求書から、請求日、請求番号、品目、金額などの情報を抽出できるスキーマを生成してください。"
                      ></textarea>
                    </div>
                    <div>
                      {/* ファイル入力要素 */}
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept=".pdf,.jpg,.jpeg,.png"
                        onChange={handleFileSelect}
                      />
                      <div
                        className="border-2 border-dashed border-neutral-300 rounded-lg p-4 text-center cursor-pointer hover:bg-neutral-50 flex flex-col items-center justify-center"
                        onClick={triggerFileInput}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={handleFileDrop}
                      >
                        {!uploadedFile ? (
                          <div>
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              className="mx-auto h-10 w-10 text-neutral-400"
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
                              クリックしてファイルを選択、またはドラッグ＆ドロップ
                            </p>
                            <p className="mt-1 text-xs text-neutral-500">
                              PDF・画像ファイル (最大10MB)
                            </p>
                          </div>
                        ) : (
                          <div>
                            <p className="text-sm font-medium text-neutral-900">
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
                    </div>
                  </div>
                  <div className="mt-3 flex items-center gap-4">
                    <Button
                      variant="primary"
                      onClick={generateSchema}
                      disabled={(!uploadedFile && !savedSampleImage?.s3Key) || isGenerating}
                    >
                      {isGenerating ? "生成中... (最大3分)" : "スキーマを生成"}
                    </Button>
                    {!uploadedFile && savedSampleImage?.s3Key && (
                      <span className="text-xs text-muted">
                        保存済みサンプル画像で再生成します（指示を変えてリトライ可能）
                      </span>
                    )}
                    {!generatedSchema && (
                      <button
                        onClick={() => {
                          setGeneratedSchema({ fields: [] } as unknown as SchemaData);
                          setIsFieldsEditing(true);
                        }}
                        className="text-sm text-primary hover:text-primary-hover"
                      >
                        または手動でフィールドを定義する
                      </button>
                    )}
                  </div>
              </div>
            )}
          </div>
        )}

        {error && (
          <Alert type="error" className="mb-4">
            <span className="block sm:inline whitespace-pre-line">{error}</span>
          </Alert>
        )}

        {/* プレビュー + スキーマ定義 (画像アップロード直後 / スキーマ生成後 / 既存スキーマ読込後に表示) */}
        {(generatedSchema || uploadedFile) && (
        <div className="rounded-xl border border-default shadow-sm p-6 bg-bg">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-semibold">スキーマ定義</h2>
            {generatedSchema && !isViewMode && (
              <div className="flex items-center gap-3">
                <button
                  onClick={regenerateSchema}
                  className="text-primary hover:text-primary-hover flex items-center disabled:opacity-50"
                  disabled={(!uploadedFile && !savedSampleImage?.s3Key) || isGenerating}
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
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => setIsFieldsEditing(!isFieldsEditing)}
                >
                  {isFieldsEditing ? "編集を終了" : "編集"}
                </Button>
              </div>
            )}
          </div>

          <div className={`grid grid-cols-1 ${showPreviewPane ? "lg:grid-cols-2" : ""} gap-6`}>
            {/* 左: サンプル画像プレビュー (A4 比率固定で常に全体が見える) */}
            {showPreviewPane && (
              <div className="w-full">
                <h3 className="text-sm font-medium text-muted mb-2">
                  サンプル画像プレビュー
                  {(uploadedFile?.name || savedSampleImage?.filename) && (
                    <span className="ml-2 font-normal">
                      ({uploadedFile?.name || savedSampleImage?.filename})
                    </span>
                  )}
                </h3>
                <div
                  className="w-full border rounded-md bg-neutral-50 overflow-hidden"
                  style={{ aspectRatio: "210 / 297" }}
                >
                  {uploadedFile && filePreviewUrl ? (
                    isImageFile(uploadedFile) ? (
                      <img
                        src={filePreviewUrl}
                        className="w-full h-full object-contain"
                        alt="アップロードされた画像"
                      />
                    ) : (
                      <iframe
                        src={filePreviewUrl}
                        className="w-full h-full"
                        title="アップロードされたPDFのプレビュー"
                      />
                    )
                  ) : savedSampleImage ? (
                    savedSampleImage.contentType?.startsWith("image/") ? (
                      <img
                        src={savedSampleImage.url}
                        className="w-full h-full object-contain"
                        alt="保存済みサンプル画像"
                      />
                    ) : (
                      <iframe
                        src={savedSampleImage.url}
                        className="w-full h-full"
                        title="保存済みサンプルPDFのプレビュー"
                      />
                    )
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-center text-neutral-400 text-sm">
                      <p>
                        サンプル画像をアップロードすると
                        <br />
                        ここにプレビューが表示されます
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* 右: スキーマ定義 (プレビューと同じ高さで個別スクロール) */}
            {/* lg では左の A4 ボックスがグリッド行の高さを決め、右は absolute でその高さに収まり内部スクロールする */}
            <div className={`w-full ${showPreviewPane ? "lg:relative" : ""}`}>
              <div className={showPreviewPane ? "lg:absolute lg:inset-0 lg:overflow-y-auto lg:pr-1" : ""}>
              {generatedSchema ? (
                <div>
                  {/* デフォルトは表示のみ。編集ボタンでエディタに切替 (閲覧モードは常に表示のみ) */}
                  {!isViewMode && isFieldsEditing ? (
                    <SchemaFieldsEditor
                      fields={generatedSchema.fields || []}
                      onChange={handleFieldsChange}
                    />
                  ) : (
                    <SchemaPreview schema={generatedSchema} />
                  )}
                </div>
              ) : (
                <div className="text-center py-12 text-neutral-400 text-sm">
                  {isGenerating ? (
                    <p>スキーマを生成中... (最大3分)</p>
                  ) : (
                    <p>
                      「スキーマを生成」を実行すると
                      <br />
                      ここにスキーマ定義が表示されます
                    </p>
                  )}
                </div>
              )}
              </div>
            </div>
          </div>
        </div>
        )}
    </div>
  );
};

export default SchemaGenerator;

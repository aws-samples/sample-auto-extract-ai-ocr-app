import { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Info } from "lucide-react";
import api from "../../services/api";
import { useAppContext } from "../../contexts/AppContext";
import { Alert, Button, Skeleton } from "../../components/ui";

interface UsecaseSettingsProps {
  mode?: 'create' | 'view' | 'edit';
}

const UsecaseSettings: React.FC<UsecaseSettingsProps> = ({ mode = 'create' }) => {
  const navigate = useNavigate();
  const { appName: urlAppName } = useParams<{ appName: string }>();
  const { refreshApps, apps } = useAppContext();

  const [isViewMode] = useState(mode === 'view');
  const [isEditMode] = useState(mode === 'edit');
  const [isCreateMode] = useState(mode === 'create');
  const [isLoading, setIsLoading] = useState(false);

  const [appName, setAppName] = useState("");
  const [appDisplayName, setAppDisplayName] = useState("");
  const [appDescription, setAppDescription] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [appNameError, setAppNameError] = useState<string | null>(null);
  const [isSaving, setIsSaving] = useState(false);

  // 入力方法の設定
  const [fileUploadEnabled, setFileUploadEnabled] = useState(true);
  const [s3SyncEnabled, setS3SyncEnabled] = useState(false);
  const [s3Uri, setS3Uri] = useState("");

  // エージェント設定
  const [agentEnabled, setAgentEnabled] = useState(false);
  const [usecaseTools, setUsecaseTools] = useState<any[]>([]);
  const [availableTools, setAvailableTools] = useState<any[]>([]);

  // 既存のユースケースを読み込む
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

          if (appData.input_methods) {
            setFileUploadEnabled(appData.input_methods.file_upload);
            setS3SyncEnabled(appData.input_methods.s3_sync);
            setS3Uri(appData.input_methods.s3_uri || "");
          }

          setAgentEnabled(appData.agent_enabled || false);
        })
        .catch((err) => {
          setError(`設定の読み込みに失敗しました: ${err.message}`);
        })
        .finally(() => {
          setIsLoading(false);
        });

      // ツール設定を取得
      api.get(`/usecases/${urlAppName}/tools`).then((res) => {
        setUsecaseTools(res.data.tools || []);
      }).catch(() => {});

      api.get(`/usecases/${urlAppName}/available-tools`).then((res) => {
        setAvailableTools(res.data.tools || []);
      }).catch(() => {});
    }
  }, [isViewMode, isEditMode, urlAppName]);

  const validateAppName = (name: string): boolean => {
    if (!name) {
      setAppNameError("アプリ名は必須です");
      return false;
    }
    if (isCreateMode && apps.find(app => app.name === name)) {
      setAppNameError("このアプリ名は既に使用されています");
      return false;
    }
    setAppNameError(null);
    return true;
  };

  const saveSettings = async () => {
    if (!appName || !appDisplayName) {
      setError("アプリ名と表示名は必須です");
      return;
    }

    if (!validateAppName(appName)) {
      setError(appNameError || "アプリ名が無効です");
      return;
    }

    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const inputMethods: any = {
        file_upload: fileUploadEnabled,
        s3_sync: s3SyncEnabled,
      };
      if (s3SyncEnabled && s3Uri) {
        inputMethods.s3_uri = s3Uri;
      }

      const payload = {
        name: appName,
        display_name: appDisplayName,
        description: appDescription,
        input_methods: inputMethods,
        agent_enabled: agentEnabled,
      };

      if (isEditMode && urlAppName) {
        await api.put(`/apps/${urlAppName}`, payload);
        setSuccessMessage("設定を更新しました");
        await refreshApps();
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        // 新規作成: fields なしで作成 → スキーマ編集ページへ
        await api.post("/apps", { ...payload, fields: [] });
        await refreshApps();
        navigate(`/apps/${appName}/schema`);
        return;
      }
    } catch (err: any) {
      const msg = err.response?.data?.detail || err.message || "不明なエラー";
      setError(`保存に失敗しました: ${msg}`);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-default">
            {isCreateMode ? '新規ユースケース作成' :
             isViewMode ? 'ユースケース設定' : 'ユースケース設定'}
          </h1>
          <p className="text-sm text-muted mt-1">
            {isCreateMode ? '基本情報を入力してユースケースを作成します' :
             isViewMode ? 'ユースケースの設定内容を確認' : 'ユースケースの設定を編集'}
          </p>
        </div>

        {isViewMode ? (
          <div className="flex gap-2">
            <Button variant="primary" onClick={() => navigate(`/apps/${urlAppName}/edit`)}>
              編集
            </Button>
            <Button variant="secondary" onClick={() => urlAppName ? navigate(`/app/${urlAppName}`) : navigate("/")}>
              戻る
            </Button>
          </div>
        ) : (
          <div className="flex gap-2">
            <Button variant="success" onClick={saveSettings} disabled={isSaving || !!appNameError}>
              {isSaving ? (isCreateMode ? "作成中..." : "保存中...") : (isCreateMode ? "作成してスキーマ編集へ" : "保存")}
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
          {successMessage && (
            <Alert type="success" className="mb-6">
              <span className="block sm:inline">{successMessage}</span>
            </Alert>
          )}

          {error && (
            <Alert type="error" className="mb-6">
              <span className="block sm:inline">{error}</span>
            </Alert>
          )}

          {/* 基本情報 */}
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
                  <label htmlFor="fileUpload" className="ml-2 block text-sm text-neutral-900">
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
                  <label htmlFor="s3Sync" className="ml-2 block text-sm text-neutral-900">
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
                  <label htmlFor="agentEnabled" className="ml-2 block text-sm text-neutral-900">
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
                        {/* 割当済みツール */}
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
                                      api.put(`/usecases/${urlAppName}/tools`, {
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
                        {/* 追加可能ツール */}
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
                                        api.put(`/usecases/${urlAppName}/tools`, {
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

          {/* スキーマ編集へのリンク（編集・閲覧モード） */}
          {(isEditMode || isViewMode) && urlAppName && (
            <div className="rounded-xl border border-default shadow-sm p-6 bg-bg">
              <div className="flex justify-between items-center">
                <div>
                  <h2 className="text-lg font-semibold">スキーマ定義</h2>
                  <p className="text-sm text-muted mt-1">抽出フィールドの定義を編集します</p>
                </div>
                <Button variant="primary" onClick={() => navigate(`/apps/${urlAppName}/schema`)}>
                  スキーマ編集へ →
                </Button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default UsecaseSettings;

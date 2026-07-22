import { useState, useEffect, useCallback, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { ArrowLeft, FileText, RefreshCw, Pencil, Bot } from "lucide-react";
import api from "../../services/api";
import { runAgent as apiRunAgent, getAgentToolsForImage, getAgentJobByImage, pollAgentJobStatus, updateSuggestionStatus } from "../../services/ocrApi";
import { updateVerificationStatus } from "../../services/imageApi";
import { OcrWord, OcrBoundingBox, OcrResponse, PresignedDownloadUrlResponse } from "../../types/ocr";
import { ExtractionResponse, ExtractionMapping } from "../../types/extraction";
import { Field, AppSchema } from "../../types/app-schema";
import { Suggestion, Tool, AgentJobResponse } from "../../types/agent";
import { isOcrEnabled } from "../../config";
import { usePresence } from "../../hooks/usePresence";
import PresenceBadge from "../../components/shared/PresenceBadge";
import ImagePreview from "./ImagePreview";
import OcrResultEditor from "./OcrResultEditor";
import ExtractionStatusDisplay from "./ExtractionStatusDisplay";
import ExtractedInfoDisplay from "./ExtractedInfoDisplay";
import ReExtractModal from "./ReExtractModal";
import AgentModal from "./AgentModal";
import StatusProgressBar from "../../components/ocr-result/StatusProgressBar";
import Toast from "../../components/ui/Toast";
import { Button } from "../../components/ui";

const styles = {
  container: "p-4 w-full h-screen overflow-y-auto lg:overflow-hidden",
  row: "flex flex-col lg:flex-row w-full lg:h-[calc(100vh-2rem)] gap-4",
  panel: "w-full flex flex-col",
  leftPanel: "lg:w-1/2 p-4 overflow-hidden flex flex-col min-h-[50vh] lg:min-h-0",
  rightPanel: "lg:w-1/2 p-4 flex flex-col min-h-[50vh] lg:min-h-0",
  header: "flex justify-between mb-4 flex-shrink-0 min-h-[60px] items-center flex-wrap gap-2",
  title: "text-xl font-semibold",
  loadingContainer: "flex justify-center items-center flex-grow",
  spinner:
    "animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-info",
  scrollContainer: "flex-grow overflow-y-auto min-h-0",
  contentArea: "flex-grow overflow-hidden flex flex-col min-h-0",
  scrollContainerStyle: { maxHeight: "calc(100vh - 200px)" },
};

const LoadingSpinner = () => (
  <div className={styles.loadingContainer}>
    <div className={styles.spinner}></div>
  </div>
);

function OcrResult() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const { viewers: presenceViewers } = usePresence({ imageId: id });

  // OCR結果の状態
  const [loading, setLoading] = useState(true);
  const [imageSrc, setImageSrc] = useState("");
  const [imageUrls, setImageUrls] = useState<Array<{page: number, url: string}>>([]);
  const [currentPageIndex, setCurrentPageIndex] = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [isMultipage, setIsMultipage] = useState(false);
  const [ocrWords, setOcrWords] = useState<OcrWord[]>([]);
  const [boundingBoxes, setBoundingBoxes] = useState<OcrBoundingBox[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [filename, setFilename] = useState<string>("");

  // 抽出結果の状態
  const [extractionStatus, setExtractionStatus] = useState<string>("pending");
  const [extractedInfo, setExtractedInfo] = useState<Record<string, any>>({});
  const [appFields, setAppFields] = useState<Field[]>([]);
  const [editMode, setEditMode] = useState(false);
  const [appName, setAppName] = useState<string>("");
  const [agentEnabled, setAgentEnabled] = useState(false);
  const [mapping, setMapping] = useState<ExtractionMapping>({});
  const [pollingAttemptCount, setPollingAttemptCount] = useState(0);
  const [activeView, setActiveView] = useState<"ocr" | "extraction">(
    "extraction" // OCRモードに関係なく、常に抽出画面から開始
  );
  const [verificationCompleted, setVerificationCompleted] = useState(false);

  // ポーリング設定
  const POLLING_INTERVAL = 3000; // 3秒
  const MAX_POLLING_ATTEMPTS = 20; // 最大60秒（3秒 x 20回）
  
  // 状態変数を追加
  const [toast, setToast] = useState<{
    show: boolean;
    message: string;
    type: 'success' | 'error' | 'info';
  }>({
    show: false,
    message: '',
    type: 'success'
  });
  const [agentStatus, setAgentStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [initialAgentResult, setInitialAgentResult] = useState<AgentJobResponse | null>(null);
  const [agentSuggestions, setAgentSuggestions] = useState<Suggestion[]>([]);
  const [pendingSuggestionActions, setPendingSuggestionActions] = useState<{ index: number; status: 'accepted' | 'rejected' }[]>([]);
  const suggestionsSnapshotRef = useRef<Suggestion[]>([]);
  const extractedInfoSnapshotRef = useRef<Record<string, any>>({});
  const [agentFoundIssues, setAgentFoundIssues] = useState(false);
  const [showReExtractModal, setShowReExtractModal] = useState(false);
  const [schemaChanged, setSchemaChanged] = useState(false);
  const [showAgentModal, setShowAgentModal] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);
  const statusCheckTimerRef = useRef<NodeJS.Timeout | null>(null);
  const pollingAttemptRef = useRef(0);

  // 現在のページのバウンディングボックスを生成
  const updateBoundingBoxesForCurrentPage = useCallback((pageIndex?: number) => {
    const targetPageIndex = pageIndex !== undefined ? pageIndex : currentPageIndex;
    const currentPage = targetPageIndex + 1;
    
    let currentPageWords;
    if (!isMultipage) {
      currentPageWords = ocrWords;
    } else {
      currentPageWords = ocrWords.filter(word => word.page === currentPage);
    }
    
    const boxes: OcrBoundingBox[] = currentPageWords.map((word, index) => {
      if (!word.points || word.points.length < 4) {
        return {
          id: index,
          top: 0,
          left: 0,
          width: 0,
          height: 0,
          text: word.content || "",
        };
      }

      const points = word.points;
      const minX = Math.min(...points.map((p) => p[0]));
      const minY = Math.min(...points.map((p) => p[1]));
      const maxX = Math.max(...points.map((p) => p[0]));
      const maxY = Math.max(...points.map((p) => p[1]));

      return {
        id: index,
        top: minY,
        left: minX,
        width: maxX - minX,
        height: maxY - minY,
        text: word.content || "",
      };
    });

    setBoundingBoxes(boxes);
  }, [ocrWords, currentPageIndex, isMultipage]);

  // ページ切り替え機能
  const goToPreviousPage = () => {
    if (currentPageIndex > 0) {
      const newIndex = currentPageIndex - 1;
      setCurrentPageIndex(newIndex);
      setImageSrc(imageUrls[newIndex]?.url || "");
      
      // バウンディングボックスを現在のページ用に更新
      updateBoundingBoxesForCurrentPage(newIndex);
      // 選択状態をリセット
      setSelectedIndex(null);
    }
  };

  const goToNextPage = () => {
    if (currentPageIndex < totalPages - 1) {
      const newIndex = currentPageIndex + 1;
      setCurrentPageIndex(newIndex);
      setImageSrc(imageUrls[newIndex]?.url || "");
      
      // バウンディングボックスを現在のページ用に更新
      updateBoundingBoxesForCurrentPage(newIndex);
      // 選択状態をリセット
      setSelectedIndex(null);
    }
  };

  // 表示切替
  const changeView = (view: "ocr" | "extraction") => {
    // 画面切り替え時に初期スクロールフラグをリセット
    setActiveView(view);
  };

  // ネストされたオブジェクトパスから値を取得するヘルパー関数
  const getNestedValue = (obj: any, path: string): any => {
    return path.split('.').reduce((current, key) => {
      return current && current[key] !== undefined ? current[key] : undefined;
    }, obj);
  };

  // 抽出項目クリック時のハイライト機能
  const handleExtractedFieldClick = useCallback((fieldPath: string) => {
    const wordIndices = getNestedValue(mapping, fieldPath) || [];
    if (wordIndices.length === 0) {
      return;
    }

    // 最初の単語のページを取得
    const firstWordIndex = wordIndices[0];
    const targetWord = ocrWords[firstWordIndex];
    
    if (!targetWord) {
      return;
    }
    
    const targetPage = targetWord.page || 1;
    const targetPageIndex = targetPage - 1;

    // 現在のページと異なる場合はページ切り替え
    if (targetPageIndex !== currentPageIndex) {
      setCurrentPageIndex(targetPageIndex);
      setImageSrc(imageUrls[targetPageIndex]?.url || "");
      
      // ページ切り替え後にハイライト処理を実行
      setTimeout(() => {
        highlightWordsOnCurrentPage(fieldPath, targetPage);
      }, 100); // 画像読み込み待ち
    } else {
      // 同じページの場合は即座にハイライト
      highlightWordsOnCurrentPage(fieldPath, targetPage);
    }
  }, [mapping, ocrWords, currentPageIndex, imageUrls]);

  const highlightWordsOnCurrentPage = useCallback((fieldPath: string, targetPage: number) => {
    const wordIndices = getNestedValue(mapping, fieldPath) || [];
    
    // 対象ページの単語のみをフィルタ
    const currentPageWordIndices = wordIndices.filter((index: number) => {
      const word = ocrWords[index];
      return word && (word.page === targetPage || (!word.page && targetPage === 1));
    });
    
    if (currentPageWordIndices.length === 0) {
      return;
    }
    
    // 現在のページでの相対インデックスに変換
    const currentPageWords = isMultipage 
      ? ocrWords.filter(word => (word.page === targetPage || (!word.page && targetPage === 1)))
      : ocrWords;
      
    const relativeIndices = currentPageWordIndices.map((globalIndex: number) => {
      const word = ocrWords[globalIndex];
      return currentPageWords.findIndex(pageWord => 
        pageWord.content === word.content && 
        JSON.stringify(pageWord.points) === JSON.stringify(word.points)
      );
    }).filter((index: number) => index !== -1);
    
    // 複数の単語をハイライト（最初の単語を選択状態に）
    if (relativeIndices.length > 0) {
      setSelectedIndex(relativeIndices[0]);
    }
  }, [mapping, ocrWords, isMultipage, boundingBoxes]);

  // OCR結果またはページが変更された時にバウンディングボックスを更新
  useEffect(() => {
    if (ocrWords.length > 0) {
      updateBoundingBoxesForCurrentPage();
    }
  }, [ocrWords, currentPageIndex, isMultipage, updateBoundingBoxesForCurrentPage]);

  // OCR結果の取得
  const fetchOcrResult = useCallback(async () => {
    if (!id) return;

    try {
      setLoading(true);

      // OCR結果を取得
      const response = await api.get<OcrResponse>(`/images/${id}/ocr`);
      const { ocrResult, filename, app_name } = response.data;
      
      // ファイル名を設定
      setFilename(filename || "");
      
      // app_nameを設定
      if (app_name) {
        setAppName(app_name);
      }

      // 署名付きURLで画像を取得
      try {
        const urlResponse = await api.get<PresignedDownloadUrlResponse>(
          `/images/${id}/download-url`
        );
        
        if (urlResponse.data.is_multipage && urlResponse.data.presigned_urls) {
          // 複数画像の場合
          setIsMultipage(true);
          setTotalPages(urlResponse.data.total_pages);
          setImageUrls(urlResponse.data.presigned_urls.map(item => ({
            page: item.page,
            url: item.presigned_url
          })));
          setImageSrc(urlResponse.data.presigned_urls[0]?.presigned_url || "");
          setCurrentPageIndex(0);
        } else {
          // 単一画像の場合
          setIsMultipage(false);
          setTotalPages(1);
          setImageUrls([{page: 1, url: urlResponse.data.presigned_url}]);
          setImageSrc(urlResponse.data.presigned_url);
          setCurrentPageIndex(0);
        }
      } catch (error) {
        console.error("署名付きURL取得エラー:", error);
        setImageSrc("");
        setIsMultipage(false);
        setTotalPages(1);
        setImageUrls([]);
      }

      // OCR結果を設定（OCRが有効な場合のみ）
      if (isOcrEnabled()) {
        if (ocrResult && ocrResult.words) {
          setOcrWords(ocrResult.words);
        } else if (isOcrEnabled()) {
          // OCRが有効だがOCR結果がない場合
          console.warn("OCRが有効ですが、OCR結果が見つかりません");
          setOcrWords([]);
          setBoundingBoxes([]);
        }
      } else {
        // OCRが無効な場合
        setOcrWords([]);
        setBoundingBoxes([]);
      }

      // 抽出情報を取得
      await fetchExtractionInfo();

      // 既存の agent job を取得
      try {
        const agentResult = await getAgentJobByImage(id);
        if (agentResult.status === 'completed' || agentResult.status === 'failed' || agentResult.status === 'skipped') {
          setAgentStatus('completed');
          setInitialAgentResult(agentResult);
          if ((agentResult.total_suggestions_count ?? 0) > 0) {
            setAgentFoundIssues(true);
          }
        } else if (agentResult.status === 'processing') {
          setAgentStatus('running');
          // job_id 未確定（AgentKick がジョブ作成中）のときは poll しない
          if (agentResult.job_id) {
            pollAgentJobStatus(agentResult.job_id).then(result => {
              setAgentStatus('completed');
              setInitialAgentResult({ status: 'completed', suggestions: result.suggestions });
            }).catch(() => setAgentStatus('idle'));
          }
        }
      } catch {
        // No agent result - that's fine
      }

      // ツール一覧を取得
      try {
        const toolsResponse = await getAgentToolsForImage(id);
        setTools(toolsResponse.tools || []);
      } catch {
        // Tools fetch failure is non-critical
      }

      setLoading(false);
    } catch (error) {
      console.error("OCR結果の取得に失敗しました:", error);
      setLoading(false);
    }
  }, [id]);

  // エージェント検証はユースケース単位の agent_enabled で有効化する
  useEffect(() => {
    if (!appName) return;
    api
      .get<AppSchema>(`/apps/${appName}`)
      .then((res) => setAgentEnabled(res.data.agent_enabled ?? false))
      .catch(() => setAgentEnabled(false));
  }, [appName]);

  // 抽出情報の取得
  const fetchExtractionInfo = async () => {
    if (!id) return;

    try {
      const response = await api.get<ExtractionResponse>(`/images/${id}/extraction`);

      // ステータスを明示的に設定
      setExtractionStatus(response.data.status || "pending");

      // 抽出時スキーマと現スキーマの差分（再抽出 warning 用）
      setSchemaChanged(response.data.schema_changed || false);

      // app_nameを設定
      if (response.data.app_name) {
        setAppName(response.data.app_name);
      }

      // 確認完了状態を設定
      setVerificationCompleted(response.data.verification_completed || false);

      if (response.data.status === "completed") {
        if (statusCheckTimerRef.current) {
          clearInterval(statusCheckTimerRef.current);
          statusCheckTimerRef.current = null;
        }

        if (response.data.fields) {
          setAppFields(response.data.fields);
        }

        setExtractedInfo(response.data.extracted_info || {});
        if (response.data.mapping) {
          setMapping(response.data.mapping);
        } else {
          generateSimpleMapping();
        }
      } else if (response.data.status === "failed") {
        setExtractionStatus("failed");
      } else {
        // 完了・失敗以外（ocr / extracting / processing など）は処理中。
        // 生の status は L366 で設定済みなので、ポーリングだけ開始する。
        if (!statusCheckTimerRef.current) {
          startPolling();
        }
      }
    } catch (error) {
      console.error("抽出情報の取得に失敗しました:", error);
    }
  };

  // シンプルなマッピングを生成（テキスト内容に基づく簡易的な推測）
  const generateSimpleMapping = () => {
    // 初期化
    const newMapping: ExtractionMapping = {};

    // 通常フィールドとテーブルフィールドを区別して処理
    appFields.forEach((field) => {
      // 新しいスキーマ形式に対応
      if (field.type === "string") {
        // 文字列フィールドの場合
        newMapping[field.name] = [];
      } 
      else if (field.type === "map" && field.fields) {
        // マップ型フィールドの場合
        newMapping[field.name] = {};
        field.fields.forEach(subField => {
          newMapping[field.name][subField.name] = [];
        });
      }
      else if (field.type === "list" && field.items) {
        // リスト型フィールドの場合
        if (field.items.type === "map" && field.items.fields) {
          const listData = extractedInfo[field.name];
          if (Array.isArray(listData) && listData.length > 0) {
            newMapping[field.name] = listData.map(() => {
              const rowMapping: Record<string, number[]> = {};
              field.items!.fields!.forEach(itemField => {
                rowMapping[itemField.name] = [];
              });
              return rowMapping;
            });
          } else {
            newMapping[field.name] = [];
          }
        } else {
          newMapping[field.name] = [];
        }
      } else {
        // 通常フィールドの場合
        const fieldName = field.name;
        const fieldValue = extractedInfo[fieldName];

        if (fieldValue && typeof fieldValue === "string") {
          // OCR結果から該当するテキストを検索
          const matchIndices = findMatchingIndices(fieldValue);
          newMapping[fieldName] = matchIndices;
        } else {
          newMapping[fieldName] = [];
        }
      }
    });

    setMapping(newMapping);
  };

  // テキスト値に一致するOCR結果のインデックスを探す補助関数
  const findMatchingIndices = (value: string) => {
    if (!value || typeof value !== "string" || !ocrWords || !ocrWords.length) {
      return [];
    }

    // 正規化された値
    const normalizedValue = value.replace(/\s+/g, "").toLowerCase();

    // 完全一致か部分一致するテキストのインデックスを探す
    return ocrWords
      .map((result, index) => {
        const normalizedContent = result.content
          .replace(/\s+/g, "")
          .toLowerCase();
        return normalizedContent.includes(normalizedValue) ? index : -1;
      })
      .filter((index) => index !== -1);
  };

  // 情報抽出の開始
  const startExtraction = async () => {
    if (!id) return;

    try {
      setExtractionStatus("processing");
      setPollingAttemptCount(0);

      // 抽出処理を開始（パイプラインで実行）
      await api.post(`/images/${id}/process`, {
        skip_ocr: true,
      });

      // 抽出画面に切り替え
      setActiveView("extraction");
      // ポーリングを開始
      startPolling();
    } catch (error) {
      console.error("情報抽出の開始に失敗しました:", error);
      setExtractionStatus("failed");
    }
  };

  // 抽出ステータスのポーリング
  const startPolling = () => {
    if (statusCheckTimerRef.current) {
      clearInterval(statusCheckTimerRef.current);
    }
    pollingAttemptRef.current = 0;

    statusCheckTimerRef.current = setInterval(async () => {
      pollingAttemptRef.current += 1;
      setPollingAttemptCount(pollingAttemptRef.current);

      try {
        await checkExtractionStatus();

        if (pollingAttemptRef.current >= MAX_POLLING_ATTEMPTS) {
          console.warn("抽出処理に時間がかかっています。処理は継続中です。");
          if (statusCheckTimerRef.current) {
            clearInterval(statusCheckTimerRef.current);
            statusCheckTimerRef.current = null;
          }
        }
      } catch (error) {
        console.error("ステータスチェック中にエラーが発生しました:", error);
        if (pollingAttemptRef.current >= MAX_POLLING_ATTEMPTS) {
          setExtractionStatus("failed");
          if (statusCheckTimerRef.current) {
            clearInterval(statusCheckTimerRef.current);
            statusCheckTimerRef.current = null;
          }
        }
      }
    }, POLLING_INTERVAL);
  };

  // OCR結果の保存
  const saveOcrEdit = async (words = ocrWords) => {
    if (!id) return;

    try {
      const response = await api.put(`/images/${id}/ocr`, {
        words: words,
      });

      if (response.data.status === "success") {
        showToast("OCR結果が保存されました", "success");
      }
    } catch (error) {
      console.error("OCR結果の保存に失敗しました:", error);
      showToast("OCR結果の保存に失敗しました", "error");
    }
  };

  // 抽出情報の保存
  const saveExtractedInfo = async (dataToSave?: Record<string, any>) => {
    if (!id) return;

    try {
      const infoToSave = dataToSave || extractedInfo;
      // 最新の状態を使用してPOSTリクエストを送信

      const response = await api.put(`/images/${id}/extraction`, {
        extracted_info: infoToSave,
        mapping: mapping,
      });

      if (response.data.status === "success") {
        // 保存成功後に最新の抽出情報を再取得
        await fetchExtractionInfo();
        showToast("抽出情報が保存されました", "success");
      }
    } catch (error) {
      console.error("抽出情報の保存に失敗しました:", error);
      showToast("抽出情報の保存に失敗しました", "error");
    }
  };

  // 通知を表示する関数
  const showToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({
      show: true,
      message,
      type
    });
  };
  
  // トースト通知を閉じる
  const closeToast = () => {
    setToast(prev => ({ ...prev, show: false }));
  };

  // 再抽出（or 初回抽出）完了後に Step Functions で連動した AgentKick の結果を取りに行く。
  // Agent ジョブが見つかるまで少しリトライし、processing なら追加でポーリング。
  const syncAgentAfterExtraction = async () => {
    if (!id) return;

    const MAX_LOOKUP_ATTEMPTS = 10;
    const LOOKUP_INTERVAL = 2000;

    for (let attempt = 0; attempt < MAX_LOOKUP_ATTEMPTS; attempt++) {
      try {
        const agentResult = await getAgentJobByImage(id);

        if (agentResult.status === 'skipped') {
          setAgentStatus('completed');
          return;
        }

        if (agentResult.status === 'processing') {
          setAgentStatus('running');
          // job_id がまだ無い場合（AgentKick がジョブ作成中）はリトライへ
          if (!agentResult.job_id) {
            await new Promise((resolve) => setTimeout(resolve, LOOKUP_INTERVAL));
            continue;
          }
          try {
            const result = await pollAgentJobStatus(agentResult.job_id);
            const suggestions: Suggestion[] = result.suggestions || [];
            setAgentStatus('completed');
            setAgentSuggestions(suggestions);
            if (suggestions.length > 0) {
              showToast(`${suggestions.length}件の修正提案があります`, 'info');
            }
          } catch {
            setAgentStatus('idle');
          }
          return;
        }

        if (agentResult.status === 'completed' || agentResult.status === 'failed') {
          setAgentStatus('completed');
          setAgentSuggestions(agentResult.suggestions || []);
          return;
        }
        // 'none' などジョブが未生成の場合はリトライ
      } catch {
        // ジョブ取得失敗時もリトライ
      }
      await new Promise((resolve) => setTimeout(resolve, LOOKUP_INTERVAL));
    }
  };

  // エージェント実行
  const runAgent = async (): Promise<Suggestion[]> => {
    if (!id) return [];

    try {
      setAgentStatus('running');

      const result = await apiRunAgent(id);
      const suggestions: Suggestion[] = result.suggestions || [];

      setAgentStatus('completed');
      setAgentSuggestions(suggestions);
      if (suggestions.length > 0) {
        setAgentFoundIssues(true);
        showToast(`${suggestions.length}件の修正提案があります`, 'info');
      } else {
        showToast('問題は検出されませんでした', 'success');
      }

      return suggestions;
    } catch (error) {
      console.error("エージェント実行エラー:", error);
      setAgentStatus('idle');
      showToast('エージェント実行に失敗しました', 'error');
      return [];
    }
  };

  // initialAgentResult が変わったら suggestions を更新
  // Backend already filters to pending-only with index attached
  useEffect(() => {
    if (initialAgentResult?.suggestions?.length) {
      setAgentSuggestions(initialAgentResult.suggestions);
      setAgentFoundIssues(true);
    }
  }, [initialAgentResult]);

  const handleAcceptSuggestion = (suggestion: Suggestion) => {
    setPendingSuggestionActions(prev => [
      ...prev.filter(a => a.index !== suggestion.index),
      { index: suggestion.index, status: 'accepted' },
    ]);
    setAgentSuggestions(current => current.filter(s => s.index !== suggestion.index));
  };

  const handleRejectSuggestion = (suggestion: Suggestion) => {
    setPendingSuggestionActions(prev => [
      ...prev.filter(a => a.index !== suggestion.index),
      { index: suggestion.index, status: 'rejected' },
    ]);
    setAgentSuggestions(current => current.filter(s => s.index !== suggestion.index));
  };

  const flushPendingSuggestionActions = async () => {
    if (!id || pendingSuggestionActions.length === 0) return;
    const actions = [...pendingSuggestionActions];
    setPendingSuggestionActions([]);
    let failedCount = 0;
    for (const action of actions) {
      try {
        await updateSuggestionStatus(id, action.index, action.status);
      } catch (error) {
        console.error("提案ステータス更新エラー:", error);
        failedCount++;
      }
    }
    if (failedCount > 0) {
      showToast(`提案ステータスの更新に${failedCount}件失敗しました`, 'error');
    }
  };

  // 抽出ステータスの確認
  const checkExtractionStatus = async () => {
    if (!id) return;

    try {
      const response = await api.get(`/images/${id}/status`);
      const status = response.data.extraction_status;

      if (status === "completed") {
        setExtractionStatus("completed");
        await fetchExtractionInfo();
        if (statusCheckTimerRef.current) {
          clearInterval(statusCheckTimerRef.current);
          statusCheckTimerRef.current = null;
        }
        syncAgentAfterExtraction();
      } else if (status === "failed") {
        setExtractionStatus("failed");
        if (statusCheckTimerRef.current) {
          clearInterval(statusCheckTimerRef.current);
          statusCheckTimerRef.current = null;
        }
      } else {
        // ocr / extracting / processing はそのまま保持（ステッパーがフェーズを表示する）
        setExtractionStatus(status || "processing");
      }
    } catch (error) {
      console.error("抽出ステータスの確認に失敗しました:", error);
      throw error;
    }
  };

  // 特定のフィールドに関連するテキストとバウンディングボックスをハイライト
  const highlightField = (fieldPath: string, stayOnExtractionView = true) => {
    // 新しいページ対応ハイライト機能を使用
    handleExtractedFieldClick(fieldPath);
    
    // 明示的に画面遷移が要求された場合のみ切り替える
    if (stayOnExtractionView === false) {
      setActiveView("ocr");
    }
  };

  // テーブルセルのハイライト処理（ページ指定版）
  const highlightTableCellOnPage = (cellIndices: number[], targetPage: number) => {
    const firstGlobalIndex = cellIndices[0];
    const targetWord = ocrWords[firstGlobalIndex];
    
    if (!targetWord) {
      return;
    }
    
    // 現在のページでの相対インデックスに変換
    const currentPageWords = isMultipage 
      ? ocrWords.filter(word => (word.page === targetPage || (!word.page && targetPage === 1)))
      : ocrWords;
      
    const relativeIndex = currentPageWords.findIndex(pageWord => 
      pageWord.content === targetWord.content && 
      JSON.stringify(pageWord.points) === JSON.stringify(targetWord.points)
    );
    
    if (relativeIndex !== -1) {
      setSelectedIndex(relativeIndex);
    }
  };

  // テーブルセルの関連テキストをハイライト
  const highlightTableCell = (
    fieldName: string,
    rowIndex: number,
    columnName: string
  ) => {
    const fieldMapping = mapping[fieldName];
    
    // 新しいスキーマ形式（list型）の場合
    if (Array.isArray(fieldMapping) && fieldMapping[rowIndex]) {
      const rowMapping = fieldMapping[rowIndex];
      if (typeof rowMapping === 'object' && rowMapping !== null && !Array.isArray(rowMapping)) {
        const cellIndices = rowMapping[columnName];
        
        if (Array.isArray(cellIndices) && cellIndices.length > 0) {
          // グローバルインデックスを相対インデックスに変換
          const firstGlobalIndex = cellIndices[0];
          const targetWord = ocrWords[firstGlobalIndex];
          
          if (targetWord) {
            const targetPage = targetWord.page || 1;
            const targetPageIndex = targetPage - 1;
            
            // 現在のページと異なる場合はページ切り替え
            if (targetPageIndex !== currentPageIndex) {
              setCurrentPageIndex(targetPageIndex);
              setImageSrc(imageUrls[targetPageIndex]?.url || "");
              
              // ページ切り替え後にハイライト処理を実行
              setTimeout(() => {
                highlightTableCellOnPage(cellIndices, targetPage);
              }, 100); // 画像読み込み待ち
            } else {
              // 同じページの場合は即座にハイライト
              highlightTableCellOnPage(cellIndices, targetPage);
            }
          }
          
          // 画面遷移しない
          return;
        }
      }
    }
    
    // 旧形式（table型）の場合も同様に処理
    if (Array.isArray(fieldMapping) && fieldMapping[rowIndex]) {
      const rowMapping = fieldMapping[rowIndex];
      if (typeof rowMapping === 'object' && rowMapping !== null && !Array.isArray(rowMapping)) {
        const cellIndices = rowMapping[columnName];
        
        if (Array.isArray(cellIndices) && cellIndices.length > 0) {
          // グローバルインデックスを相対インデックスに変換
          const firstGlobalIndex = cellIndices[0];
          const targetWord = ocrWords[firstGlobalIndex];
          
          if (targetWord) {
            const targetPage = targetWord.page || 1;
            const targetPageIndex = targetPage - 1;
            
            // 現在のページと異なる場合はページ切り替え
            if (targetPageIndex !== currentPageIndex) {
              setCurrentPageIndex(targetPageIndex);
              setImageSrc(imageUrls[targetPageIndex]?.url || "");
              
              // ページ切り替え後にハイライト処理を実行
              setTimeout(() => {
                highlightTableCellOnPage(cellIndices, targetPage);
              }, 100); // 画像読み込み待ち
            } else {
              // 同じページの場合は即座にハイライト
              highlightTableCellOnPage(cellIndices, targetPage);
            }
          }
          
          // 画面遷移しない
          return;
        }
      }
    }
  };

  // 抽出情報の更新通知用関数
  const updateExtractedInfo = (newInfo: Record<string, any>) => {
    setExtractedInfo(newInfo);
  };

  // OCR結果の更新
  const updateOcrResults = (updatedWords: OcrWord[]) => {
    setOcrWords(updatedWords);
    // 自動的に保存処理を実行
    saveOcrEdit(updatedWords);
  };

  // テキストボックスをクリックしたときのハンドラ
  const handleTextBoxClick = (index: number) => {
    setSelectedIndex(index);
  };

  // コンポーネントのマウント時にOCR結果を取得
  useEffect(() => {
    fetchOcrResult();

    // コンポーネントのアンマウント時にクリーンアップ
    return () => {
      if (statusCheckTimerRef.current) {
        clearInterval(statusCheckTimerRef.current);
        statusCheckTimerRef.current = null;
      }

      if (imageSrc && imageSrc.startsWith("blob:")) {
        URL.revokeObjectURL(imageSrc);
      }
    };
  }, [fetchOcrResult]);

  // 再度抽出ハンドラー
  const handleReExtract = () => {
    setShowReExtractModal(true);
  };

  const handleVerificationChange = async (completed: boolean) => {
    if (!id) return;
    
    setVerificationCompleted(completed); // 楽観的UI更新
    try {
      await updateVerificationStatus(id, completed);
      showToast(completed ? '確認完了にしました' : '確認完了を解除しました', 'success');
    } catch (error) {
      setVerificationCompleted(!completed); // エラー時はロールバック
      showToast('確認完了の更新に失敗しました', 'error');
    }
  };

  const executeReExtract = async () => {
    if (!id) return;

    try {
      setShowReExtractModal(false);
      setExtractionStatus("processing");
      setPollingAttemptCount(0);
      // 既存の Agent 結果をクリア（再抽出後に連動して走る AgentKick の結果で更新される）
      setAgentSuggestions([]);
      setAgentStatus('idle');
      showToast("情報抽出を開始しました。", "info");

      // 情報抽出のみを実行（OCRスキップ）
      await api.post(`/images/${id}/process`, { skip_ocr: true });
      
      // 抽出画面に切り替え
      setActiveView("extraction");
      // ポーリングを開始
      startPolling();
    } catch (error: any) {
      console.error("再抽出エラー:", error);
      showToast(error?.userMessage ?? error?.message ?? "再抽出に失敗しました", "error");
      setExtractionStatus("failed");
    }
  };

  // 表示タイトル
  const currentViewTitle = activeView === "ocr" ? "OCR結果" : "抽出結果";

  return (
    <div className={styles.container}>
      {/* トーストコンポーネント */}
      <Toast
        show={toast.show}
        message={toast.message}
        type={toast.type}
        onClose={closeToast}
        duration={3000}
      />
      
      {/* 再抽出モーダル（カスタムプロンプト修正統合） */}
      <ReExtractModal
        isOpen={showReExtractModal}
        onClose={() => setShowReExtractModal(false)}
        onExecute={executeReExtract}
        appName={appName}
        loading={extractionStatus === 'processing'}
        ocrEnabled={isOcrEnabled()}
        schemaChanged={schemaChanged}
        onViewOcr={() => changeView("ocr")}
      />

      {/* エージェント検証モーダル */}
      {agentEnabled && (
        <AgentModal
          isOpen={showAgentModal}
          onClose={() => setShowAgentModal(false)}
          tools={tools}
          onRunAgent={runAgent}
          agentStatus={agentStatus}
        />
      )}
      
      <div className={styles.row}>
        {/* 左側：画像プレビューとバウンディングボックス */}
        <div className={styles.leftPanel}>
          <div className={styles.header}>
            <div className="flex items-center gap-2">
              <button
                onClick={() => navigate(`/app/${appName}`)}
                disabled={!appName}
                className="p-2 text-neutral-600 hover:bg-neutral-100 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="アップロード画面に戻る"
              >
                <ArrowLeft size={20} />
              </button>
              <FileText size={20} className="text-neutral-500" />
              <span className="text-xl font-semibold text-neutral-800 truncate max-w-md">
                {filename || "画像プレビュー"}
              </span>
              <PresenceBadge viewers={presenceViewers} />
            </div>
            <div className="flex items-center gap-4 flex-wrap">
            {isMultipage && (
              <div className="flex items-center gap-2">
                <Button variant="secondary" size="sm" onClick={goToPreviousPage} disabled={currentPageIndex === 0}>←</Button>
                <span className="text-sm text-muted">
                  {currentPageIndex + 1} / {totalPages}
                </span>
                <Button variant="secondary" size="sm" onClick={goToNextPage} disabled={currentPageIndex === totalPages - 1}>→</Button>
              </div>
            )}
            </div>
          </div>

          <div className={styles.contentArea}>
            {loading ? (
              <div className={styles.loadingContainer}>
                <LoadingSpinner />
              </div>
            ) : (
              <ImagePreview
                imageSrc={imageSrc}
                boundingBoxes={boundingBoxes}
                selectedIndex={selectedIndex}
                onSelectBox={setSelectedIndex}
                onImageLoad={() => {}}
                onImageError={(error) => {
                  console.error("画像の読み込みに失敗しました:", error);
                }}
              />
            )}
          </div>
        </div>

        {/* 右側：OCR結果と情報抽出 */}
        <div className={styles.rightPanel}>
          {/* ステータスプログレスバー */}
          <div className="mb-3 px-2">
            <StatusProgressBar
              steps={[
                {
                  label: 'OCR',
                  status: ocrWords.length > 0 || !isOcrEnabled() ? 'completed' : extractionStatus === 'ocr' || extractionStatus === 'processing' ? 'processing' : 'idle',
                },
                {
                  label: '抽出',
                  status: extractionStatus === 'completed' ? 'completed' : extractionStatus === 'extracting' || extractionStatus === 'processing' ? 'processing' : extractionStatus === 'failed' ? 'failed' : 'idle',
                },
                {
                  label: 'AI検証',
                  status: agentStatus === 'completed' ? 'completed' : agentStatus === 'running' ? 'processing' : initialAgentResult?.status === 'failed' ? 'failed' : 'idle',
                },
                {
                  label: '完了',
                  status: verificationCompleted ? 'completed' : 'idle',
                },
              ]}
            />
          </div>

          <div className={styles.header}>
            <h2 className={styles.title}>{currentViewTitle}</h2>
            {activeView === "ocr" && (
              <div className="flex items-center gap-2">
                <Button variant="outline" size="sm" onClick={() => changeView("extraction")}>
                  <ArrowLeft size={14} className="mr-1" />抽出結果
                </Button>
              </div>
            )}
            {activeView === "extraction" && extractionStatus === "completed" && (
              <div className="flex items-center gap-2">
                {editMode ? (
                  <>
                    <Button variant="outline" size="sm" onClick={() => {
                      setAgentSuggestions(suggestionsSnapshotRef.current);
                      setExtractedInfo(extractedInfoSnapshotRef.current);
                      setPendingSuggestionActions([]);
                      setEditMode(false);
                    }}>キャンセル</Button>
                    <Button variant="primary" size="sm" onClick={async () => {
                      await saveExtractedInfo();
                      await flushPendingSuggestionActions();
                      setEditMode(false);
                    }}>保存</Button>
                  </>
                ) : (
                  <>
                    <Button variant="outline" size="sm" disabled={agentStatus === 'running'} onClick={() => { suggestionsSnapshotRef.current = [...agentSuggestions]; extractedInfoSnapshotRef.current = { ...extractedInfo }; setEditMode(true); }}>
                      <Pencil size={14} className="mr-1" />編集
                    </Button>
                    <Button variant="outline" size="sm" onClick={handleReExtract} disabled={loading || agentStatus === 'running'}>
                      <RefreshCw size={14} className="mr-1" />抽出設定
                    </Button>
                    {agentEnabled && (
                      <Button variant="outline" size="sm" onClick={() => setShowAgentModal(true)} disabled={agentStatus === 'running'}>
                        <Bot size={14} className="mr-1" />エージェント検証
                      </Button>
                    )}
                    <div className="h-4 w-px bg-neutral-300" />
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={verificationCompleted}
                        onChange={(e) => handleVerificationChange(e.target.checked)}
                        className="w-3.5 h-3.5 rounded focus:ring-primary accent-success"
                      />
                      <span className="text-xs text-muted whitespace-nowrap">確認完了</span>
                    </label>
                  </>
                )}
              </div>
            )}
          </div>

          {loading ? (
            <div className={styles.loadingContainer}>
              <LoadingSpinner />
            </div>
          ) : activeView === "ocr" ? (
            /* OCR結果表示 */
            <div className={styles.contentArea}>
              {isOcrEnabled() ? (
                ocrWords.length > 0 ? (
                  <>
                    <OcrResultEditor
                      ocrResults={ocrWords}
                      selectedIndex={selectedIndex}
                      onUpdateOcrResults={updateOcrResults}
                      onStartExtraction={startExtraction}
                      onSelectIndex={handleTextBoxClick}
                    />
                  </>
                ) : (
                  <div className="flex justify-center items-center flex-grow text-neutral-500">
                    <div className="text-center">
                      <p className="text-lg mb-2">OCR情報がありません</p>
                      <p className="text-sm">この画像はOCR処理されていないか、OCR結果が見つかりませんでした。</p>
                    </div>
                  </div>
                )
              ) : (
                <div className="flex justify-center items-center flex-grow text-neutral-500">
                  <div className="text-center">
                    <p className="text-lg mb-2">OCRモードが無効です</p>
                    <p className="text-sm">このデプロイメントではOCR機能が無効になっています。</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* 情報抽出結果表示 */
            <div className={styles.scrollContainer}>
              {/* 抽出ステータス表示 */}
              {extractionStatus !== "completed" ? (
                <ExtractionStatusDisplay
                  status={extractionStatus}
                  pollingAttemptCount={pollingAttemptCount}
                  onRetry={startExtraction}
                  onStartExtraction={startExtraction}
                />
              ) : (
                <>
                  {agentStatus === 'completed' && agentSuggestions.length === 0 && !agentFoundIssues && (
                    <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded flex items-center gap-2">
                      <span className="text-green-600 font-medium text-sm">✓ AI検証完了</span>
                      <span className="text-sm text-green-700">問題は検出されませんでした</span>
                    </div>
                  )}
                  <ExtractedInfoDisplay
                    extractedInfo={extractedInfo}
                    fields={appFields}
                    editMode={editMode}
                    onHighlightField={highlightField}
                    onHighlightCell={highlightTableCell}
                    onUpdateExtractedInfo={updateExtractedInfo}
                    agentSuggestions={agentSuggestions}
                    onAcceptSuggestion={handleAcceptSuggestion}
                    onRejectSuggestion={handleRejectSuggestion}
                    onEnterEditMode={() => { if (agentStatus === 'running') return; suggestionsSnapshotRef.current = [...agentSuggestions]; extractedInfoSnapshotRef.current = { ...extractedInfo }; setEditMode(true); }}
                  />
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default OcrResult;

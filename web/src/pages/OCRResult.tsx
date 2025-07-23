import { useState, useEffect, useCallback } from "react";
import { useParams } from "react-router-dom";
import api from "../utils/api";
import { OcrWord, OcrBoundingBox, OcrResponse, PresignedDownloadUrlResponse } from "../types/ocr";
import { ExtractionResponse, ExtractionMapping } from "../types/extraction";
import { Field } from "../types/app-schema";
import { isOcrEnabled } from "../config";
import ImagePreview from "../components/ImagePreview";
import OcrResultEditor from "../components/OcrResultEditor";
import ExtractionStatusDisplay from "../components/ExtractionStatusDisplay";
import ExtractedInfoDisplay from "../components/ExtractedInfoDisplay";
import Toast from "../components/Toast"; // 新しいトーストコンポーネント

const styles = {
  container: "p-4 w-full h-screen overflow-hidden",
  row: "flex flex-col lg:flex-row w-full h-[calc(100vh-2rem)] gap-4",
  panel: "w-full flex flex-col",
  leftPanel: "lg:w-1/2 p-4 overflow-hidden flex flex-col", // 50%に戻す
  rightPanel: "lg:w-1/2 p-4 flex flex-col", // 50%に戻す
  header: "flex justify-between mb-4 flex-shrink-0 min-h-[60px] items-center",
  title: "text-xl font-semibold",
  loadingContainer: "flex justify-center items-center flex-grow",
  spinner:
    "animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500",
  scrollContainer: "flex-grow overflow-y-auto min-h-0",
  contentArea: "flex-grow overflow-hidden flex flex-col min-h-0",
  scrollContainerStyle: { maxHeight: "calc(100vh - 200px)" },
};

function OcrResult() {
  const { id } = useParams<{ id: string }>();

  // OCR結果の状態
  const [loading, setLoading] = useState(true);
  const [imageSrc, setImageSrc] = useState("");
  const [ocrWords, setOcrWords] = useState<OcrWord[]>([]);
  const [boundingBoxes, setBoundingBoxes] = useState<OcrBoundingBox[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);

  // 抽出結果の状態
  const [extractionStatus, setExtractionStatus] = useState<string>("pending");
  const [extractedInfo, setExtractedInfo] = useState<Record<string, any>>({});
  const [appFields, setAppFields] = useState<Field[]>([]);
  const [appDisplayName, setAppDisplayName] = useState<string>("運送発注書");
  const [mapping, setMapping] = useState<ExtractionMapping>({});
  const [pollingAttemptCount, setPollingAttemptCount] = useState(0);
  const [activeView, setActiveView] = useState<"ocr" | "extraction">(
    "extraction" // OCRモードに関係なく、常に抽出画面から開始
  );

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
  let statusCheckTimer: NodeJS.Timeout | null = null;

  // 表示切替
  const changeView = (view: "ocr" | "extraction") => {
    // 画面切り替え時に初期スクロールフラグをリセット
    setActiveView(view);
  };

  // OCR結果の取得
  const fetchOcrResult = useCallback(async () => {
    if (!id) return;

    try {
      setLoading(true);

      // OCR結果を取得
      const response = await api.get<OcrResponse>(`/ocr/result/${id}`);
      const { ocrResult } = response.data;

      // 署名付きURLで画像を取得
      try {
        console.log("署名付きURLで画像を取得しています...");
        const urlResponse = await api.get<PresignedDownloadUrlResponse>(
          `/generate-presigned-download-url/${id}`
        );
        console.log("取得した署名付きURL:", urlResponse.data.presigned_url);
        console.log("変換済み画像:", urlResponse.data.is_converted ? "はい" : "いいえ");
        setImageSrc(urlResponse.data.presigned_url);
        console.log("署名付きURLで画像を取得しました");
      } catch (error) {
        console.error("署名付きURL取得エラー:", error);
        // フォールバック: 直接APIエンドポイントを使用
        setImageSrc(`/image/${id}`);
      }

      // OCR結果を設定（OCRが有効な場合のみ）
      if (isOcrEnabled()) {
        if (ocrResult && ocrResult.words) {
          setOcrWords(ocrResult.words);

          // バウンディングボックスを生成
          const boxes: OcrBoundingBox[] = ocrResult.words.map((word, index) => {
          if (!word.points || word.points.length < 4) {
            // ポイントがない場合はダミーのバウンディングボックスを返す
            return {
              id: index,
              top: 0,
              left: 0,
              width: 0,
              height: 0,
              text: word.content || "",
            };
          }

          // ポイントからバウンディングボックスを計算
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
        } else if (isOcrEnabled()) {
          // OCRが有効だがOCR結果がない場合
          console.warn("OCRが有効ですが、OCR結果が見つかりません");
          setOcrWords([]);
          setBoundingBoxes([]);
        }
      } else {
        // OCRが無効な場合
        console.log("OCRモードが無効のため、OCR結果の処理をスキップします");
        setOcrWords([]);
        setBoundingBoxes([]);
      }

      // 抽出情報を取得
      await fetchExtractionInfo();

      setLoading(false);
    } catch (error) {
      console.error("OCR結果の取得に失敗しました:", error);
      setLoading(false);
    }
  }, [id]);

  // 抽出情報の取得
  const fetchExtractionInfo = async () => {
    if (!id) return;

    try {
      console.log("抽出情報を取得中...");
      const response = await api.get<ExtractionResponse>(`/ocr/extract/${id}`);
      console.log("抽出情報のレスポンス:", response.data);

      // ステータスを明示的に設定
      setExtractionStatus(response.data.status || "pending");

      if (response.data.status === "completed") {
        console.log("抽出情報の取得完了 - ステータス: completed");
        // アプリ情報を更新
        setAppDisplayName(response.data.app_display_name || "運送発注書");
        
        // ポーリングを停止
        if (statusCheckTimer) {
          clearInterval(statusCheckTimer);
          statusCheckTimer = null;
        }

        // 抽出フィールド情報を更新
        if (response.data.fields) {
          setAppFields(response.data.fields);
        }

        // 抽出情報が空でも明示的に設定
        setExtractedInfo(response.data.extracted_info || {});
        if (response.data.mapping) {
          setMapping(response.data.mapping);
        } else {
          // マッピング情報がない場合はシンプルな推測を試みる
          generateSimpleMapping();
        }
      } else if (response.data.status === "failed") {
        console.log("抽出情報の取得完了 - ステータス: failed");
        setExtractionStatus("failed");
      } else {
        console.log("抽出情報の取得完了 - ステータス:", response.data.status);
        // 完了以外のステータスが返された場合は処理中とみなす
        setExtractionStatus("processing");
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
      }
      // 後方互換性のため、古い形式もサポート
      else if (field.type === "table" && "columns" in field) {
        // テーブルフィールドの場合
        const fieldName = field.name;
        const tableData = extractedInfo[fieldName];

        if (Array.isArray(tableData) && tableData.length > 0) {
          // 各行に対するマッピングを生成
          const tableMapping: Array<Record<string, number[]>> = [];

          tableData.forEach((row) => {
            const rowMapping: Record<string, number[]> = {};

            // 各列のマッピングを生成
            field.columns!.forEach((column) => {
              const columnName = column.name;
              const columnValue = row[columnName];

              if (columnValue && typeof columnValue === "string") {
                // OCR結果から該当するテキストを検索
                const matchIndices = findMatchingIndices(columnValue);
                rowMapping[columnName] = matchIndices;
              } else {
                rowMapping[columnName] = [];
              }
            });

            tableMapping.push(rowMapping);
          });

          newMapping[fieldName] = tableMapping;
        } else {
          // データがない場合は空の配列を設定
          newMapping[fieldName] = [];
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

      // 抽出処理を開始
      const response = await api.post(`/ocr/extract/${id}`, {
        words: ocrWords,
      });

      if (response.data.status === "success") {
        // 抽出画面に切り替え
        setActiveView("extraction");
        // ポーリングを開始
        startPolling();
      } else {
        setExtractionStatus("failed");
      }
    } catch (error) {
      console.error("情報抽出の開始に失敗しました:", error);
      setExtractionStatus("failed");
    }
  };

  // 抽出ステータスのポーリング
  const startPolling = () => {
    if (statusCheckTimer) {
      clearInterval(statusCheckTimer);
    }

    statusCheckTimer = setInterval(async () => {
      setPollingAttemptCount((prev) => prev + 1);

      try {
        // 情報抽出のステータスを確認
        await checkExtractionStatus();

        // タイムアウト処理
        if (
          pollingAttemptCount >= MAX_POLLING_ATTEMPTS &&
          extractionStatus === "processing"
        ) {
          console.warn("抽出処理に時間がかかっています。処理は継続中です。");
          if (statusCheckTimer) {
            clearInterval(statusCheckTimer);
            statusCheckTimer = null;
          }
        }
      } catch (error) {
        console.error("ステータスチェック中にエラーが発生しました:", error);
        // エラーが発生しても即座に失敗にはせず、タイムアウトまで継続
        if (pollingAttemptCount >= MAX_POLLING_ATTEMPTS) {
          setExtractionStatus("failed");
          if (statusCheckTimer) {
            clearInterval(statusCheckTimer);
            statusCheckTimer = null;
          }
        }
      }
    }, POLLING_INTERVAL);
  };

  // OCR結果の保存
  const saveOcrEdit = async (words = ocrWords) => {
    if (!id) return;

    try {
      const response = await api.post(`/ocr/edit/${id}`, {
        words: words,
      });

      if (response.data.status === "success") {
        console.log("OCR結果が保存されました");
        showToast("OCR結果が保存されました", "success");
      }
    } catch (error) {
      console.error("OCR結果の保存に失敗しました:", error);
      showToast("OCR結果の保存に失敗しました", "error");
    }
  };

  // 抽出情報の保存
  const saveExtractedInfo = async () => {
    if (!id) return;

    try {
      // 最新の状態を使用してPOSTリクエストを送信
      console.log("保存するデータ:", extractedInfo);
      
      const response = await api.post(`/ocr/extract/edit/${id}`, {
        extracted_info: extractedInfo,
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
  
  // 抽出ステータスの確認
  const checkExtractionStatus = async () => {
    if (!id) return;

    try {
      console.log("抽出ステータスを確認中...");
      const response = await api.get(`/ocr/extract/status/${id}`);
      const status = response.data.status;
      console.log("抽出ステータス:", status);

      if (status === "completed") {
        setExtractionStatus("completed");
        // 抽出情報を再取得
        await fetchExtractionInfo();
        // ポーリングを停止
        if (statusCheckTimer) {
          clearInterval(statusCheckTimer);
          statusCheckTimer = null;
        }
      } else if (status === "failed") {
        setExtractionStatus("failed");
        // ポーリングを停止
        if (statusCheckTimer) {
          clearInterval(statusCheckTimer);
          statusCheckTimer = null;
        }
      } else {
        setExtractionStatus("processing");
      }
    } catch (error) {
      console.error("抽出ステータスの確認に失敗しました:", error);
      throw error;
    }
  };

  // 特定のフィールドに関連するテキストとバウンディングボックスをハイライト
  const highlightField = (fieldPath: string, stayOnExtractionView = true) => {
    // ドット表記でネストされたフィールドをサポート
    const parts = fieldPath.split('.');
    
    let indices: number[] = [];
    
    if (parts.length === 1) {
      // 単一フィールド
      const fieldName = parts[0];
      const fieldMapping = mapping[fieldName];
      
      if (Array.isArray(fieldMapping)) {
        indices = fieldMapping;
      } else if (typeof fieldMapping === 'object' && fieldMapping !== null && !Array.isArray(fieldMapping)) {
        // マップ型の場合、すべてのサブフィールドのインデックスを結合
        Object.values(fieldMapping).forEach(value => {
          if (Array.isArray(value)) {
            indices = [...indices, ...value];
          }
        });
      }
    } else if (parts.length === 2) {
      // ネストされたフィールド (例: operation_info.contract_work)
      const [parentField, childField] = parts;
      const parentMapping = mapping[parentField];
      
      if (typeof parentMapping === 'object' && parentMapping !== null && !Array.isArray(parentMapping)) {
        const childIndices = parentMapping[childField];
        if (Array.isArray(childIndices)) {
          indices = childIndices;
        }
      }
    }
    
    // インデックスが見つかった場合、最初のインデックスをハイライト
    if (indices.length > 0) {
      setSelectedIndex(indices[0]);
      // 明示的に画面遷移が要求された場合のみ切り替える
      if (stayOnExtractionView === false) {
        setActiveView("ocr");
      }
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
          // 最初のマッピングされたインデックスをハイライト
          setSelectedIndex(cellIndices[0]);
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
          setSelectedIndex(cellIndices[0]);
          // 画面遷移しない
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
    console.log("OCRResult コンポーネントがマウントされました");
    fetchOcrResult();

    // コンポーネントのアンマウント時にクリーンアップ
    return () => {
      console.log("OCRResult コンポーネントがアンマウントされます");
      // ステータスチェックタイマーのクリア
      if (statusCheckTimer) {
        clearInterval(statusCheckTimer);
        statusCheckTimer = null;
      }

      // オブジェクトURLを解放
      if (imageSrc && imageSrc.startsWith("blob:")) {
        URL.revokeObjectURL(imageSrc);
      }
    };
  }, [fetchOcrResult]);

  // 表示タイトル
  const currentViewTitle = activeView === "ocr" ? "OCR結果" : "情報抽出結果";

  // ローディングスピナーコンポーネント
  const LoadingSpinner = () => (
    <div className={styles.loadingContainer}>
      <div className={styles.spinner}></div>
    </div>
  );

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
      
      <div className={styles.row}>
        {/* 左側：画像プレビューとバウンディングボックス */}
        <div className={styles.leftPanel}>
          <div className={styles.header}>
            <h2 className={styles.title}>画像プレビュー</h2>
            <div></div> {/* 右側と高さを合わせるための空のdiv */}
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
          <div className={styles.header}>
            <h2 className={styles.title}>{currentViewTitle}</h2>
            <div>
              {/* 表示切替とアクションボタン */}
              {!loading &&
                isOcrEnabled() && (
                  activeView === "ocr" ? (
                    <>
                      <button
                        onClick={() => changeView("extraction")}
                        className="bg-gray-500 text-white px-4 py-2 rounded"
                      >
                        抽出画面へ戻る
                      </button>
                    </>
                  ) : (
                    <button
                      onClick={() => changeView("ocr")}
                      className="bg-blue-500 text-white px-4 py-2 rounded"
                    >
                      OCR結果を確認/編集する
                    </button>
                  )
                )}
              
              {/* OCRが無効な場合の表示 */}
              {!loading && !isOcrEnabled() && activeView === "ocr" && (
                <div className="text-gray-500 text-sm">
                  OCRモードが無効です
                </div>
              )}
            </div>
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
                  <OcrResultEditor
                    ocrResults={ocrWords}
                    selectedIndex={selectedIndex}
                    onUpdateOcrResults={updateOcrResults}
                    onStartExtraction={startExtraction}
                    onSelectIndex={handleTextBoxClick}
                  />
                ) : (
                  <div className="flex justify-center items-center flex-grow text-gray-500">
                    <div className="text-center">
                      <p className="text-lg mb-2">OCR情報がありません</p>
                      <p className="text-sm">この画像はOCR処理されていないか、OCR結果が見つかりませんでした。</p>
                    </div>
                  </div>
                )
              ) : (
                <div className="flex justify-center items-center flex-grow text-gray-500">
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
                /* 抽出結果の表示 */
                <ExtractedInfoDisplay
                  extractedInfo={extractedInfo}
                  fields={appFields}
                  appDisplayName={appDisplayName}
                  onSave={saveExtractedInfo}
                  onHighlightField={highlightField}
                  onHighlightCell={highlightTableCell}
                  onUpdateExtractedInfo={updateExtractedInfo}
                />
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default OcrResult;

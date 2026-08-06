import React from 'react';
import { AlertCircle, Info } from 'lucide-react';

interface ExtractionStatusDisplayProps {
  status: string;
  pollingAttemptCount: number;
  onRetry: () => void;
  onStartExtraction: () => void;
}

const ExtractionStatusDisplay: React.FC<ExtractionStatusDisplayProps> = ({
  status,
  pollingAttemptCount,
  onRetry,
  onStartExtraction
}) => {
  // ステータスに応じたメッセージとアクション
  const renderContent = () => {
    switch (status) {
      case 'ocr':
      case 'extracting':
      case 'processing':
        return (
          <div className="text-center py-10">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-info mx-auto mb-4"></div>
            <p className="text-lg font-medium mb-2">
              {status === 'ocr' ? 'OCR処理中...' : status === 'extracting' ? '情報抽出中...' : '情報抽出処理中...'}
            </p>
            <p className="text-sm text-neutral-500">
              {pollingAttemptCount > 30
                ? '処理に時間がかかっています。しばらくお待ちください。'
                : status === 'ocr' ? '文書を OCR で読み取っています。' : '文書から情報を抽出しています。'}
            </p>
          </div>
        );
      
      case 'failed':
        return (
          <div className="text-center py-10">
            <div className="bg-danger-light text-danger rounded-full h-12 w-12 flex items-center justify-center mx-auto mb-4">
              <AlertCircle size={32} />
            </div>
            <p className="text-lg font-medium mb-2">情報抽出に失敗しました</p>
            <p className="text-sm text-neutral-500 mb-4">
              処理中にエラーが発生しました。もう一度お試しください。
            </p>
            <button 
              onClick={onRetry}
              className="bg-primary text-on-primary px-4 py-2 rounded hover:bg-primary-hover transition-colors"
            >
              再試行
            </button>
          </div>
        );
      
      default:
        return (
          <div className="text-center py-10">
            <div className="bg-warning-light text-warning-text rounded-full h-12 w-12 flex items-center justify-center mx-auto mb-4">
              <Info size={32} />
            </div>
            <p className="text-lg font-medium mb-2">情報抽出が必要です</p>
            <p className="text-sm text-neutral-500 mb-4">
              OCR結果から情報を抽出するには、抽出処理を開始してください。
            </p>
            <button 
              onClick={onStartExtraction}
              className="bg-primary text-on-primary px-4 py-2 rounded hover:bg-primary-hover transition-colors"
            >
              情報抽出を開始
            </button>
          </div>
        );
    }
  };

  return (
    <div className="bg-bg rounded-lg border border-neutral-200 p-4">
      {renderContent()}
    </div>
  );
};

export default ExtractionStatusDisplay;

import React from 'react';

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
      case 'processing':
        return (
          <div className="text-center py-10">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-lg font-medium mb-2">情報抽出処理中...</p>
            <p className="text-sm text-gray-500">
              {pollingAttemptCount > 30 
                ? '処理に時間がかかっています。しばらくお待ちください。' 
                : '文書から情報を抽出しています。'}
            </p>
          </div>
        );
      
      case 'failed':
        return (
          <div className="text-center py-10">
            <div className="bg-red-100 text-red-600 rounded-full h-12 w-12 flex items-center justify-center mx-auto mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-lg font-medium mb-2">情報抽出に失敗しました</p>
            <p className="text-sm text-gray-500 mb-4">
              処理中にエラーが発生しました。もう一度お試しください。
            </p>
            <button 
              onClick={onRetry}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
            >
              再試行
            </button>
          </div>
        );
      
      default:
        return (
          <div className="text-center py-10">
            <div className="bg-yellow-100 text-yellow-600 rounded-full h-12 w-12 flex items-center justify-center mx-auto mb-4">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <p className="text-lg font-medium mb-2">情報抽出が必要です</p>
            <p className="text-sm text-gray-500 mb-4">
              OCR結果から情報を抽出するには、抽出処理を開始してください。
            </p>
            <button 
              onClick={onStartExtraction}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
            >
              情報抽出を開始
            </button>
          </div>
        );
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      {renderContent()}
    </div>
  );
};

export default ExtractionStatusDisplay;

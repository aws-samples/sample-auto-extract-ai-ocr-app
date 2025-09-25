import React from 'react';

interface OcrActionBarProps {
  isProcessing: boolean;
  hasPending: boolean;
  hasFiles: boolean;
  onStartOcr: () => void;
}

const OcrActionBar: React.FC<OcrActionBarProps> = ({ isProcessing, hasPending, hasFiles, onStartOcr }) => {
  // ボタンのテキスト
  const buttonText = isProcessing ? 'OCR処理中...' : 'OCR処理開始';
  
  // ボタンの無効化条件
  const isDisabled = isProcessing || !hasPending || !hasFiles;

  return (
    <div className="p-4 border-b border-gray-200 flex justify-between items-center">
      <h3 className="text-xl font-medium flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
        </svg>
        アップロード済みファイル
      </h3>
      <button 
        onClick={onStartOcr} 
        className="bg-green-500 text-white py-2 px-4 rounded-lg hover:bg-green-600 transition duration-200 flex items-center" 
        disabled={isDisabled}
      >
        {isProcessing ? (
          <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z" clipRule="evenodd" />
          </svg>
        )}
        {buttonText}
      </button>
    </div>
  );
};

export default OcrActionBar;

import React from 'react';

interface StatusBadgeProps {
  status: string;
}

const StatusBadge: React.FC<StatusBadgeProps> = ({ status }) => {
  // ステータスに応じたスタイルとラベルを設定
  const getStatusStyle = () => {
    switch (status) {
      case 'uploading':
      case 'converting':
        return 'bg-info-light text-info-text';
      case 'pending':
        return 'bg-warning-light text-warning-text';
      case 'processing':
        return 'bg-info-light text-info-text';
      case 'completed':
        return 'bg-success-light text-success-text';
      case 'failed':
        return 'bg-danger-light text-danger-text';
      default:
        return 'bg-surface-alt text-default';
    }
  };

  const getStatusLabel = () => {
    switch (status) {
      case 'uploading':
      case 'converting':
        return '前処理中';
      case 'pending':
        return '未処理';
      case 'processing':
        return '処理中';
      case 'completed':
        return 'OCR 済み';
      case 'failed':
        return '失敗';
      default:
        return '不明';
    }
  };

  return (
    <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${getStatusStyle()}`}>
      {getStatusLabel()}
      {(status === 'uploading' || status === 'converting' || status === 'processing') && (
        <svg className="animate-spin ml-1 h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
    </span>
  );
};

export default StatusBadge;

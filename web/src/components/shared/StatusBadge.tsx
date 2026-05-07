import React from 'react';
import { Loader2 } from 'lucide-react';
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
        <Loader2 size={16} className="animate-spin ml-1" />
      )}
    </span>
  );
};

export default StatusBadge;

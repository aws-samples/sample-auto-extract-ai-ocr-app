import React from 'react';
import { Loader2, PlayCircle } from 'lucide-react';
import { Button } from '../../components/ui';

interface OcrActionBarProps {
  isProcessing: boolean;
  selectedCount: number;
  disabled?: boolean;
  onStartOcr: () => void;
}

const OcrActionBar: React.FC<OcrActionBarProps> = ({
  isProcessing,
  selectedCount,
  disabled = false,
  onStartOcr,
}) => {
  const isDisabled = disabled || isProcessing || selectedCount === 0;

  return (
    <Button
      variant="success"
      size="sm"
      onClick={onStartOcr}
      disabled={isDisabled}
      className="flex items-center"
    >
      {isProcessing ? (
        <Loader2 size={16} className="animate-spin mr-1" />
      ) : (
        <PlayCircle size={16} className="mr-1" />
      )}
      {isProcessing
        ? 'OCR処理を開始中...'
        : selectedCount > 0
          ? `${selectedCount} 件をOCR処理開始`
          : 'OCR処理開始'}
    </Button>
  );
};

export default OcrActionBar;

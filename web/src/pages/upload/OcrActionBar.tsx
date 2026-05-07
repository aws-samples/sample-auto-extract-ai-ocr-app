import React from 'react';
import { Archive, Loader2, PlayCircle } from 'lucide-react';
import { Button } from '../../components/ui';

interface OcrActionBarProps {
  isProcessing: boolean;
  hasPending: boolean;
  hasFiles: boolean;
  onStartOcr: () => void;
}

const OcrActionBar: React.FC<OcrActionBarProps> = ({ isProcessing, hasPending, hasFiles, onStartOcr }) => {
  const isDisabled = isProcessing || !hasPending || !hasFiles;

  return (
    <div className="p-4 border-b border-default flex justify-between items-center">
      <h3 className="text-xl font-medium flex items-center">
        <Archive size={24} className="mr-2 text-success" />
        アップロード済みファイル
      </h3>
      <Button variant="success" onClick={onStartOcr} disabled={isDisabled} className="flex items-center">
        {isProcessing ? (
          <Loader2 size={20} className="animate-spin -ml-1 mr-2" />
        ) : (
          <PlayCircle size={20} className="mr-2" />
        )}
        {isProcessing ? 'OCR処理中...' : 'OCR処理開始'}
      </Button>
    </div>
  );
};

export default OcrActionBar;

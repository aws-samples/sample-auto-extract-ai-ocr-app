import { useState, useEffect } from 'react';
import api from '../utils/api';
import { S3SyncFile, S3ImportResponse } from '../types/app-schema';
import { Alert, Button, Modal } from './ui';

interface S3SyncModalProps {
  isOpen: boolean;
  onClose: () => void;
  appName: string;
  onImportComplete: () => void;
}

const S3SyncModal: React.FC<S3SyncModalProps> = ({ isOpen, onClose, appName, onImportComplete }) => {
  const [files, setFiles] = useState<S3SyncFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFiles, setSelectedFiles] = useState<Set<string>>(new Set());
  const [pageProcessingMode, setPageProcessingMode] = useState<'combined' | 'individual'>('combined');

  // S3ファイル一覧を取得（重複チェック付き）
  const fetchS3Files = async () => {
    if (!appName) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.get(`/s3-sync/${appName}/list`);
      setFiles(response.data.files || []);
    } catch (err: any) {
      console.error('S3ファイル一覧の取得に失敗しました:', err);
      setError(err.response?.data?.detail || 'S3ファイル一覧の取得に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  // チェックボックス操作
  const toggleFileSelection = (fileKey: string) => {
    setSelectedFiles(prev => {
      const newSet = new Set(prev);
      if (newSet.has(fileKey)) {
        newSet.delete(fileKey);
      } else {
        newSet.add(fileKey);
      }
      return newSet;
    });
  };

  // フォルダごとの選択切り替え
  const togglePathSelection = (pathFiles: S3SyncFile[]) => {
    const importablePathFiles = pathFiles.filter(file => !file.is_existing);
    const pathFileKeys = importablePathFiles.map(file => file.key);
    const allPathFilesSelected = pathFileKeys.every(key => selectedFiles.has(key));
    
    const newSelectedFiles = new Set(selectedFiles);
    
    if (allPathFilesSelected) {
      pathFileKeys.forEach(key => newSelectedFiles.delete(key));
    } else {
      pathFileKeys.forEach(key => newSelectedFiles.add(key));
    }
    
    setSelectedFiles(newSelectedFiles);
  };

  // フォルダの選択状態を取得
  const getPathSelectionState = (pathFiles: S3SyncFile[]) => {
    const importablePathFiles = pathFiles.filter(file => !file.is_existing);
    if (importablePathFiles.length === 0) return { checked: false, indeterminate: false };
    
    const pathFileKeys = importablePathFiles.map(file => file.key);
    const selectedCount = pathFileKeys.filter(key => selectedFiles.has(key)).length;
    
    if (selectedCount === 0) return { checked: false, indeterminate: false };
    if (selectedCount === pathFileKeys.length) return { checked: true, indeterminate: false };
    return { checked: false, indeterminate: true };
  };

  // 選択されたファイルをインポート
  const importSelectedFiles = async () => {
    const selectedFileObjects = files.filter(file => selectedFiles.has(file.key) && !file.is_existing);
    if (selectedFileObjects.length === 0) return;
    
    setImporting(true);
    setError(null);
    
    try {
      for (const file of selectedFileObjects) {
        const importData = {
          ...file,
          page_processing_mode: pageProcessingMode
        };
        await api.post<S3ImportResponse>(`/s3-sync/${appName}/import`, importData);
      }
      
      await fetchS3Files();
      setSelectedFiles(new Set());
      onImportComplete();
      
    } catch (err: any) {
      console.error('ファイルのインポートに失敗しました:', err);
      setError(err.response?.data?.detail || 'ファイルのインポートに失敗しました');
    } finally {
      setImporting(false);
    }
  };

  // S3キーからディレクトリパスを抽出
  const extractDirectoryPath = (s3Key: string): string => {
    const parts = s3Key.split('/');
    if (parts.length <= 1) return 'ルート';
    return parts.slice(0, -1).join('/') + '/';
  };

  // パス別グループ化
  const groupFilesByPath = (files: S3SyncFile[]): Record<string, S3SyncFile[]> => {
    const grouped: Record<string, S3SyncFile[]> = {};
    
    files.forEach(file => {
      const path = extractDirectoryPath(file.key);
      if (!grouped[path]) {
        grouped[path] = [];
      }
      grouped[path].push(file);
    });

    Object.keys(grouped).forEach(path => {
      grouped[path].sort((a, b) => a.filename.localeCompare(b.filename));
    });

    return grouped;
  };

  // モーダルが開かれたときにS3ファイル一覧を取得
  useEffect(() => {
    if (isOpen && appName) {
      fetchS3Files();
    }
  }, [isOpen, appName]);

  if (!isOpen) return null;

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="w-full max-w-4xl max-h-[80vh] overflow-hidden">
        <div className="p-4 border-b border-neutral-200 flex justify-between items-center">
          <h2 className="text-xl font-bold">S3ファイル同期</h2>
          <button
            onClick={onClose}
            className="text-neutral-500 hover:text-neutral-700"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-4">
          {error && (
            <Alert type="error" className="mb-4">
              <p>{error}</p>
            </Alert>
          )}

          {/* 処理モード選択 */}
          <div className="mb-4 p-4 bg-neutral-50 rounded-lg">
            <label className="block text-sm font-medium text-neutral-700 mb-2">
              処理モード
            </label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  value="combined"
                  checked={pageProcessingMode === 'combined'}
                  onChange={(e) => setPageProcessingMode(e.target.value as 'combined' | 'individual')}
                  className="mr-2"
                />
                <span className="text-sm">結合モード（全ページを1つのファイルとして処理）</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  value="individual"
                  checked={pageProcessingMode === 'individual'}
                  onChange={(e) => setPageProcessingMode(e.target.value as 'combined' | 'individual')}
                  className="mr-2"
                />
                <span className="text-sm">個別モード（各ページを個別ファイルとして処理）</span>
              </label>
            </div>
          </div>

          <div className="flex justify-between mb-4">
            <Button
              onClick={fetchS3Files}
              disabled={loading}
              variant="primary"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-on-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  更新中...
                </span>
              ) : (
                <span className="flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  更新
                </span>
              )}
            </Button>

            <Button
              onClick={importSelectedFiles}
              disabled={importing || selectedFiles.size === 0}
              variant="success"
            >
              {importing ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-on-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  インポート中...
                </span>
              ) : (
                <span className="flex items-center">
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                  選択ファイルをインポート ({selectedFiles.size})
                </span>
              )}
            </Button>
          </div>

          <div className="overflow-y-auto max-h-[50vh]">
            {loading ? (
              <div className="flex justify-center items-center py-8">
                <svg className="animate-spin h-8 w-8 text-primary" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
            ) : files.length === 0 ? (
              <div className="text-center py-8 text-neutral-500">
                S3バケットにファイルが見つかりませんでした
              </div>
            ) : (
              <div className="space-y-4">
                {Object.entries(groupFilesByPath(files)).map(([path, pathFiles]) => {
                  const selectionState = getPathSelectionState(pathFiles);
                  return (
                    <div key={path} className="border rounded-lg">
                      <div className="bg-neutral-50 px-4 py-2 border-b flex items-center">
                        <input
                          type="checkbox"
                          checked={selectionState.checked}
                          ref={(el) => {
                            if (el) el.indeterminate = selectionState.indeterminate;
                          }}
                          onChange={() => togglePathSelection(pathFiles)}
                          className="mr-2"
                        />
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-neutral-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-5l-2-2H5a2 2 0 00-2 2z" />
                        </svg>
                        <span className="font-medium text-neutral-700">{path}</span>
                        <span className="ml-2 text-sm text-neutral-500">({pathFiles.length} ファイル)</span>
                      </div>
                      <div className="divide-y divide-gray-200">
                        {pathFiles.map((file) => (
                          <div key={file.key} className="px-4 py-3 flex items-center justify-between">
                            <div className="flex items-center">
                              <input
                                type="checkbox"
                                checked={selectedFiles.has(file.key)}
                                onChange={() => toggleFileSelection(file.key)}
                                disabled={file.is_existing}
                                className="mr-3"
                              />
                              <div>
                                <div className="flex items-center">
                                  <span className="text-sm font-medium text-neutral-900">{file.filename}</span>
                                  {file.is_existing && (
                                    <span className="ml-2 px-2 py-1 text-xs bg-surface-alt text-light rounded">
                                      インポート済み
                                    </span>
                                  )}
                                </div>
                                <div className="text-xs text-neutral-500">
                                  {(file.size / 1024).toFixed(1)} KB • {new Date(file.last_modified).toLocaleString()}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>

        <div className="p-4 border-t border-neutral-200 flex justify-end">
          <button
            onClick={onClose}
            className="bg-neutral-300 text-neutral-800 px-4 py-2 rounded hover:bg-neutral-400"
          >
            閉じる
          </button>
        </div>
    </Modal>
  );
};

export default S3SyncModal;

import { useState, useEffect } from 'react';
import api from '../utils/api';
import { S3SyncFile, S3ImportResponse } from '../types/app-schema';

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
  const [importStatus, setImportStatus] = useState<Record<string, string>>({});

  // S3ファイル一覧を取得
  const fetchS3Files = async () => {
    if (!appName) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.post(`/s3-sync/${appName}`);
      setFiles(response.data.files || []);
    } catch (err: any) {
      console.error('S3ファイル一覧の取得に失敗しました:', err);
      setError(err.response?.data?.detail || 'S3ファイル一覧の取得に失敗しました');
    } finally {
      setLoading(false);
    }
  };

  // ファイルをインポート
  const importFile = async (file: S3SyncFile) => {
    if (!appName) return;
    
    setImporting(true);
    setImportStatus(prev => ({ ...prev, [file.key]: 'importing' }));
    
    try {
      const response = await api.post<S3ImportResponse>(`/s3-sync/${appName}/import`, file);
      setImportStatus(prev => ({ ...prev, [file.key]: 'imported' }));
      
      // インポート完了後にコールバックを呼び出す
      onImportComplete();
      
      return response.data;
    } catch (err: any) {
      console.error('ファイルのインポートに失敗しました:', err);
      setImportStatus(prev => ({ ...prev, [file.key]: 'error' }));
      throw err;
    } finally {
      setImporting(false);
    }
  };

  // 全てのファイルをインポート
  const importAllFiles = async () => {
    if (files.length === 0) return;
    
    setImporting(true);
    setError(null);
    
    try {
      // 各ファイルを順番にインポート
      for (const file of files) {
        if (importStatus[file.key] !== 'imported') {
          await importFile(file);
        }
      }
    } catch (err: any) {
      console.error('ファイルのインポートに失敗しました:', err);
      setError(err.response?.data?.detail || 'ファイルのインポートに失敗しました');
    } finally {
      setImporting(false);
    }
  };

  // モーダルが開かれたときにS3ファイル一覧を取得
  useEffect(() => {
    if (isOpen && appName) {
      fetchS3Files();
    }
  }, [isOpen, appName]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-4xl max-h-[80vh] overflow-hidden">
        <div className="p-4 border-b border-gray-200 flex justify-between items-center">
          <h2 className="text-xl font-bold">S3ファイル同期</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="p-4">
          {error && (
            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
              <p>{error}</p>
            </div>
          )}

          <div className="flex justify-between mb-4">
            <button
              onClick={fetchS3Files}
              disabled={loading}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:opacity-50"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
            </button>

            <button
              onClick={importAllFiles}
              disabled={importing || files.length === 0}
              className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 disabled:opacity-50"
            >
              {importing ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
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
                  全てインポート
                </span>
              )}
            </button>
          </div>

          <div className="overflow-y-auto max-h-[50vh]">
            {loading ? (
              <div className="flex justify-center items-center py-8">
                <svg className="animate-spin h-8 w-8 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
              </div>
            ) : files.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                S3バケットにファイルが見つかりませんでした
              </div>
            ) : (
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      ファイル名
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      サイズ
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      更新日時
                    </th>
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      ステータス
                    </th>
                    <th scope="col" className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                      アクション
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {files.map((file) => (
                    <tr key={file.key}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {file.filename}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {new Date(file.last_modified).toLocaleString()}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        {importStatus[file.key] === 'importing' ? (
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">
                            インポート中
                          </span>
                        ) : importStatus[file.key] === 'imported' ? (
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                            インポート済み
                          </span>
                        ) : importStatus[file.key] === 'error' ? (
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                            エラー
                          </span>
                        ) : (
                          <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-gray-100 text-gray-800">
                            未インポート
                          </span>
                        )}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                        <button
                          onClick={() => importFile(file)}
                          disabled={importing || importStatus[file.key] === 'importing' || importStatus[file.key] === 'imported'}
                          className="text-indigo-600 hover:text-indigo-900 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          インポート
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        <div className="p-4 border-t border-gray-200 flex justify-end">
          <button
            onClick={onClose}
            className="bg-gray-300 text-gray-800 px-4 py-2 rounded hover:bg-gray-400"
          >
            閉じる
          </button>
        </div>
      </div>
    </div>
  );
};

export default S3SyncModal;

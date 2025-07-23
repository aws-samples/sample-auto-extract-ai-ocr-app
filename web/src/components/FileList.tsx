import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ImageFile } from '../types/ocr';
import StatusBadge from './StatusBadge';

interface FileListProps {
  files: ImageFile[];
  onRefresh: () => void;
}

const FileList: React.FC<FileListProps> = ({ files, onRefresh }) => {
  const navigate = useNavigate();

  // 日付のフォーマット
  const formatDate = (dateString: string) => {
    if (!dateString) return '';
    const date = new Date(dateString);
    return date.toLocaleString('ja-JP');
  };

  // 結果表示ボタンのクリックハンドラ
  const handleViewResult = (id: string) => {
    navigate(`/ocr-result/${id}`);
  };

  return (
    <div className="p-4">
      {files.length > 0 ? (
        <>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-500">全{files.length}件</span>
            <div className="flex items-center">
              <button onClick={onRefresh} className="text-blue-500 hover:text-blue-700 mr-2 flex items-center text-sm">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                更新
              </button>
            </div>
          </div>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ファイル名</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">アップロード日時</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ステータス</th>
                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">操作</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {files.map((file) => (
                  <tr key={file.id}>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <div className="flex items-center">
                        <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 mr-2 ${file.name.toLowerCase().endsWith('.pdf') ? 'text-red-500' : 'text-blue-500'}`} viewBox="0 0 20 20" fill="currentColor">
                          {file.name.toLowerCase().endsWith('.pdf') ? (
                            <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                          ) : (
                            <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                          )}
                        </svg>
                        <span className="truncate max-w-xs text-sm">{file.name}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{formatDate(file.uploadTime)}</td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <StatusBadge status={file.status} />
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      {file.status === 'completed' ? (
                        <button 
                          onClick={() => handleViewResult(file.id)} 
                          className="text-blue-600 hover:text-blue-900"
                        >
                          結果表示
                        </button>
                      ) : (
                        <span className="text-gray-400">処理待ち</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="bg-white rounded-lg p-6 border border-dashed border-gray-300 flex flex-col items-center justify-center text-gray-400">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-12 w-12 mb-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <p className="text-center">
            ファイルがありません。PDFをアップロードしてください。
          </p>
        </div>
      )}
    </div>
  );
};

export default FileList;

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ImageFile } from '../types/ocr';
import StatusBadge from './StatusBadge';
import { formatDateTimeJST } from '../utils/dateUtils';

interface FileListProps {
  files: ImageFile[];
  onRefresh: () => void;
}

interface GroupedFiles {
  parentDocuments: ImageFile[];
  childPages: { [parentId: string]: ImageFile[] };
  standaloneFiles: ImageFile[];
}

const FileList: React.FC<FileListProps> = ({ files, onRefresh }) => {
  const navigate = useNavigate();
  const [expandedParents, setExpandedParents] = useState<Set<string>>(new Set());

  // 結果表示ボタンのクリックハンドラ
  const handleViewResult = (id: string) => {
    navigate(`/ocr-result/${id}`);
  };

  // 親ドキュメントの展開/折りたたみ
  const toggleParentExpansion = (parentId: string) => {
    const newExpanded = new Set(expandedParents);
    if (newExpanded.has(parentId)) {
      newExpanded.delete(parentId);
    } else {
      newExpanded.add(parentId);
    }
    setExpandedParents(newExpanded);
  };

  // ファイルをグループ化
  const groupFiles = (files: ImageFile[]): GroupedFiles => {
    // まず全体を時間順にソート（新しい順）
    const sortedFiles = [...files].sort((a, b) => {
      const timeA = new Date(a.uploadTime || 0).getTime();
      const timeB = new Date(b.uploadTime || 0).getTime();
      return timeB - timeA; // 降順（新しい順）
    });

    const parentDocuments: ImageFile[] = [];
    const childPages: { [parentId: string]: ImageFile[] } = {};
    const standaloneFiles: ImageFile[] = [];

    sortedFiles.forEach(file => {
      if (file.pageProcessingMode === 'individual' && !file.parentDocumentId && (file.totalPages || 0) > 1) {
        // 親ドキュメント（2ページ以上の個別処理のみ）
        parentDocuments.push(file);
      } else if (file.parentDocumentId) {
        // 子ページ
        if (!childPages[file.parentDocumentId]) {
          childPages[file.parentDocumentId] = [];
        }
        childPages[file.parentDocumentId].push(file);
      } else {
        // 通常ファイル（統合処理、既存データ、1ページの個別処理）
        standaloneFiles.push(file);
      }
    });

    // 子ページをページ番号順にソート
    Object.keys(childPages).forEach(parentId => {
      childPages[parentId].sort((a, b) => (a.pageNumber || 0) - (b.pageNumber || 0));
    });

    return { parentDocuments, childPages, standaloneFiles };
  };

  const groupedFiles = groupFiles(files);
  const totalFiles = files.length;

  // 親ドキュメントの進捗状況を計算
  const getParentProgress = (parentId: string) => {
    const children = groupedFiles.childPages[parentId] || [];
    const completed = children.filter(child => child.status === 'completed').length;
    const total = children.length;
    return { completed, total };
  };

  // 親ドキュメントの全体ステータスを取得
  const getParentOverallStatus = (parentId: string) => {
    const children = groupedFiles.childPages[parentId] || [];
    if (children.length === 0) return 'pending';
    
    const statuses = children.map(child => child.status);
    if (statuses.every(status => status === 'completed')) return 'completed';
    if (statuses.some(status => status === 'failed')) return 'failed';
    if (statuses.some(status => status === 'processing')) return 'processing';
    return 'pending';
  };

  return (
    <div className="p-4">
      {totalFiles > 0 ? (
        <>
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-500">全{totalFiles}件</span>
            <div className="flex items-center">
              <button onClick={onRefresh} className="text-blue-500 hover:text-blue-700 mr-2 flex items-center text-sm">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                更新
              </button>
            </div>
          </div>
          
          <div className="space-y-2">
            {/* 親ドキュメント（個別処理） */}
            {groupedFiles.parentDocuments.map((parentFile) => {
              const isExpanded = expandedParents.has(parentFile.id);
              const children = groupedFiles.childPages[parentFile.id] || [];
              const progress = getParentProgress(parentFile.id);
              const overallStatus = getParentOverallStatus(parentFile.id);
              
              return (
                <div key={parentFile.id} className="border border-gray-200 rounded-lg">
                  {/* 親ドキュメント行 */}
                  <div 
                    className="flex items-center p-4 cursor-pointer hover:bg-gray-50"
                    onClick={() => toggleParentExpansion(parentFile.id)}
                  >
                    <div className="flex items-center flex-1">
                      {/* 展開/折りたたみアイコン */}
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        className={`h-4 w-4 mr-2 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                      
                      {/* ファイルアイコン */}
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2 text-red-500" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                      </svg>
                      
                      {/* ファイル名と情報 */}
                      <div className="flex-1">
                        <div className="font-medium text-gray-900">{parentFile.name}</div>
                        <div className="text-sm text-gray-500">
                          個別処理 - {parentFile.totalPages}ページ ({progress.completed}/{progress.total} 完了)
                        </div>
                      </div>
                    </div>
                    
                    {/* アップロード日時 */}
                    <div className="text-sm text-gray-500 mr-4">
                      {formatDateTimeJST(parentFile.uploadTime)}
                    </div>
                    
                    {/* 全体ステータス */}
                    <StatusBadge status={overallStatus} />
                  </div>
                  
                  {/* 子ページ一覧 */}
                  {isExpanded && children.length > 0 && (
                    <div className="border-t border-gray-100">
                      {children.map((childFile) => (
                        <div key={childFile.id} className="flex items-center p-4 pl-12 hover:bg-gray-50">
                          {/* ページアイコン */}
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-blue-500" viewBox="0 0 20 20" fill="currentColor">
                            <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                          </svg>
                          
                          {/* ページ情報 */}
                          <div className="flex-1">
                            <div className="text-sm font-medium text-gray-700">
                              {childFile.name} (ページ {childFile.pageNumber}/{childFile.totalPages})
                            </div>
                          </div>
                          
                          {/* アップロード日時 */}
                          <div className="text-sm text-gray-500 mr-4">
                            {formatDateTimeJST(childFile.uploadTime)}
                          </div>
                          
                          {/* ステータス */}
                          <div className="mr-4">
                            <StatusBadge status={childFile.status} />
                          </div>
                          
                          {/* 操作ボタン */}
                          <div className="text-sm">
                            {childFile.status === 'completed' ? (
                              <button 
                                onClick={() => handleViewResult(childFile.id)} 
                                className="text-blue-600 hover:text-blue-900"
                              >
                                結果表示
                              </button>
                            ) : (
                              <span className="text-gray-400">処理待ち</span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
            
            {/* 通常ファイル（統合処理・既存データ） */}
            {groupedFiles.standaloneFiles.map((file) => (
              <div key={file.id} className="border border-gray-200 rounded-lg p-4">
                <div className="flex items-center">
                  {/* ファイルアイコン */}
                  <svg xmlns="http://www.w3.org/2000/svg" className={`h-5 w-5 mr-2 ${file.name.toLowerCase().endsWith('.pdf') ? 'text-red-500' : 'text-blue-500'}`} viewBox="0 0 20 20" fill="currentColor">
                    {file.name.toLowerCase().endsWith('.pdf') ? (
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                    ) : (
                      <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
                    )}
                  </svg>
                  
                  {/* ファイル名と処理情報 */}
                  <div className="flex-1">
                    <div className="font-medium text-gray-900">{file.name}</div>
                    <div className="text-sm text-gray-500">
                      {file.pageProcessingMode === 'combined' ? (
                        <span>
                          統合処理
                          {file.totalPages && file.totalPages > 1 && ` - ${file.totalPages}ページ`}
                        </span>
                      ) : file.pageProcessingMode === 'individual' && file.totalPages === 1 ? (
                        <span>1ページ</span>
                      ) : (
                        <span>-</span>
                      )}
                    </div>
                  </div>
                  
                  {/* アップロード日時 */}
                  <div className="text-sm text-gray-500 mr-4">
                    {formatDateTimeJST(file.uploadTime)}
                  </div>
                  
                  {/* ステータス */}
                  <div className="mr-4">
                    <StatusBadge status={file.status} />
                  </div>
                  
                  {/* 操作ボタン */}
                  <div className="text-sm">
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
                  </div>
                </div>
              </div>
            ))}
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

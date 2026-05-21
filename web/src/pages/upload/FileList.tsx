import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw, ChevronRight, FileText, Image, Trash2, CheckCircle, Upload } from 'lucide-react';
import { ImageFile } from '../../types/ocr';
import StatusBadge from '../../components/shared/StatusBadge';
import { formatDateTimeJST } from '../../utils/dateUtils';
import { deleteImage } from '../../services/imageApi';
import Toast from '../../components/ui/Toast';
import { Modal, Button, EmptyState } from '../../components/ui';

interface FileListProps {
  files: ImageFile[];
  onRefresh: () => void;
}

interface GroupedFiles {
  parentDocuments: ImageFile[];
  childPages: { [parentId: string]: ImageFile[] };
  standaloneFiles: ImageFile[];
}

type SortField = 'uploadTime' | 'status' | 'name';

const FileList: React.FC<FileListProps> = ({ files, onRefresh }) => {
  const navigate = useNavigate();
  const [expandedParents, setExpandedParents] = useState<Set<string>>(new Set());
  const [deleteConfirm, setDeleteConfirm] = useState<{ show: boolean; imageId: string; imageName: string }>({ 
    show: false, 
    imageId: '', 
    imageName: '' 
  });
  const [toast, setToast] = useState<{ show: boolean; message: string; type: 'success' | 'error' }>({ 
    show: false, 
    message: '', 
    type: 'success' 
  });
  const [deleting, setDeleting] = useState(false);
  const sortField: SortField = 'uploadTime';

  // 親ドキュメントをデフォルトで開く
  React.useEffect(() => {
    const grouped = groupFiles(files);
    const parentIds = grouped.parentDocuments.map(p => p.id);
    setExpandedParents(new Set(parentIds));
  }, [files]);

  const sortFiles = (fileList: ImageFile[]) => {
    return [...fileList].sort((a, b) => {
      let aValue: any = a[sortField];
      let bValue: any = b[sortField];

      // 値が存在しない場合の処理
      if (!aValue) aValue = '';
      if (!bValue) bValue = '';

      // 文字列比較（降順）
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return -aValue.localeCompare(bValue);
      }

      // 数値比較（降順）
      if (aValue < bValue) return 1;
      if (aValue > bValue) return -1;
      return 0;
    });
  };

  // 結果表示ボタンのクリックハンドラ
  const handleViewResult = (id: string) => {
    navigate(`/ocr-result/${id}`);
  };

  // 削除確認ダイアログを表示
  const handleDeleteClick = (imageId: string, imageName: string) => {
    setDeleteConfirm({ show: true, imageId, imageName });
  };

  // 削除実行
  const handleDeleteConfirm = async () => {
    setDeleting(true);
    try {
      await deleteImage(deleteConfirm.imageId);
      setToast({ show: true, message: '画像を削除しました', type: 'success' });
      setDeleteConfirm({ show: false, imageId: '', imageName: '' });
      onRefresh();
    } catch (error) {
      setToast({ show: true, message: '削除に失敗しました', type: 'error' });
    } finally {
      setDeleting(false);
    }
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
    // ソート適用
    const sortedFiles = sortFiles(files);

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

  // 表示用に全ファイルを統合（親ファイルと通常ファイルを混在させる）
  const getMergedFilesForDisplay = () => {
    const grouped = groupFiles(files);
    const merged: Array<{ type: 'parent' | 'standalone', file: ImageFile }> = [];
    
    // 親ファイルと通常ファイルを統合
    [...grouped.parentDocuments, ...grouped.standaloneFiles].forEach(file => {
      if (file.pageProcessingMode === 'individual' && !file.parentDocumentId && (file.totalPages || 0) > 1) {
        merged.push({ type: 'parent', file });
      } else {
        merged.push({ type: 'standalone', file });
      }
    });
    
    // ユーザー選択のソートフィールドでソート
    merged.sort((a, b) => {
      let aValue: any = a.file[sortField];
      let bValue: any = b.file[sortField];

      // 値が存在しない場合の処理
      if (!aValue) aValue = '';
      if (!bValue) bValue = '';

      // 文字列比較（降順）
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return -aValue.localeCompare(bValue);
      }

      // 数値比較（降順）
      if (aValue < bValue) return 1;
      if (aValue > bValue) return -1;
      return 0;
    });
    
    return { merged, childPages: grouped.childPages };
  };

  const { merged: mergedFiles, childPages } = getMergedFilesForDisplay();
  const totalFiles = files.length;

  // 親ドキュメントの進捗状況を計算
  const getParentProgress = (parentId: string) => {
    const children = childPages[parentId] || [];
    const completed = children.filter(child => child.status === 'completed').length;
    const total = children.length;
    return { completed, total };
  };

  // 親ドキュメントの全体ステータスを取得
  const getParentOverallStatus = (parentId: string) => {
    const children = childPages[parentId] || [];
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
            <span className="text-sm text-neutral-500">全{totalFiles}件</span>
            <div className="flex items-center">
              <Button variant="ghost" size="sm" onClick={onRefresh} className="flex items-center">
                <RefreshCw size={16} className="mr-1" />
                更新
              </Button>
            </div>
          </div>
          
          <div className="space-y-2">
            {/* 親ドキュメントと通常ファイルを統合して表示 */}
            {mergedFiles.map(({ type, file }) => {
              if (type === 'parent') {
                // 親ドキュメント（個別処理）
                const isExpanded = expandedParents.has(file.id);
                const children = childPages[file.id] || [];
                const progress = getParentProgress(file.id);
                const overallStatus = getParentOverallStatus(file.id);
                
                return (
                  <div key={file.id} className="rounded-xl border border-default shadow-sm">
                    {/* 親ドキュメント行 */}
                    <div 
                      className="flex items-center p-4 cursor-pointer hover:bg-neutral-50 transition-colors"
                      onClick={() => toggleParentExpansion(file.id)}
                    >
                      {/* アイコンエリア: 固定幅 */}
                      <div className="w-12 flex-shrink-0 flex items-center">
                        {/* 展開/折りたたみアイコン */}
                        <ChevronRight size={16} className={`mr-1 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                        
                        {/* ファイルアイコン */}
                        <FileText size={20} className="text-danger" />
                      </div>
                      
                      {/* ファイル名と情報 */}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-neutral-900">{file.name}</div>
                        <div className="text-sm text-neutral-500">
                          個別処理 - {file.totalPages}ページ ({progress.completed}/{progress.total} 完了)
                        </div>
                      </div>
                    
                    {/* アップロード日時 */}
                    <div className="w-40 flex-shrink-0 text-sm text-neutral-500">
                      {formatDateTimeJST(file.uploadTime)}
                    </div>
                    
                    {/* 全体ステータス */}
                    <div className="w-24 flex-shrink-0">
                      <StatusBadge status={overallStatus} />
                    </div>
                    
                    {/* 確認済み（親は表示しない） */}
                    <div className="w-16 flex-shrink-0 flex justify-center">
                      <span className="text-neutral-300">-</span>
                    </div>
                    
                    {/* 操作ボタン（空白でスペース確保） */}
                    <div className="text-sm w-20 flex-shrink-0">
                      <span className="text-neutral-400">-</span>
                    </div>
                    
                    {/* 削除ボタン */}
                    <div className="w-8 flex-shrink-0 flex justify-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteClick(file.id, file.name);
                        }}
                        className="text-neutral-400 hover:text-neutral-600"
                        title="削除（全ページ削除）"
                      >
                        <Trash2 size={20} />
                      </button>
                    </div>
                  </div>
                  
                  {/* 子ページ一覧 */}
                  {isExpanded && children.length > 0 && (
                    <div className="border-t border-default">
                      {children.map((childFile) => (
                        <div key={childFile.id} className="flex items-center p-4 pl-12 hover:bg-neutral-50">
                          {/* ページアイコン */}
                          <Image size={16} className="mr-2 text-primary" />
                          
                          {/* ページ情報 */}
                          <div className="flex-1 min-w-0">
                            <div className="text-sm font-medium text-neutral-700">
                              {childFile.name} (ページ {childFile.pageNumber}/{childFile.totalPages})
                            </div>
                          </div>
                          
                          {/* アップロード日時 */}
                          <div className="w-40 flex-shrink-0 text-sm text-neutral-500">
                            {formatDateTimeJST(childFile.uploadTime)}
                          </div>
                          
                          {/* ステータス */}
                          <div className="w-24 flex-shrink-0">
                            <StatusBadge status={childFile.status} />
                          </div>
                          
                          {/* 確認済み */}
                          <div className="w-16 flex-shrink-0 flex justify-center">
                            {childFile.verificationCompleted ? (
                              <CheckCircle size={20} className="text-success" />
                            ) : (
                              <span className="text-neutral-300">-</span>
                            )}
                          </div>
                          
                          {/* 操作ボタン */}
                          <div className="text-sm w-20 flex-shrink-0">
                            {childFile.status === 'completed' ? (
                              <button 
                                onClick={() => handleViewResult(childFile.id)} 
                                className="text-info hover:text-info-hover"
                              >
                                結果表示
                              </button>
                            ) : (
                              <span className="text-neutral-400">処理待ち</span>
                            )}
                          </div>
                          
                          {/* 削除ボタン（子ページは削除不可） */}
                          <div className="w-8 flex-shrink-0"></div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
              } else {
                // 通常ファイル（統合処理・既存データ）
                return (
                  <div key={file.id} className="rounded-xl border border-default shadow-sm p-4">
                    <div className="flex items-center">
                      {/* アイコンエリア: 固定幅 */}
                      <div className="w-12 flex-shrink-0 flex items-center justify-center">
                        {/* ファイルアイコン */}
                        {file.name.toLowerCase().endsWith('.pdf') ? (
                          <FileText size={20} className="text-danger" />
                        ) : (
                          <Image size={20} className="text-primary" />
                        )}
                      </div>
                      
                      {/* ファイル名と処理情報 */}
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-neutral-900">{file.name}</div>
                        <div className="text-sm text-neutral-500">
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
                      <div className="w-40 flex-shrink-0 text-sm text-neutral-500">
                        {formatDateTimeJST(file.uploadTime)}
                      </div>
                      
                      {/* ステータス */}
                      <div className="w-24 flex-shrink-0">
                        <StatusBadge status={file.status} />
                        {file.agent_status === 'processing' && (
                          <span className="mt-1 px-1.5 py-0.5 text-xs rounded bg-blue-50 text-blue-700 block text-center">
                            検証中
                          </span>
                        )}
                      </div>
                      
                      {/* 確認済み */}
                      <div className="w-16 flex-shrink-0 flex justify-center">
                        {file.verificationCompleted ? (
                          <CheckCircle size={20} className="text-success" />
                        ) : (
                          <span className="text-neutral-300">-</span>
                        )}
                      </div>
                      
                      {/* 操作ボタン */}
                      <div className="text-sm w-20 flex-shrink-0">
                        {file.status === 'completed' ? (
                          <button 
                            onClick={() => handleViewResult(file.id)} 
                            className="text-info hover:text-info-hover"
                          >
                            結果表示
                          </button>
                        ) : (
                          <span className="text-neutral-400">処理待ち</span>
                        )}
                      </div>
                      
                      {/* 削除ボタン */}
                      <div className="w-8 flex-shrink-0 flex justify-center">
                        <button
                          onClick={() => handleDeleteClick(file.id, file.name)}
                          className="text-neutral-400 hover:text-neutral-600"
                          title="削除"
                        >
                          <Trash2 size={20} />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              }
            })}
          </div>
        </>
      ) : (
        <EmptyState icon={Upload} message="ファイルがありません。PDFをアップロードしてください。" />
      )}

      {/* 削除確認モーダル */}
      <Modal isOpen={deleteConfirm.show} onClose={() => setDeleteConfirm({ show: false, imageId: '', imageName: '' })} className="max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold mb-4">画像の削除</h3>
            <p className="text-muted mb-6">
              「{deleteConfirm.imageName}」を削除します。この操作は取り消せません。
            </p>
            <div className="flex justify-end gap-3">
              <Button
                variant="secondary"
                size="sm"
                onClick={() => setDeleteConfirm({ show: false, imageId: '', imageName: '' })}
                disabled={deleting}
              >
                キャンセル
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={handleDeleteConfirm}
                disabled={deleting}
              >
                {deleting ? '削除中...' : '削除'}
              </Button>
            </div>
      </Modal>

      {/* Toast通知 */}
      <Toast
        show={toast.show}
        message={toast.message}
        type={toast.type}
        onClose={() => setToast({ ...toast, show: false })}
      />
    </div>
  );
};

export default FileList;

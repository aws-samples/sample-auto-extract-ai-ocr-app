import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { RefreshCw, ChevronRight, FileText, Image, Trash2, Upload, CheckCircle } from 'lucide-react';
import { ImageFile } from '../../types/ocr';
import ProcessStatusBadge from '../../components/shared/ProcessStatusBadge';
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
  const collapsedByUser = React.useRef<Set<string>>(new Set());
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
  const [filter, setFilter] = useState<'all' | 'action_needed' | 'confirmed' | 'processing' | 'failed'>('all');
  const sortField: SortField = 'uploadTime';

  // フィルタ判定ヘルパー
  const getConfirmationState = (file: ImageFile) => {
    if (file.status === 'failed' || file.agentStatus === 'failed') return 'failed';
    if (file.status !== 'completed') return 'processing';
    if (file.agentStatus === 'processing') return 'processing';
    if (file.verificationCompleted) return 'confirmed';
    return 'action_needed';
  };

  // フィルタ適用
  const filteredFiles = useMemo(() => {
    if (filter === 'all') return files;
    return files.filter(file => {
      const state = getConfirmationState(file);
      return state === filter;
    });
  }, [files, filter]);

  // 新しく追加された親ドキュメントのみ展開（既存の展開状態は保持）
  React.useEffect(() => {
    const grouped = groupFiles(filteredFiles);
    const parentIds = grouped.parentDocuments.map(p => p.id);
    setExpandedParents(prev => {
      const next = new Set(prev);
      for (const id of parentIds) {
        if (!prev.has(id) && !collapsedByUser.current.has(id)) {
          next.add(id);
        }
      }
      return next;
    });
  }, [filteredFiles]);

  const sortFiles = (fileList: ImageFile[]) => {
    return [...fileList].sort((a, b) => {
      let aValue: any = a[sortField];
      let bValue: any = b[sortField];
      if (!aValue) aValue = '';
      if (!bValue) bValue = '';
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return -aValue.localeCompare(bValue);
      }
      if (aValue < bValue) return 1;
      if (aValue > bValue) return -1;
      return 0;
    });
  };

  // 行クリックで結果画面へ遷移
  const handleRowClick = (file: ImageFile) => {
    if (file.status === 'completed') {
      navigate(`/ocr-result/${file.id}`);
    }
  };

  // 削除確認ダイアログを表示
  const handleDeleteClick = (e: React.MouseEvent, imageId: string, imageName: string) => {
    e.stopPropagation();
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
      collapsedByUser.current.add(parentId);
    } else {
      newExpanded.add(parentId);
      collapsedByUser.current.delete(parentId);
    }
    setExpandedParents(newExpanded);
  };

  // ファイルをグループ化
  const groupFiles = (files: ImageFile[]): GroupedFiles => {
    const sortedFiles = sortFiles(files);
    const parentDocuments: ImageFile[] = [];
    const childPages: { [parentId: string]: ImageFile[] } = {};
    const standaloneFiles: ImageFile[] = [];

    sortedFiles.forEach(file => {
      if (file.pageProcessingMode === 'individual' && !file.parentDocumentId && (file.totalPages || 0) > 1) {
        parentDocuments.push(file);
      } else if (file.parentDocumentId) {
        if (!childPages[file.parentDocumentId]) {
          childPages[file.parentDocumentId] = [];
        }
        childPages[file.parentDocumentId].push(file);
      } else {
        standaloneFiles.push(file);
      }
    });

    Object.keys(childPages).forEach(parentId => {
      childPages[parentId].sort((a, b) => (a.pageNumber || 0) - (b.pageNumber || 0));
    });

    return { parentDocuments, childPages, standaloneFiles };
  };

  // 表示用に全ファイルを統合
  const getMergedFilesForDisplay = () => {
    const grouped = groupFiles(filteredFiles);
    const merged: Array<{ type: 'parent' | 'standalone', file: ImageFile }> = [];

    [...grouped.parentDocuments, ...grouped.standaloneFiles].forEach(file => {
      if (file.pageProcessingMode === 'individual' && !file.parentDocumentId && (file.totalPages || 0) > 1) {
        merged.push({ type: 'parent', file });
      } else {
        merged.push({ type: 'standalone', file });
      }
    });

    merged.sort((a, b) => {
      let aValue: any = a.file[sortField];
      let bValue: any = b.file[sortField];
      if (!aValue) aValue = '';
      if (!bValue) bValue = '';
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return -aValue.localeCompare(bValue);
      }
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
            <div className="flex items-center gap-2">
              <span className="text-sm text-neutral-500">全{totalFiles}件</span>
              <select
                value={filter}
                onChange={(e) => setFilter(e.target.value as typeof filter)}
                className="text-sm border border-default rounded px-2 py-1"
              >
                <option value="all">全て</option>
                <option value="action_needed">要対応</option>
                <option value="confirmed">確認済み</option>
                <option value="processing">処理中</option>
                <option value="failed">失敗</option>
              </select>
            </div>
            <div className="flex items-center">
              <Button variant="ghost" size="sm" onClick={onRefresh} className="flex items-center">
                <RefreshCw size={16} className="mr-1" />
                更新
              </Button>
            </div>
          </div>

          {/* テーブルヘッダー */}
          <div className="rounded-xl p-4 mb-1">
            <div className="flex items-center text-xs font-medium text-neutral-500">
              <div className="w-12 flex-shrink-0"></div>
              <div className="flex-1 min-w-0">ファイル名</div>
              <div className="w-40 flex-shrink-0">アップロード</div>
              <div className="w-28 flex-shrink-0 text-center">処理状態</div>
              <div className="w-16 flex-shrink-0 text-center">確認</div>
            </div>
          </div>

          <div className="space-y-2">
            {mergedFiles.map(({ type, file }) => {
              if (type === 'parent') {
                // 親ドキュメント（個別処理）
                const isExpanded = expandedParents.has(file.id);
                const children = childPages[file.id] || [];
                const progress = getParentProgress(file.id);
                const overallStatus = getParentOverallStatus(file.id);

                return (
                  <div key={file.id} className="rounded-xl border border-default shadow-sm">
                    {/* 親ドキュメント行 — クリックで展開 */}
                    <div
                      className="group relative flex items-center p-4 cursor-pointer hover:bg-neutral-50 transition-colors"
                      onClick={() => toggleParentExpansion(file.id)}
                    >
                      <div className="w-12 flex-shrink-0 flex items-center">
                        <ChevronRight size={16} className={`mr-1 transform transition-transform ${isExpanded ? 'rotate-90' : ''}`} />
                        <FileText size={20} className="text-danger" />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-neutral-900">{file.name}</div>
                        <div className="text-sm text-neutral-500">
                          個別処理 - {file.totalPages}ページ ({progress.completed}/{progress.total} 完了)
                        </div>
                      </div>

                      <div className="w-40 flex-shrink-0 text-sm text-neutral-500">
                        {formatDateTimeJST(file.uploadTime)}
                      </div>

                      <div className="w-32 flex-shrink-0 flex justify-center">
                        <ProcessStatusBadge status={overallStatus} />
                      </div>

                      <div className="w-16 flex-shrink-0 flex justify-center">
                        <span className="text-neutral-300">-</span>
                      </div>

                      {/* hover 時削除ボタン */}
                      <button
                        onClick={(e) => handleDeleteClick(e, file.id, file.name)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity text-neutral-400 hover:text-danger p-1"
                        title="削除"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>

                    {/* 子ページ一覧 */}
                    {isExpanded && children.length > 0 && (
                      <div className="border-t border-default">
                        {children.map((childFile) => (
                          <div
                            key={childFile.id}
                            className={`group relative flex items-center p-4 pl-12 hover:bg-neutral-50 transition-colors ${childFile.status === 'completed' ? 'cursor-pointer' : ''}`}
                            onClick={() => handleRowClick(childFile)}
                          >
                            <Image size={16} className="mr-2 text-primary" />

                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-neutral-700">
                                {childFile.name} (ページ {childFile.pageNumber}/{childFile.totalPages})
                              </div>
                            </div>

                            <div className="w-40 flex-shrink-0 text-sm text-neutral-500">
                              {formatDateTimeJST(childFile.uploadTime)}
                            </div>

                            <div className="w-32 flex-shrink-0 flex justify-center">
                              <ProcessStatusBadge
                                status={childFile.status}
                                agentStatus={childFile.agentStatus}
                              />
                            </div>

                            <div className="w-16 flex-shrink-0 flex justify-center">
                              {childFile.verificationCompleted ? (
                                <CheckCircle size={16} className="text-success" />
                              ) : (childFile.agentSuggestionsCount ?? 0) > 0 ? (
                                <span className="inline-flex items-center justify-center min-w-5 h-5 px-1.5 text-[10px] font-bold text-white bg-warning rounded-full">
                                  {childFile.agentSuggestionsCount}
                                </span>
                              ) : (
                                <span className="text-neutral-300">-</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                );
              } else {
                // 通常ファイル（統合処理・既存データ）
                return (
                  <div
                    key={file.id}
                    className={`group relative rounded-xl border border-default shadow-sm p-4 hover:bg-neutral-50 transition-colors ${file.status === 'completed' ? 'cursor-pointer' : ''}`}
                    onClick={() => handleRowClick(file)}
                  >
                    <div className="flex items-center">
                      <div className="w-12 flex-shrink-0 flex items-center justify-center">
                        {file.name.toLowerCase().endsWith('.pdf') ? (
                          <FileText size={20} className="text-danger" />
                        ) : (
                          <Image size={20} className="text-primary" />
                        )}
                      </div>

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

                      <div className="w-40 flex-shrink-0 text-sm text-neutral-500">
                        {formatDateTimeJST(file.uploadTime)}
                      </div>

                      <div className="w-32 flex-shrink-0 flex justify-center">
                        <ProcessStatusBadge
                          status={file.status}
                          agentStatus={file.agentStatus}
                        />
                      </div>

                      <div className="w-16 flex-shrink-0 flex justify-center">
                        {file.verificationCompleted ? (
                          <CheckCircle size={16} className="text-success" />
                        ) : (file.agentSuggestionsCount ?? 0) > 0 ? (
                          <span className="inline-flex items-center justify-center min-w-5 h-5 px-1.5 text-[10px] font-bold text-white bg-warning rounded-full">
                            {file.agentSuggestionsCount}
                          </span>
                        ) : (
                          <span className="text-neutral-300">-</span>
                        )}
                      </div>

                      {/* hover 時削除ボタン */}
                      <button
                        onClick={(e) => handleDeleteClick(e, file.id, file.name)}
                        className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity text-neutral-400 hover:text-danger p-1"
                        title="削除"
                      >
                        <Trash2 size={14} />
                      </button>
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

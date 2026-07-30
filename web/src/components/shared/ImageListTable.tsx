import { useState, useMemo } from 'react';
import { CheckCircle, ChevronRight, FileText, Image as ImageIcon } from 'lucide-react';
import type { ImageFile } from '../../types/ocr';
import { Table, Thead, Tbody, EmptyState, Tooltip } from '../ui';
import ProcessStatusBadge from './ProcessStatusBadge';
import PresenceBadge, { PresenceViewer } from './PresenceBadge';
import { formatDateJST, formatDateTimeJST } from '../../utils/dateUtils';
import {
  groupFiles,
  getTopLevelFiles,
  getParentProgress,
  getParentOverallStatus,
  canOpenImage,
  toggleImageSelection,
} from '../../utils/imageListHelpers';

export interface ImageListTableProps {
  /** フィルタ済みの Image family 全体（親子分類は内部で行う） */
  files: ImageFile[];
  /** ページング後に表示する親 / standalone の ID。省略時は全件表示 */
  visibleTopLevelIds?: ReadonlySet<string>;
  /** 行クリック時のコールバック。canOpenImage が false の行では発火しない */
  onRowClick?: (file: ImageFile) => void;
  /** ユースケース列を表示するか（デフォルト true）。app 画面では false */
  showUsecaseColumn?: boolean;
  /** チェックボックス列と全選択を表示するか（デフォルト false）。Upload 画面のみ true */
  showSelection?: boolean;
  /** 選択中の Image ID セット（showSelection=true のとき使用） */
  selectedIds?: Set<string>;
  /** 選択変更コールバック */
  onSelectionChange?: (ids: Set<string>) => void;
  /** image_id → 現在の視聴者一覧 */
  presenceByImageId?: Record<string, PresenceViewer[]>;
  /** appName → display_name のマップ（ユースケース列の表示用） */
  appDisplayNameMap?: Record<string, string>;
  /** 0 件時に表示するメッセージ（デフォルト: "ファイルがありません"） */
  emptyMessage?: string;
}

/**
 * Image 一覧テーブル（Upload / History / Admin 共通）。
 *
 * - 6 列（+ 選択列オプション）: [チェック?] ファイル名 / ユースケース? / アップロード者 / 日時 / ステータス / 確認
 * - PDF 親子ドキュメントは親行 + 子ページ（展開時のみ表示）のツリー構造。初期状態は全折りたたみ
 * - ページングは親 / standalone 単位で行い、子ページは親と同じページに保つ
 * - `canOpenImage(file) === true`（status === 'completed'）の行のみ行クリックで onRowClick が発火
 * - 0 件時も Thead は残し、tbody に EmptyState 行を表示する
 */
export default function ImageListTable({
  files,
  visibleTopLevelIds,
  onRowClick,
  showUsecaseColumn = true,
  showSelection = false,
  selectedIds,
  onSelectionChange,
  presenceByImageId,
  appDisplayNameMap = {},
  emptyMessage = 'ファイルがありません',
}: ImageListTableProps) {
  const [expandedParents, setExpandedParents] = useState<Set<string>>(new Set());

  const { childPages } = useMemo(() => groupFiles(files), [files]);

  const topLevelRows = useMemo(() => {
    const all = getTopLevelFiles(files);
    if (!visibleTopLevelIds) return all;
    return all.filter((file) => visibleTopLevelIds.has(file.id));
  }, [files, visibleTopLevelIds]);

  const toggleParent = (parentId: string) => {
    setExpandedParents((prev) => {
      const next = new Set(prev);
      if (next.has(parentId)) next.delete(parentId);
      else next.add(parentId);
      return next;
    });
  };

  const isSelected = (id: string) => selectedIds?.has(id) ?? false;

  const toggleSelection = (id: string) => {
    if (!onSelectionChange || !selectedIds) return;
    onSelectionChange(toggleImageSelection(files, selectedIds, id));
  };

  // 表示中の全 ID（親 + 子 + standalone）— 全選択チェック用
  const allVisibleIds = useMemo(() => {
    const ids: string[] = [];
    topLevelRows.forEach((file) => {
      ids.push(file.id);
      const children = childPages[file.id] || [];
      children.forEach((c) => ids.push(c.id));
    });
    return ids;
  }, [topLevelRows, childPages]);

  const allSelected =
    showSelection &&
    allVisibleIds.length > 0 &&
    allVisibleIds.every((id) => selectedIds?.has(id));
  const someSelected =
    showSelection && !allSelected && allVisibleIds.some((id) => selectedIds?.has(id));

  const toggleSelectAll = () => {
    if (!onSelectionChange || !selectedIds) return;
    const next = new Set(selectedIds);
    if (allSelected) {
      allVisibleIds.forEach((id) => next.delete(id));
    } else {
      allVisibleIds.forEach((id) => next.add(id));
    }
    onSelectionChange(next);
  };

  const handleRowClick = (file: ImageFile) => {
    if (!onRowClick) return;
    if (!canOpenImage(file)) return;
    onRowClick(file);
  };

  // ファイル名列に表示するアイコン（PDF 赤 / 画像 青）
  const renderFileIcon = (file: ImageFile) => {
    if (file.name.toLowerCase().endsWith('.pdf')) {
      return <FileText size={16} className="text-danger flex-shrink-0" />;
    }
    return <ImageIcon size={16} className="text-primary flex-shrink-0" />;
  };

  // 標準ファイルのサブテキスト（PDF の処理モード等）
  const getStandaloneSubText = (file: ImageFile): string | null => {
    if (file.pageProcessingMode === 'combined' && file.totalPages && file.totalPages > 1) {
      return `統合処理 - ${file.totalPages}ページ`;
    }
    return null;
  };

  const usecaseColSpan = showUsecaseColumn ? 1 : 0;
  const selectionColSpan = showSelection ? 1 : 0;
  const totalCols = 5 + usecaseColSpan + selectionColSpan;
  const minTableWidth = showUsecaseColumn ? 1100 : 900;

  return (
    <div className="overflow-x-auto">
      <Table className="table-fixed w-full" style={{ minWidth: `${minTableWidth}px` }}>
        <colgroup>
          {showSelection && <col style={{ width: '48px' }} />}
          {/* 幅未指定のファイル名列だけが、固定列を除いた残り幅を使う。 */}
          <col />
          {showUsecaseColumn && <col style={{ width: '224px' }} />}
          <col style={{ width: '200px' }} />
          <col style={{ width: '180px' }} />
          <col style={{ width: '152px' }} />
          <col style={{ width: '72px' }} />
        </colgroup>
        <Thead>
          <tr>
            {showSelection && (
              <th className="px-4 py-3">
                <input
                  type="checkbox"
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelected;
                  }}
                  onChange={toggleSelectAll}
                  aria-label="全選択"
                  className="cursor-pointer"
                />
              </th>
            )}
            <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
              ファイル名
            </th>
            {showUsecaseColumn && (
              <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
                ユースケース
              </th>
            )}
            <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
              アップロード者
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
              日時
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">
              ステータス
            </th>
            <th className="px-4 py-3 text-center text-xs font-medium text-muted uppercase tracking-wider">
              確認
            </th>
          </tr>
        </Thead>
        <Tbody>
          {topLevelRows.length === 0 ? (
            <tr>
              <td colSpan={totalCols}>
                <EmptyState message={emptyMessage} />
              </td>
            </tr>
          ) : (
            topLevelRows.map((file) => {
              const children = childPages[file.id] || [];
              const isParent =
                file.pageProcessingMode === 'individual' &&
                !file.parentDocumentId &&
                (file.totalPages || 0) > 1 &&
                children.length > 0;
              const isExpanded = expandedParents.has(file.id);
              const rows: JSX.Element[] = [];

              if (isParent) {
                const progress = getParentProgress(children);
                const overallStatus = getParentOverallStatus(children);
                const familyIds = [file.id, ...children.map((child) => child.id)];
                const isFamilySelected = familyIds.every((id) => isSelected(id));
                const isFamilyPartiallySelected =
                  !isFamilySelected && familyIds.some((id) => isSelected(id));
                rows.push(
                  <tr
                    key={file.id}
                    className="cursor-pointer hover:bg-neutral-50"
                    onClick={() => toggleParent(file.id)}
                  >
                    {showSelection && (
                      <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={isFamilySelected}
                          ref={(el) => {
                            if (el) el.indeterminate = isFamilyPartiallySelected;
                          }}
                          onChange={() => toggleSelection(file.id)}
                          aria-label={`${file.name} を選択`}
                          className="cursor-pointer"
                        />
                      </td>
                    )}
                    <td className="px-4 py-3 text-sm">
                      <div className="flex items-center gap-2">
                        <ChevronRight
                          size={16}
                          className={`text-muted transition-transform ${isExpanded ? 'rotate-90' : ''}`}
                        />
                        {renderFileIcon(file)}
                        <div className="min-w-0">
                          <div className="font-medium truncate">{file.name}</div>
                          <div className="text-xs text-muted">
                            個別処理 - {file.totalPages}ページ ({progress.completed}/{progress.total} 完了)
                          </div>
                        </div>
                      </div>
                    </td>
                    {showUsecaseColumn && (
                      <td className="px-4 py-3 text-sm truncate">
                        {renderUsecase(file.appName, appDisplayNameMap)}
                      </td>
                    )}
                    <td className="px-4 py-3 text-sm text-muted truncate">
                      {file.uploaded_by_email || file.uploaded_by || '-'}
                    </td>
                    <td className="px-4 py-3 text-sm whitespace-nowrap">
                      <ResponsiveDateTime value={file.uploadTime} />
                    </td>
                    <td className="px-4 py-3">
                      <StatusWithPresence
                        status={overallStatus}
                        viewers={presenceByImageId?.[file.id] ?? []}
                      />
                    </td>
                    <td className="px-4 py-3 text-center text-neutral-300">-</td>
                  </tr>
                );

                // 子ページ行（展開時のみ）
                if (isExpanded) {
                  children.forEach((child) => {
                    const openable = canOpenImage(child);
                    rows.push(
                      <tr
                        key={child.id}
                        className={openable ? 'cursor-pointer hover:bg-neutral-50' : 'hover:bg-neutral-50'}
                        onClick={() => handleRowClick(child)}
                      >
                        {showSelection && (
                          <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                            <input
                              type="checkbox"
                              checked={isSelected(child.id)}
                              onChange={() => toggleSelection(child.id)}
                              aria-label={`${child.name} のページ ${child.pageNumber} を選択`}
                              className="cursor-pointer"
                            />
                          </td>
                        )}
                        <td className="px-4 py-3 text-sm">
                          <div className="flex items-center gap-2 pl-8">
                            <ImageIcon size={16} className="text-primary flex-shrink-0" />
                            <span className="text-sm">
                              ページ {child.pageNumber}/{child.totalPages}
                            </span>
                          </div>
                        </td>
                        {showUsecaseColumn && (
                          <td className="px-4 py-3 text-sm text-muted truncate">-</td>
                        )}
                        <td className="px-4 py-3 text-sm text-muted truncate">
                          {child.uploaded_by_email || child.uploaded_by || '-'}
                        </td>
                        <td className="px-4 py-3 text-sm whitespace-nowrap">
                          <ResponsiveDateTime value={child.uploadTime} />
                        </td>
                        <td className="px-4 py-3">
                          <StatusWithPresence
                            status={child.status}
                            agentStatus={child.agentStatus}
                            viewers={presenceByImageId?.[child.id] ?? []}
                          />
                        </td>
                        <td className="px-4 py-3 text-center">
                          {renderVerificationCell(child)}
                        </td>
                      </tr>
                    );
                  });
                }
              } else {
                // 標準ファイル（1 行）
                const openable = canOpenImage(file);
                const subText = getStandaloneSubText(file);
                rows.push(
                  <tr
                    key={file.id}
                    className={openable ? 'cursor-pointer hover:bg-neutral-50' : 'hover:bg-neutral-50'}
                    onClick={() => handleRowClick(file)}
                  >
                    {showSelection && (
                      <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                        <input
                          type="checkbox"
                          checked={isSelected(file.id)}
                          onChange={() => toggleSelection(file.id)}
                          aria-label={`${file.name} を選択`}
                          className="cursor-pointer"
                        />
                      </td>
                    )}
                    <td className="px-4 py-3 text-sm">
                      <div className="flex items-center gap-2">
                        {/* ChevronRight のスペースを合わせるためのプレースホルダ */}
                        <span className="w-4" />
                        {renderFileIcon(file)}
                        <div className="min-w-0">
                          <div className="font-medium truncate">{file.name}</div>
                          {subText && <div className="text-xs text-muted">{subText}</div>}
                        </div>
                      </div>
                    </td>
                    {showUsecaseColumn && (
                      <td className="px-4 py-3 text-sm truncate">
                        {renderUsecase(file.appName, appDisplayNameMap)}
                      </td>
                    )}
                    <td className="px-4 py-3 text-sm text-muted truncate">
                      {file.uploaded_by_email || file.uploaded_by || '-'}
                    </td>
                    <td className="px-4 py-3 text-sm whitespace-nowrap">
                      <ResponsiveDateTime value={file.uploadTime} />
                    </td>
                    <td className="px-4 py-3">
                      <StatusWithPresence
                        status={file.status}
                        agentStatus={file.agentStatus}
                        viewers={presenceByImageId?.[file.id] ?? []}
                      />
                    </td>
                    <td className="px-4 py-3 text-center">
                      {renderVerificationCell(file)}
                    </td>
                  </tr>
                );
              }
              return rows;
            })
          )}
        </Tbody>
      </Table>
    </div>
  );
}

function StatusWithPresence({
  status,
  agentStatus,
  viewers,
}: {
  status: string;
  agentStatus?: ImageFile['agentStatus'];
  viewers: PresenceViewer[];
}): React.ReactNode {
  return (
    <div className="flex items-center min-w-0">
      <ProcessStatusBadge status={status} agentStatus={agentStatus} />
      <div className="ml-auto pl-2 flex-shrink-0">
        <PresenceBadge compact viewers={viewers} />
      </div>
    </div>
  );
}

function ResponsiveDateTime({ value }: { value?: string }): React.ReactNode {
  const dateTime = formatDateTimeJST(value || '');
  const dateOnly = formatDateJST(value || '');

  return (
    <span title={dateTime} className="whitespace-nowrap">
      <span className="hidden lg:inline">{dateTime || '-'}</span>
      <span className="lg:hidden">{dateOnly || '-'}</span>
    </span>
  );
}

function renderUsecase(
  appName: string | undefined,
  displayNameMap: Record<string, string>
): React.ReactNode {
  if (!appName) return '-';
  const display = displayNameMap[appName];
  return display ? (
    <>
      {display}
      <span className="text-neutral-400 ml-1">（{appName}）</span>
    </>
  ) : (
    appName
  );
}

function renderVerificationCell(file: ImageFile): React.ReactNode {
  if (file.verificationCompleted) {
    return (
      <Tooltip content={file.verified_by_email || '確認済み'}>
        <CheckCircle size={18} className="text-success inline-block" />
      </Tooltip>
    );
  }
  if ((file.agentSuggestionsCount ?? 0) > 0) {
    return (
      <span className="inline-flex items-center justify-center min-w-5 h-5 px-1.5 text-[10px] font-bold text-white bg-warning rounded-full">
        {file.agentSuggestionsCount}
      </span>
    );
  }
  return <span className="text-neutral-300">-</span>;
}

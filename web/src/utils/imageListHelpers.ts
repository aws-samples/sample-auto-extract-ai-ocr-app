import type { ImageFile } from '../types/ocr';

/**
 * ファイル 1 件の「確認状態」カテゴリ。
 * FilterTabs のタブ切り替えや、Upload 画面の要対応件数バッジ算出に使う。
 */
export type ConfirmationState = 'failed' | 'processing' | 'confirmed' | 'action_needed';

/**
 * FilterTabs で選択できるフィルタキー。
 * 'all' は絞り込み無し、それ以外は ConfirmationState と一致。
 */
export type FilterKey = 'all' | ConfirmationState;

/**
 * ファイルの確認状態を判定する。
 *
 * - failed: OCR/抽出が失敗した or Agent 検証が失敗した
 * - processing: OCR/抽出中 or Agent 検証中
 * - confirmed: OCR/抽出完了 かつ verificationCompleted=true
 * - action_needed: OCR/抽出完了 かつ未確認（対応が必要）
 *
 * 判定順序が重要: failed → processing → confirmed → action_needed
 */
export function getConfirmationState(file: ImageFile): ConfirmationState {
  if (file.status === 'failed' || file.agentStatus === 'failed') return 'failed';
  if (file.status !== 'completed') return 'processing';
  if (file.agentStatus === 'processing') return 'processing';
  if (file.verificationCompleted) return 'confirmed';
  return 'action_needed';
}

/**
 * OCR 結果画面（/ocr-result/:id）を開ける状態か判定する。
 * OCR/抽出が完了していれば Agent 検証中でも開ける。
 */
export function canOpenImage(file: Pick<ImageFile, 'status'>): boolean {
  return file.status === 'completed';
}

/**
 * PDF 親子構造ごとに分類したファイルグループ。
 *
 * - parentDocuments: 個別処理された複数ページ PDF の親ドキュメント
 * - childPages: 親ドキュメント ID → 子ページ配列（ページ番号昇順）
 * - standaloneFiles: 親子関係を持たない単発ファイル、または親が取得できなかった子ページ
 */
export interface GroupedFiles {
  parentDocuments: ImageFile[];
  childPages: Record<string, ImageFile[]>;
  standaloneFiles: ImageFile[];
}

export interface ImageFamily {
  root: ImageFile;
  children: ImageFile[];
}

export const isParentDocument = (file: ImageFile): boolean =>
  file.pageProcessingMode === 'individual' &&
  !file.parentDocumentId &&
  (file.totalPages || 0) > 1;

/**
 * 親を先に確定してから子を割り当てる。
 * API の不完全な結果などで親が含まれない子は、一覧から消さず standalone として残す。
 */
export function groupFiles(files: ImageFile[]): GroupedFiles {
  const parentDocuments = files.filter(isParentDocument);
  const parentIds = new Set(parentDocuments.map((file) => file.id));
  const childPages: Record<string, ImageFile[]> = {};
  const standaloneFiles: ImageFile[] = [];

  files.forEach((file) => {
    if (parentIds.has(file.id)) return;

    if (file.parentDocumentId && parentIds.has(file.parentDocumentId)) {
      if (!childPages[file.parentDocumentId]) childPages[file.parentDocumentId] = [];
      childPages[file.parentDocumentId].push(file);
      return;
    }

    standaloneFiles.push(file);
  });

  Object.values(childPages).forEach((children) => {
    children.sort((a, b) => (a.pageNumber || 0) - (b.pageNumber || 0));
  });

  return { parentDocuments, childPages, standaloneFiles };
}

/**
 * ページネーションの単位となる親ドキュメント / standalone を返す。
 * 子ページは親と同じページに表示するため、ページ件数には含めない。
 */
export function getTopLevelFiles(files: ImageFile[]): ImageFile[] {
  const { parentDocuments, standaloneFiles } = groupFiles(files);
  return [...parentDocuments, ...standaloneFiles].sort((a, b) =>
    (b.uploadTime || '').localeCompare(a.uploadTime || '')
  );
}

export function getImageFamilies(files: ImageFile[]): ImageFamily[] {
  const { childPages } = groupFiles(files);
  return getTopLevelFiles(files).map((root) => ({
    root,
    children: childPages[root.id] || [],
  }));
}

/**
 * family のどれか 1 件が条件に一致した場合、親と全子ページをまとめて保持する。
 * フィルタや検索で親だけが消え、子ページが孤立することを防ぐ。
 */
export function filterImageFamilies(
  files: ImageFile[],
  predicate: (file: ImageFile) => boolean
): ImageFile[] {
  const includedIds = new Set<string>();

  getImageFamilies(files).forEach(({ root, children }) => {
    const members = [root, ...children];
    if (members.some(predicate)) {
      members.forEach((member) => includedIds.add(member.id));
    }
  });

  return files.filter((file) => includedIds.has(file.id));
}

/**
 * Image の選択状態を親子 family 単位で更新する。
 * 親操作は family 全体へ連動し、子操作後は全子選択時だけ親も選択状態にする。
 */
export function toggleImageSelection(
  files: ImageFile[],
  selectedIds: ReadonlySet<string>,
  targetId: string
): Set<string> {
  const next = new Set(selectedIds);
  const { childPages } = groupFiles(files);
  const children = childPages[targetId] || [];

  if (children.length > 0) {
    const familyIds = [targetId, ...children.map((child) => child.id)];
    const shouldSelectAll = !familyIds.every((id) => next.has(id));
    familyIds.forEach((id) => {
      if (shouldSelectAll) next.add(id);
      else next.delete(id);
    });
    return next;
  }

  if (next.has(targetId)) next.delete(targetId);
  else next.add(targetId);

  const target = files.find((file) => file.id === targetId);
  const parentId = target?.parentDocumentId;
  const siblings = parentId ? childPages[parentId] || [] : [];

  if (parentId && siblings.length > 0) {
    if (siblings.every((child) => next.has(child.id))) next.add(parentId);
    else next.delete(parentId);
  }

  return next;
}

/**
 * 親削除はバックエンドで全子ページも削除するため、親と子を重複して API 呼び出ししない。
 * 親を選択した場合、または全子ページを選択した場合は親 1 件に畳み込む。
 */
export function normalizeDeletionTargets(
  files: ImageFile[],
  selectedIds: ReadonlySet<string>
): string[] {
  const targets = new Set<string>();
  const knownIds = new Set<string>();

  getImageFamilies(files).forEach(({ root, children }) => {
    knownIds.add(root.id);
    children.forEach((child) => knownIds.add(child.id));

    if (selectedIds.has(root.id)) {
      targets.add(root.id);
      return;
    }

    const selectedChildren = children.filter((child) => selectedIds.has(child.id));
    if (children.length > 0 && selectedChildren.length === children.length) {
      targets.add(root.id);
      return;
    }

    selectedChildren.forEach((child) => targets.add(child.id));
  });

  // ポーリング直後など、現在の files に存在しない選択 ID も黙って落とさない。
  selectedIds.forEach((id) => {
    if (!knownIds.has(id)) targets.add(id);
  });

  return Array.from(targets);
}

/**
 * 選択された Image から OCR 開始対象を抽出する。
 * 個別ページ PDF の親は表示用コンテナなので、親選択時は pending の子ページへ展開する。
 * standalone / orphan child は、自身が pending かつ選択済みの場合だけ対象にする。
 */
export function normalizeOcrTargets(
  files: ImageFile[],
  selectedIds: ReadonlySet<string>
): string[] {
  const targets = new Set<string>();

  getImageFamilies(files).forEach(({ root, children }) => {
    if (isParentDocument(root)) {
      if (selectedIds.has(root.id)) {
        children
          .filter((child) => child.status === 'pending')
          .forEach((child) => targets.add(child.id));
        return;
      }

      children
        .filter((child) => selectedIds.has(child.id) && child.status === 'pending')
        .forEach((child) => targets.add(child.id));
      return;
    }

    if (selectedIds.has(root.id) && root.status === 'pending') {
      targets.add(root.id);
    }
  });

  return Array.from(targets);
}

/**
 * 親ドキュメントの進捗（完了ページ数 / 総ページ数）。
 */
export function getParentProgress(children: ImageFile[]): { completed: number; total: number } {
  const completed = children.filter((c) => c.status === 'completed').length;
  return { completed, total: children.length };
}

/**
 * 親ドキュメント全体のステータスを子ページから算出する。
 * ProcessStatusBadge に渡す想定なので Image.status と互換の値を返す。
 *
 * 優先順位:
 *   1. 子に失敗があれば failed
 *   2. 子に処理中（ocr/extracting/processing）があれば processing
 *   3. 子に pending があれば pending
 *   4. 子に検証中があれば processing（表示上は「検証中」になる）
 *   5. すべて完了なら completed
 */
export function getParentOverallStatus(children: ImageFile[]): ImageFile['status'] {
  if (children.length === 0) return 'pending';
  if (children.some((c) => c.status === 'failed' || c.agentStatus === 'failed')) return 'failed';
  if (children.some((c) => ['ocr', 'extracting', 'processing'].includes(c.status))) return 'processing';
  if (children.some((c) => c.status === 'pending')) return 'pending';
  if (children.some((c) => c.agentStatus === 'processing')) return 'processing';
  return 'completed';
}

/**
 * フィルタキーに応じて family 単位で絞り込む。
 * 子ページだけが一致しても、表示コンテナとなる親と兄弟ページを保持する。
 */
export function applyFilter(files: ImageFile[], filter: FilterKey): ImageFile[] {
  if (filter === 'all') return files;
  return filterImageFamilies(files, (file) => getConfirmationState(file) === filter);
}

import type { ImageFile } from '../../types/ocr';
import type { FilterKey, ConfirmationState } from '../../utils/imageListHelpers';
import { getConfirmationState } from '../../utils/imageListHelpers';

interface ImageListFilterTabsProps {
  files: ImageFile[];
  value: FilterKey;
  onChange: (key: FilterKey) => void;
}

const TAB_ITEMS: { key: FilterKey; label: string }[] = [
  { key: 'all', label: '全て' },
  { key: 'action_needed', label: '要対応' },
  { key: 'confirmed', label: '確認済み' },
  { key: 'processing', label: '処理待ち' },
  { key: 'failed', label: '失敗' },
];

/**
 * Image 一覧の確認状態カテゴリを切り替えるタブ。
 * 各タブに現在の件数を併記して、対応が必要な件数が一目で分かるようにする。
 *
 * 件数計算は親渡しの `files` に対して都度 reduce するため、
 * files が変化しないうちは同じ結果になる（メモ化はページ側に任せる）。
 */
export function ImageListFilterTabs({ files, value, onChange }: ImageListFilterTabsProps) {
  const counts = files.reduce<Record<ConfirmationState, number>>(
    (acc, f) => {
      const state = getConfirmationState(f);
      acc[state] = (acc[state] || 0) + 1;
      return acc;
    },
    { failed: 0, processing: 0, confirmed: 0, action_needed: 0 }
  );

  const getCount = (key: FilterKey): number => (key === 'all' ? files.length : counts[key]);

  return (
    <div className="flex gap-1 border-b border-default">
      {TAB_ITEMS.map((item) => {
        const active = value === item.key;
        const count = getCount(item.key);
        return (
          <button
            key={item.key}
            type="button"
            onClick={() => onChange(item.key)}
            className={`relative flex items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors
              ${active ? 'text-primary' : 'text-muted hover:text-default'}`}
          >
            <span>{item.label}</span>
            <span className={`text-xs ${active ? 'text-primary' : 'text-neutral-400'}`}>
              ({count})
            </span>
            {active && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-t" />
            )}
          </button>
        );
      })}
    </div>
  );
}

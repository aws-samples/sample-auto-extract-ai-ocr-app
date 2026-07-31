export interface PresenceViewer {
  userId: string;
  displayName?: string | null;
  /** 自分自身の接続かどうか（usePresence が付与する） */
  isSelf?: boolean;
}

interface PresenceBadgeProps {
  viewers: PresenceViewer[];
  /** 一覧向け。先頭1人の小アイコンと、残り人数だけを表示する。 */
  compact?: boolean;
}

const DEFAULT_AVATAR_SIZE = 'w-7 h-7 text-xs';
const COMPACT_AVATAR_SIZE = 'w-5 h-5 text-[10px]';

const getViewerName = (viewer: PresenceViewer): string =>
  viewer.displayName || viewer.userId;

const getViewerColor = (viewer: PresenceViewer): string =>
  viewer.isSelf ? 'bg-primary text-on-primary' : 'bg-neutral-400 text-white';

/**
 * 同じリソース（image_id）を今見ているユーザーを表示する。
 * 詳細画面では全員のアバター、一覧では情報量を抑えた先頭1人 + 残り人数を使う。
 */
function PresenceBadge({ viewers, compact = false }: PresenceBadgeProps) {
  if (viewers.length === 0) return null;

  if (compact) {
    const first = viewers[0];
    const remaining = viewers.slice(1);
    const initial = getViewerName(first).charAt(0).toUpperCase() || '?';
    const firstTitle = first.isSelf ? `自分（${getViewerName(first)}）` : getViewerName(first);

    return (
      <div className="flex items-center gap-1 flex-shrink-0" aria-label={`${viewers.length}人が閲覧中`}>
        <div
          className={`rounded-full flex items-center justify-center font-medium ring-1 ring-bg ${COMPACT_AVATAR_SIZE} ${getViewerColor(first)}`}
          title={firstTitle}
        >
          {initial}
        </div>
        {remaining.length > 0 && (
          <span
            className="text-[11px] text-muted whitespace-nowrap"
            title={remaining.map(getViewerName).join('、')}
          >
            +{remaining.length}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="flex items-center -space-x-1.5">
      {viewers.map((viewer, i) => {
        const initial = getViewerName(viewer).charAt(0).toUpperCase() || '?';
        return (
          <div
            key={`${viewer.userId}-${i}`}
            className={`rounded-full flex items-center justify-center font-medium ring-2 ring-bg ${DEFAULT_AVATAR_SIZE} ${getViewerColor(viewer)}`}
            title={viewer.isSelf ? `自分（${getViewerName(viewer)}）` : getViewerName(viewer)}
          >
            {initial}
          </div>
        );
      })}
    </div>
  );
}

export default PresenceBadge;

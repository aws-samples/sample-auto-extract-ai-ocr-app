export interface PresenceViewer {
  userId: string;
  displayName?: string | null;
  /** 自分自身の接続かどうか（usePresence が付与する） */
  isSelf?: boolean;
}

interface PresenceBadgeProps {
  viewers: PresenceViewer[];
}

const AVATAR_SIZE = "w-7 h-7 text-xs";

/**
 * 同じリソース（image_id）を今見ている他ユーザーのアバターアイコン一覧。
 * 右上の UserMenu の Avatar と同じ丸アイコン・イニシャル表示に揃えている。
 * 色は role 別ではなく「自分/他人」の2値のみで区別する:
 *   - 自分:  bg-primary（塗り、目立たせる）
 *   - 他人:  bg-neutral-400（グレー、控えめ）
 */
function PresenceBadge({ viewers }: PresenceBadgeProps) {
  if (viewers.length === 0) {
    return null;
  }

  return (
    <div className="flex items-center -space-x-1.5">
      {viewers.map((viewer, i) => {
        const initial = (viewer.displayName || viewer.userId).charAt(0).toUpperCase() || "?";
        const color = viewer.isSelf ? "bg-primary text-on-primary" : "bg-neutral-400 text-white";
        return (
          <div
            key={`${viewer.userId}-${i}`}
            className={`rounded-full flex items-center justify-center font-medium ring-2 ring-bg ${AVATAR_SIZE} ${color}`}
            title={viewer.isSelf ? `自分（${viewer.displayName || viewer.userId}）` : viewer.displayName || viewer.userId}
          >
            {initial}
          </div>
        );
      })}
    </div>
  );
}

export default PresenceBadge;

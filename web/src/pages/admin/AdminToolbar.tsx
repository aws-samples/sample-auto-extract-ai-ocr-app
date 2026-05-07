import type { ReactNode } from 'react';

interface AdminToolbarProps {
  left?: ReactNode;
  right?: ReactNode;
}

/** 各タブ共通のツールバー（左: 検索+フィルター、右: アクションボタン） */
export function AdminToolbar({ left, right }: AdminToolbarProps) {
  return (
    <div className="flex items-center justify-between mb-4 flex-wrap gap-3">
      <div className="flex items-center gap-3 flex-wrap">{left}</div>
      <div className="flex items-center gap-2">{right}</div>
    </div>
  );
}

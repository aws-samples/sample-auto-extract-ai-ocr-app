import { useState, useRef, useEffect, type ReactNode } from 'react';
import { MoreHorizontal } from 'lucide-react';

interface RowActionMenuProps {
  children: (close: () => void) => ReactNode;
}

/** 行末の「...」ボタン + ドロップダウンメニュー */
export function RowActionMenu({ children }: RowActionMenuProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={(e) => { e.stopPropagation(); setOpen(!open); }}
        className="p-1 rounded hover:bg-neutral-100 transition-colors"
        aria-label="操作メニュー"
      >
        <MoreHorizontal size={16} className="text-neutral-500" />
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-20 bg-bg border border-default rounded-lg shadow-lg py-1 min-w-[160px]">
          {children(() => setOpen(false))}
        </div>
      )}
    </div>
  );
}

interface MenuItemProps {
  onClick: () => void;
  children: ReactNode;
  danger?: boolean;
}

export function MenuItem({ onClick, children, danger }: MenuItemProps) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      className={`w-full text-left px-3 py-1.5 text-sm transition-colors
        ${danger ? 'text-danger hover:bg-danger-light' : 'text-default hover:bg-neutral-50'}`}
    >
      {children}
    </button>
  );
}

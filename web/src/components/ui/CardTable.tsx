import type { ReactNode } from 'react';

interface CardTableProps {
  children: ReactNode;
  className?: string;
}

/** テーブルを角丸カードで囲むラッパー */
export function CardTable({ children, className = '' }: CardTableProps) {
  return (
    <div className={`rounded-xl border border-default shadow-sm overflow-x-auto ${className}`}>
      {children}
    </div>
  );
}

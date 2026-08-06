import { useState } from 'react';
import { Search, ChevronsLeft, ChevronLeft, ChevronRight, ChevronsRight } from 'lucide-react';

const PAGE_SIZE_OPTIONS = [10, 20, 50];

export function usePagination<T>(items: T[], defaultPageSize = 10) {
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(defaultPageSize);
  const total = Math.ceil(items.length / pageSize);
  const paged = items.slice(page * pageSize, (page + 1) * pageSize);

  const changePageSize = (size: number) => {
    setPageSize(size);
    setPage(0);
  };

  return { page, setPage, total, paged, pageSize, changePageSize, totalItems: items.length };
}

interface PaginationProps {
  page: number;
  total: number;
  setPage: (p: number) => void;
  pageSize?: number;
  totalItems?: number;
  onPageSizeChange?: (size: number) => void;
}

export function Pagination({ page, total, setPage, pageSize, totalItems, onPageSizeChange }: PaginationProps) {
  if (total <= 1 && !onPageSizeChange) return null;

  const start = totalItems ? page * (pageSize || 10) + 1 : 0;
  const end = totalItems ? Math.min((page + 1) * (pageSize || 10), totalItems) : 0;

  // Generate page numbers to show (max 5 visible)
  const pageNumbers: (number | '...')[] = [];
  if (total <= 7) {
    for (let i = 0; i < total; i++) pageNumbers.push(i);
  } else {
    pageNumbers.push(0);
    if (page > 2) pageNumbers.push('...');
    for (let i = Math.max(1, page - 1); i <= Math.min(total - 2, page + 1); i++) {
      pageNumbers.push(i);
    }
    if (page < total - 3) pageNumbers.push('...');
    pageNumbers.push(total - 1);
  }

  return (
    <div className="flex items-center justify-between mt-4 text-sm">
      <div className="flex items-center gap-4 text-muted">
        {onPageSizeChange && (
          <div className="flex items-center gap-1.5">
            <span>表示件数</span>
            <select
              value={pageSize}
              onChange={(e) => onPageSizeChange(Number(e.target.value))}
              className="border border-default rounded px-1.5 py-0.5 text-sm bg-bg"
            >
              {PAGE_SIZE_OPTIONS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        )}
        {totalItems !== undefined && totalItems > 0 && (
          <span>全 {totalItems} 件中 {start}-{end} 件</span>
        )}
      </div>

      {total > 1 && (
        <div className="flex items-center gap-1">
          <NavButton onClick={() => setPage(0)} disabled={page === 0} aria-label="最初のページ">
            <ChevronsLeft size={14} />
          </NavButton>
          <NavButton onClick={() => setPage(page - 1)} disabled={page === 0} aria-label="前のページ">
            <ChevronLeft size={14} />
          </NavButton>

          {pageNumbers.map((p, i) =>
            p === '...' ? (
              <span key={`dots-${i}`} className="px-1 text-muted">…</span>
            ) : (
              <button
                key={p}
                onClick={() => setPage(p)}
                className={`min-w-[28px] h-7 rounded text-sm transition-colors
                  ${page === p ? 'bg-primary text-on-primary' : 'hover:bg-neutral-100 text-default'}`}
              >
                {p + 1}
              </button>
            )
          )}

          <NavButton onClick={() => setPage(page + 1)} disabled={page >= total - 1} aria-label="次のページ">
            <ChevronRight size={14} />
          </NavButton>
          <NavButton onClick={() => setPage(total - 1)} disabled={page >= total - 1} aria-label="最後のページ">
            <ChevronsRight size={14} />
          </NavButton>
        </div>
      )}
    </div>
  );
}

function NavButton({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      {...props}
      className={`p-1.5 rounded border border-default transition-colors
        ${props.disabled ? 'opacity-40 cursor-not-allowed' : 'hover:bg-neutral-50'}`}
    >
      {children}
    </button>
  );
}

export function SearchBox({ value, onChange, placeholder }: { value: string; onChange: (v: string) => void; placeholder?: string }) {
  return (
    <div className="relative">
      <Search size={16} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-neutral-400" />
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder={placeholder || '検索...'}
        className="pl-8 pr-3 py-1.5 border border-default rounded-lg text-sm w-64 bg-bg"
      />
    </div>
  );
}

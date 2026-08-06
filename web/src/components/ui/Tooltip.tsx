import { useState, useRef, useCallback, type ReactNode } from 'react';
import { createPortal } from 'react-dom';

interface TooltipProps {
  content: string;
  children: ReactNode;
}

/** Portal ベースの Tooltip。親の overflow に影響されない。 */
export function Tooltip({ content, children }: TooltipProps) {
  const [visible, setVisible] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0 });
  const ref = useRef<HTMLSpanElement>(null);

  const show = useCallback(() => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    setPos({
      top: rect.top - 4,
      left: rect.left + rect.width / 2,
    });
    setVisible(true);
  }, []);

  const hide = useCallback(() => setVisible(false), []);

  return (
    <>
      <span ref={ref} onMouseEnter={show} onMouseLeave={hide} className="inline-block">
        {children}
      </span>
      {visible &&
        createPortal(
          <span
            style={{ top: pos.top, left: pos.left }}
            className="pointer-events-none fixed -translate-x-1/2 -translate-y-full whitespace-nowrap rounded bg-neutral-800 px-2 py-1 text-xs text-white z-50"
          >
            {content}
          </span>,
          document.body
        )}
    </>
  );
}

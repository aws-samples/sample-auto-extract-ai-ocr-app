import { useEffect, useRef } from 'react';

interface UsePollingOptions {
  /** ポーリング間隔（ミリ秒） */
  interval: number;
  /** ポーリングを有効にするかどうか */
  enabled?: boolean;
}

/**
 * 共通ポーリングフック
 * コンポーネントのアンマウント時に自動でクリーンアップされる
 */
export function usePolling(callback: () => void | Promise<void>, options: UsePollingOptions) {
  const { interval, enabled = true } = options;
  const savedCallback = useRef(callback);

  // コールバックを最新に保つ（再レンダリングでタイマーをリセットしない）
  useEffect(() => {
    savedCallback.current = callback;
  }, [callback]);

  useEffect(() => {
    if (!enabled) return;

    const tick = () => savedCallback.current();

    const id = setInterval(tick, interval);
    return () => clearInterval(id);
  }, [interval, enabled]);
}

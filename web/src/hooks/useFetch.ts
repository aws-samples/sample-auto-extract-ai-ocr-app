import { useEffect, useState, useCallback } from 'react';

/**
 * マウント時に fetcher を実行し、loading 状態と再取得手段を提供する共通フック。
 *
 * 各画面で散在していた「data/loading の useState + load の useCallback + useEffect」
 * の定型を一元化する。fetcher は表示用データを返す（一覧の抽出やソート等の整形は
 * 呼び出し側で行い、その結果を data にセットする）。
 *
 * @param fetcher 表示用データを返す非同期関数（安定参照を渡すこと）
 * @param initial data の初期値
 */
export function useFetch<T>(fetcher: () => Promise<T>, initial: T) {
  const [data, setData] = useState<T>(initial);
  const [loading, setLoading] = useState(true);

  const refetch = useCallback(async () => {
    setLoading(true);
    try {
      setData(await fetcher());
    } finally {
      setLoading(false);
    }
  }, [fetcher]);

  useEffect(() => {
    refetch();
  }, [refetch]);

  return { data, loading, refetch };
}

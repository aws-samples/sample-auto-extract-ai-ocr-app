import { useEffect, useRef, useState, useCallback } from "react";
import { fetchAuthSession } from "aws-amplify/auth";
import { PresenceViewer } from "../components/shared/PresenceBadge";

export type { PresenceViewer };

interface UsePresenceOptions {
  /**
   * プレゼンスを表示する対象の image_id。
   * - 文字列: 個別ページ（OCR結果画面）用の subscribe モード
   * - "__all__": 一覧ページ用の subscribeAll モード
   * - undefined: まだ image_id が確定していない（接続しない）
   */
  imageId?: string;
}

/** 一覧ページ用の全体購読を表す特別な imageId 値 */
export const PRESENCE_LIST_MODE = "__all__";

// API Gateway WebSocket API はアイドル状態が10分続くとハード切断される
// （AWS公式の制約、変更不可）。これより十分短い間隔でheartbeatを送る。
const HEARTBEAT_INTERVAL_MS = 5 * 60 * 1000; // 5分
const RECONNECT_DELAY_MS = 3000;

/**
 * WebSocketでプレゼンス（同じimage_idを見ている他ユーザー）情報を取得するフック。
 * 既存の usePolling とは責務が異なる（サーバー処理完了待ちではなく、
 * クライアント間のリアルタイム状態共有のため）。
 *
 * - imageId を指定: 個別ページ（OCR結果画面）用。その image_id の視聴者一覧(viewers)を返す
 * - imageId を省略: 一覧ページ用。全 image_id ごとの視聴者数マップ(byImageId)を返す
 *
 * 返り値の各 viewer には isSelf（自分自身の接続かどうか）が付与される。
 * 表示側は isSelf で色を分けることを想定（他人と自分の視聴を区別する）。
 */
export function usePresence({ imageId }: UsePresenceOptions) {
  const [viewers, setViewers] = useState<PresenceViewer[]>([]);
  const [byImageId, setByImageId] = useState<Record<string, PresenceViewer[]>>({});
  const wsRef = useRef<WebSocket | null>(null);
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const imageIdRef = useRef(imageId);
  imageIdRef.current = imageId;
  const selfSubRef = useRef<string | null>(null);

  const wsUrl = import.meta.env.VITE_WEBSOCKET_URL as string | undefined;
  const isListMode = imageId === PRESENCE_LIST_MODE;
  // imageId が未確定（undefined）の間は接続しない。個別ページの id 取得前に
  // 誤って全体購読モードへ接続してしまうのを防ぐため。
  const presenceEnabled = !!wsUrl && !!imageId;

  const markSelf = useCallback((list: PresenceViewer[]): PresenceViewer[] => {
    if (!selfSubRef.current) return list;
    return list.map((v) => ({ ...v, isSelf: v.userId === selfSubRef.current }));
  }, []);

  const cleanup = useCallback(() => {
    if (heartbeatTimerRef.current) {
      clearInterval(heartbeatTimerRef.current);
      heartbeatTimerRef.current = null;
    }
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
  }, []);

  useEffect(() => {
    if (!presenceEnabled) {
      return;
    }

    let disposed = false;

    const connect = async () => {
      if (disposed) return;
      try {
        const { tokens } = await fetchAuthSession();
        const idToken = tokens?.idToken?.toString();
        if (!idToken || disposed) return;
        selfSubRef.current = (tokens?.idToken?.payload?.sub as string) ?? null;

        const ws = new window.WebSocket(`${wsUrl}?idToken=${encodeURIComponent(idToken)}`);
        wsRef.current = ws;

        ws.onopen = () => {
          if (disposed) return;
          if (isListMode) {
            ws.send(JSON.stringify({ action: "subscribeAll" }));
          } else {
            ws.send(JSON.stringify({ action: "subscribe", imageId: imageIdRef.current }));
          }

          heartbeatTimerRef.current = setInterval(() => {
            if (ws.readyState === window.WebSocket.OPEN) {
              ws.send(JSON.stringify({ action: "heartbeat", imageId: imageIdRef.current }));
            }
          }, HEARTBEAT_INTERVAL_MS);
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            if (data.type === "presence") {
              setViewers(markSelf(data.viewers ?? []));
            } else if (data.type === "presence_all") {
              const mapped: Record<string, PresenceViewer[]> = {};
              for (const [imgId, list] of Object.entries(data.byImageId ?? {})) {
                mapped[imgId] = markSelf(list as PresenceViewer[]);
              }
              setByImageId(mapped);
            }
          } catch {
            // 不正なメッセージは無視
          }
        };

        ws.onclose = () => {
          if (heartbeatTimerRef.current) {
            clearInterval(heartbeatTimerRef.current);
            heartbeatTimerRef.current = null;
          }
          if (!disposed) {
            // 再接続（ネットワーク切断・10分アイドルタイムアウト等からの復帰）
            reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
          }
        };

        ws.onerror = () => {
          ws.close();
        };
      } catch {
        if (!disposed) {
          reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      }
    };

    connect();

    return () => {
      disposed = true;
      cleanup();
      setViewers([]);
      setByImageId({});
    };
    // imageId が変わった場合は再接続して re-subscribe する
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [presenceEnabled, imageId, isListMode, wsUrl, cleanup, markSelf]);

  return { viewers, byImageId };
}

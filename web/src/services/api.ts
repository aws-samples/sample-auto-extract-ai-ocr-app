import axios from 'axios';
import { fetchAuthSession } from 'aws-amplify/auth';

// 環境変数からAPIのベースURLを取得
const apiBaseUrl = import.meta.env.VITE_API_BASE_URL;

// URLが正しい形式であることを確認
const ensureHttps = (url: string) => {
  if (!url) return '';
  // すでにhttps://で始まっている場合はそのまま返す
  if (url.startsWith('https://')) return url;
  // httpsプロトコルを追加
  return `https://${url}`;
};

const api = axios.create({
  baseURL: ensureHttps(apiBaseUrl),
});

// デバッグ用のログフラグ（本番環境では無効にする）
const enableDebugLogs = false;

// リクエストインターセプター
api.interceptors.request.use(
  async (config) => {
    try {
      const { tokens } = await fetchAuthSession();
      const idToken = tokens?.idToken?.toString();

      if (idToken) {
        config.headers.Authorization = `Bearer ${idToken}`;
      } else if (enableDebugLogs) {
        console.warn("認証トークンが取得できません");
      }
      
      if (enableDebugLogs) {
        console.log('API Request:', {
          url: config.url,
          baseURL: config.baseURL,
          fullURL: `${config.baseURL}${config.url}`,
          method: config.method
        });
      }
    } catch (error) {
      if (enableDebugLogs) {
        console.error("認証トークン取得エラー:", error);
      }
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// レスポンスインターセプター
api.interceptors.response.use(
  (response) => {
    if (enableDebugLogs) {
      console.log('API Response Success:', {
        status: response.status,
        url: response.config.url
      });
    }
    return response;
  },
  (error) => {
    if (enableDebugLogs) {
      if (error.response) {
        console.error('API Error:', error.response.status, error.response.data);
      } else if (error.request) {
        console.error('API Request Error:', error.request);
      } else {
        console.error('API Config Error:', error.message);
      }
    }
    return Promise.reject(error);
  }
);

// pydantic v2 の 422 detail 1件を表示用の1行に整形する。
// - ValueError は msg が "Value error, <本文>" になるため接頭辞を除去
// - 必須キー欠落（type: "missing"）は英語 "Field required" になるため日本語化
function formatDetailEntry(d: any): string {
  if (typeof d === 'string') return d;
  if (d?.type === 'missing') return '必須項目が入力されていません';
  const msg: string = d?.msg || '';
  return msg.replace(/^Value error,\s*/, '');
}

/**
 * API エラーから表示用メッセージを取り出す。
 * FastAPI の 422 は detail が配列（[{loc, msg, type}, ...]）で返るため、
 * そのままだと "[object Object]" になる。string / 配列 の両方を読める形に整形する。
 */
export function extractApiErrorMessage(err: any, fallback = '不明なエラーが発生しました'): string {
  const detail = err?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    const lines = detail.map(formatDetailEntry).filter(Boolean);
    if (lines.length > 0) return lines.join('\n');
  }
  return err?.message || fallback;
}

export default api;

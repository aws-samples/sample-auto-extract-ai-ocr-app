/**
 * 日付・時刻フォーマット用のユーティリティ関数
 */

/**
 * タイムゾーン情報のない API 日時は UTC として扱う。
 */
const parseApiDate = (dateString: string): Date | null => {
  if (!dateString) return null;

  const hasZ = dateString.endsWith('Z');
  const hasTimezoneOffset = /[+-]\d{2}:?\d{2}$/.test(dateString);
  const date = new Date(hasZ || hasTimezoneOffset ? dateString : `${dateString}Z`);

  return isNaN(date.getTime()) ? null : date;
};

/**
 * UTC時刻文字列を日本時間の日付・時刻で表示する。
 */
export const formatDateTimeJST = (dateString: string): string => {
  const date = parseApiDate(dateString);
  if (!date) return '';

  return date.toLocaleString('ja-JP', {
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
};

/**
 * UTC時刻文字列を日本時間の日付だけで表示する。
 * 狭い一覧画面で時刻を省略する場合に使用する。
 */
export const formatDateJST = (dateString: string): string => {
  const date = parseApiDate(dateString);
  if (!date) return '';

  return date.toLocaleDateString('ja-JP', {
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  });
};

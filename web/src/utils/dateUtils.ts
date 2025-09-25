/**
 * 日付・時刻フォーマット用のユーティリティ関数
 */

/**
 * UTC時刻文字列を日本時間でフォーマットして表示
 * @param dateString UTC時刻文字列（タイムゾーン情報の有無は問わない）
 * @returns 日本時間でフォーマットされた文字列
 */
export const formatDateTimeJST = (dateString: string): string => {
  if (!dateString) return '';
  
  // タイムゾーン情報の正確な検出
  const hasZ = dateString.endsWith('Z');
  const hasTimezoneOffset = /[+-]\d{2}:?\d{2}$/.test(dateString); // +09:00 や -05:00 の形式
  
  // UTC時刻として解釈（タイムゾーン情報がない場合は強制的にUTCとして扱う）
  let date: Date;
  if (hasZ || hasTimezoneOffset) {
    // タイムゾーン情報がある場合はそのまま使用
    date = new Date(dateString);
  } else {
    // タイムゾーン情報がない場合は強制的にUTCとして扱う
    date = new Date(dateString + 'Z');
  }
  
  // 無効な日付の場合は空文字を返す
  if (isNaN(date.getTime())) {
    return '';
  }
  
  // 日本時間で表示
  return date.toLocaleString('ja-JP', {
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

/**
 * UTC時刻文字列を日本時間の日付のみでフォーマット
 * @param dateString UTC時刻文字列
 * @returns 日本時間の日付文字列（例: "2025/07/31"）
 */
export const formatDateJST = (dateString: string): string => {
  if (!dateString) return '';
  
  let date: Date;
  if (dateString.includes('Z') || dateString.includes('+') || dateString.includes('-')) {
    date = new Date(dateString);
  } else {
    date = new Date(dateString + 'Z');
  }
  
  if (isNaN(date.getTime())) {
    return '';
  }
  
  return date.toLocaleDateString('ja-JP', {
    timeZone: 'Asia/Tokyo',
    year: 'numeric',
    month: '2-digit',
    day: '2-digit'
  });
};

/**
 * UTC時刻文字列を日本時間の時刻のみでフォーマット
 * @param dateString UTC時刻文字列
 * @returns 日本時間の時刻文字列（例: "15:42:49"）
 */
export const formatTimeJST = (dateString: string): string => {
  if (!dateString) return '';
  
  let date: Date;
  if (dateString.includes('Z') || dateString.includes('+') || dateString.includes('-')) {
    date = new Date(dateString);
  } else {
    date = new Date(dateString + 'Z');
  }
  
  if (isNaN(date.getTime())) {
    return '';
  }
  
  return date.toLocaleTimeString('ja-JP', {
    timeZone: 'Asia/Tokyo',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
};

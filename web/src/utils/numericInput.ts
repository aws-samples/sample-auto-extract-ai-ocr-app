/**
 * number 型フィールドの入力を数字だけに制約するためのユーティリティ。
 *
 * number 型は「数字だけに制約された文字列」（値は文字列で保存し、先頭ゼロも保持する）。
 * 金額(479,520円)・割合(8%)・電話/郵便番号など区切りや単位を持つ値は string 型を使う想定。
 */

// 最終的に有効な数値文字列: 任意の先頭マイナス1つ・数字・任意の小数点1つ。
// 空文字は「未検出」として別途許可する。
export const NUMERIC_RE = /^-?\d+(\.\d+)?$/;

// 全角数字（０-９）を半角へ。日本語帳票の OCR 値や IME 入力に頻出するため、
// 捨てずに正規化する（捨てると値が消えてしまう）。
function toHalfWidthDigits(s: string): string {
  return s.replace(/[０-９]/g, (c) => String.fromCharCode(c.charCodeAt(0) - 0xFEE0));
}

/**
 * 入力途中の文字列から数値として不正な文字を除去する（onChange 用）。
 * "-" や "1." のような入力途中の状態は許容し、ユーザーが打ち進められるようにする。
 * - 全角数字は半角に正規化
 * - 数字を残す
 * - 先頭のマイナスは1つだけ残す
 * - 小数点は最初の1つだけ残す
 */
export function sanitizeNumericInput(raw: string): string {
  if (!raw) return '';
  const normalized = toHalfWidthDigits(raw);
  const negative = normalized[0] === '-';
  let seenDot = false;
  let body = '';
  for (const ch of normalized) {
    if (ch >= '0' && ch <= '9') {
      body += ch;
    } else if (ch === '.' && !seenDot) {
      seenDot = true;
      body += ch;
    }
    // それ以外（カンマ・単位・文字・2つ目の '.' 等）は捨てる
  }
  return negative ? `-${body}` : body;
}

/**
 * blur 時に確定形へ寄せる。入力途中で許した中間状態（"1." "-" "-.5" ".5" など）を、
 * NUMERIC_RE を満たす形に整える。整えられない残骸は空文字にする。
 */
export function finalizeNumericInput(raw: string): string {
  const v = sanitizeNumericInput(raw);
  if (v === '' || NUMERIC_RE.test(v)) return v === '' ? '' : v;
  // ここに来るのは "1." "-" "." "-.5" ".5" など。末尾/先頭の孤立記号を補正する。
  const negative = v[0] === '-';
  let digits = v.replace('-', '');
  if (digits.startsWith('.')) digits = `0${digits}`;   // ".5" -> "0.5"
  if (digits.endsWith('.')) digits = digits.slice(0, -1); // "1." -> "1"
  if (digits === '' || digits === '.') return '';         // 数字が無い残骸
  const result = negative ? `-${digits}` : digits;
  return NUMERIC_RE.test(result) ? result : '';
}

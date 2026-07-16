/**
 * スキーマ保存前のクライアント側検証。
 * バックエンド（lambda/api/app/schemas/schema.py の pydantic validators）と同じ
 * 意味的ルールをミラーし、保存前に日本語メッセージで問題を洗い出す。
 * バックが最終防御（422）なので、ここは UX 向上のための事前チェック。
 */
import { Field } from '../types/app-schema';

// フィールド名 / アプリ名に許可する文字。英数字とアンダースコアのみ。
// keep in sync: lambda/api/app/schemas/schema.py NAME_PATTERN
export const NAME_PATTERN = /^[a-zA-Z0-9_]+$/;

/** アプリ名の形式チェック（バックの NAME_PATTERN と同期）。 */
export function isValidAppName(name: string): boolean {
  return NAME_PATTERN.test(name);
}

// 同一階層のフィールド名重複を検出してメッセージを push する。
function checkSiblingNames(fields: Field[], errors: string[]): void {
  const seen = new Set<string>();
  for (const f of fields) {
    if (seen.has(f.name)) {
      errors.push(`同じ階層でフィールド名が重複しています: ${f.name}`);
    }
    seen.add(f.name);
  }
}

// 1つの Field ノードを検証（name / display_name / 型別構造）。再帰。
function validateField(field: Field, errors: string[]): void {
  const label = field.display_name || field.name || '(名称未設定)';

  // name（rule 2）
  if (!field.name) {
    errors.push('フィールド名は必須です');
  } else if (!NAME_PATTERN.test(field.name)) {
    errors.push(`フィールド名は英数字とアンダースコアのみ使用できます: ${field.name}`);
  }
  // display_name（rule 5）
  if (!field.display_name || !field.display_name.trim()) {
    errors.push(`表示名は必須です（フィールド: ${field.name || '(名称未設定)'}）`);
  }

  // 型別構造（rule 4）
  if (field.type === 'map') {
    if (!field.fields || field.fields.length === 0) {
      errors.push(`map 型「${label}」には子フィールドが必要です`);
    } else {
      checkSiblingNames(field.fields, errors);
      field.fields.forEach((c) => validateField(c, errors));
    }
    if (field.items) {
      errors.push(`map 型「${label}」に items は指定できません`);
    }
  } else if (field.type === 'list') {
    if (!field.items) {
      errors.push(`list 型「${label}」には要素定義が必要です`);
    } else {
      validateItems(field.items, label, errors);
    }
    if (field.fields) {
      errors.push(`list 型「${label}」に fields は指定できません`);
    }
  } else {
    // string / number
    if (field.fields || field.items) {
      errors.push(`${field.type} 型「${label}」に子フィールドや要素定義は指定できません`);
    }
  }
}

// list の items（FieldItems 相当。name/display_name を持たない別形）を検証。
// 型は {type, fields?} で items キーを持たないため、list-in-list はそもそも表現不能
// （エディタも要素型から list を除外済み）。防御的にチェックのみ残す。
function validateItems(
  items: NonNullable<Field['items']>,
  parentLabel: string,
  errors: string[],
): void {
  if (items.type === 'list') {
    errors.push(`list「${parentLabel}」の要素として list 型は指定できません`);
    return;
  }
  if (items.type === 'map') {
    if (!items.fields || items.fields.length === 0) {
      errors.push(`list「${parentLabel}」の map 要素には子フィールドが必要です`);
    } else {
      checkSiblingNames(items.fields, errors);
      items.fields.forEach((c) => validateField(c, errors));
    }
  } else {
    // string / number 要素
    if (items.fields) {
      errors.push(`list「${parentLabel}」の ${items.type} 要素に子フィールドは指定できません`);
    }
  }
}

/**
 * スキーマの fields を検証し、日本語エラーメッセージの配列を返す（空配列 = 問題なし）。
 * バックの rules 1〜5 をミラー。
 */
export function validateSchemaFields(fields: Field[]): string[] {
  const errors: string[] = [];
  // rule 1: 最低1件
  if (!fields || fields.length === 0) {
    errors.push('スキーマには最低1つのフィールドが必要です');
    return errors;
  }
  // rule 3: top レベルの重複
  checkSiblingNames(fields, errors);
  // 各フィールド
  fields.forEach((f) => validateField(f, errors));
  return errors;
}

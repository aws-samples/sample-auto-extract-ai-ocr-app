import React from "react";
import { Plus, Trash2 } from "lucide-react";
import { Field } from "../../types/app-schema";

/**
 * スキーマフィールドの直接編集エディタ。
 * SchemaPreview と同じカード見た目のまま、名前・型をインライン編集できる。
 * - display_name / name のテキスト編集
 * - type の変更 (string / number / map / list)
 * - フィールドの追加・削除 (map のネスト、list items の map ネスト含む)
 */

const FIELD_TYPES = ["string", "number", "map", "list"] as const;

const getTypeClass = (type: string) => {
  switch (type) {
    case "string":
      return "bg-info-light text-info-text";
    case "number":
      return "bg-success-light text-success-text";
    case "map":
      return "bg-accent-light text-accent-text";
    case "list":
      return "bg-orange-100 text-orange-800";
    default:
      return "bg-surface-alt text-default";
  }
};

/** 新規フィールドのデフォルト値を生成 */
const createNewField = (existing: Field[]): Field => {
  // name の重複を避けるため連番を振る
  let n = existing.length + 1;
  while (existing.some((f) => f.name === `field_${n}`)) n++;
  return { name: `field_${n}`, display_name: "新しいフィールド", type: "string" };
};

/** type 変更時にフィールド構造を整合させる */
const applyTypeChange = (field: Field, newType: string): Field => {
  const base: Field = { name: field.name, display_name: field.display_name, type: newType };
  if (newType === "map") {
    base.fields = field.fields || [];
  } else if (newType === "list") {
    base.items = field.items || { type: "string" };
  }
  return base;
};

interface FieldRowProps {
  field: Field;
  level: number;
  onChange: (updated: Field) => void;
  onRemove: () => void;
}

const FieldRow: React.FC<FieldRowProps> = ({ field, level, onChange, onRemove }) => {
  const updateNested = (index: number, updated: Field) => {
    const fields = (field.fields || []).map((f, i) => (i === index ? updated : f));
    onChange({ ...field, fields });
  };

  const removeNested = (index: number) => {
    const fields = (field.fields || []).filter((_, i) => i !== index);
    onChange({ ...field, fields });
  };

  const addNested = () => {
    const fields = [...(field.fields || []), createNewField(field.fields || [])];
    onChange({ ...field, fields });
  };

  // list items 用ハンドラ
  const changeItemsType = (newType: string) => {
    const items: Field["items"] = { type: newType };
    if (newType === "map") {
      items.fields = field.items?.fields || [];
    }
    onChange({ ...field, items });
  };

  const updateItemsField = (index: number, updated: Field) => {
    const fields = (field.items?.fields || []).map((f, i) => (i === index ? updated : f));
    onChange({ ...field, items: { ...(field.items as NonNullable<Field["items"]>), fields } });
  };

  const removeItemsField = (index: number) => {
    const fields = (field.items?.fields || []).filter((_, i) => i !== index);
    onChange({ ...field, items: { ...(field.items as NonNullable<Field["items"]>), fields } });
  };

  const addItemsField = () => {
    const fields = [...(field.items?.fields || []), createNewField(field.items?.fields || [])];
    onChange({ ...field, items: { ...(field.items as NonNullable<Field["items"]>), fields } });
  };

  return (
    <div
      className={`border rounded-md p-3 mb-3 ${level > 0 ? "border-l-2 border-neutral-200" : ""}`}
    >
      <div className="flex items-center gap-2 flex-wrap">
        <input
          type="text"
          value={field.display_name}
          onChange={(e) => onChange({ ...field, display_name: e.target.value })}
          className="flex-1 min-w-[8rem] border border-default rounded-lg px-2.5 py-1.5 text-sm font-medium bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="表示名"
          aria-label="表示名"
        />
        <input
          type="text"
          value={field.name}
          onChange={(e) => onChange({ ...field, name: e.target.value })}
          className="w-36 border border-default rounded-lg px-2.5 py-1.5 text-sm font-mono text-muted bg-bg focus:outline-none focus:ring-2 focus:ring-primary"
          placeholder="field_name"
          aria-label="フィールド名（英数字）"
        />
        <select
          value={field.type}
          onChange={(e) => onChange(applyTypeChange(field, e.target.value))}
          className={`border border-default rounded-lg px-2.5 py-1.5 text-xs font-medium focus:outline-none focus:ring-2 focus:ring-primary ${getTypeClass(field.type)}`}
          aria-label="型"
        >
          {FIELD_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        <button
          onClick={onRemove}
          className="p-1.5 rounded text-danger hover:bg-danger-light"
          title="フィールドを削除"
          aria-label="フィールドを削除"
        >
          <Trash2 size={16} />
        </button>
      </div>

      {/* map 型: ネストフィールド */}
      {field.type === "map" && (
        <div className="mt-2 pl-4 border-l-2 border-neutral-200">
          {(field.fields || []).map((child, i) => (
            <FieldRow
              key={i}
              field={child}
              level={level + 1}
              onChange={(updated) => updateNested(i, updated)}
              onRemove={() => removeNested(i)}
            />
          ))}
          <button
            onClick={addNested}
            className="flex items-center gap-1 text-sm text-primary hover:text-primary-hover py-1"
          >
            <Plus size={14} />
            サブフィールドを追加
          </button>
        </div>
      )}

      {/* list 型: 項目定義 */}
      {field.type === "list" && field.items && (
        <div className="mt-2 pl-4 border-l-2 border-neutral-200">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">リスト項目</span>
            <select
              value={field.items.type}
              onChange={(e) => changeItemsType(e.target.value)}
              className={`border border-default rounded-lg px-2.5 py-1.5 text-xs font-medium focus:outline-none focus:ring-2 focus:ring-primary ${getTypeClass(field.items.type)}`}
              aria-label="リスト項目の型"
            >
              {FIELD_TYPES.filter((t) => t !== "list").map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
          </div>

          {/* list 内 map 型のネストフィールド */}
          {field.items.type === "map" && (
            <div className="mt-2 pl-4 border-l-2 border-neutral-200">
              {(field.items.fields || []).map((child, i) => (
                <FieldRow
                  key={i}
                  field={child}
                  level={level + 2}
                  onChange={(updated) => updateItemsField(i, updated)}
                  onRemove={() => removeItemsField(i)}
                />
              ))}
              <button
                onClick={addItemsField}
                className="flex items-center gap-1 text-sm text-primary hover:text-primary-hover py-1"
              >
                <Plus size={14} />
                サブフィールドを追加
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface SchemaFieldsEditorProps {
  fields: Field[];
  onChange: (fields: Field[]) => void;
}

const SchemaFieldsEditor: React.FC<SchemaFieldsEditorProps> = ({ fields, onChange }) => {
  const updateAt = (index: number, updated: Field) => {
    onChange(fields.map((f, i) => (i === index ? updated : f)));
  };

  const removeAt = (index: number) => {
    onChange(fields.filter((_, i) => i !== index));
  };

  const add = () => {
    onChange([...fields, createNewField(fields)]);
  };

  return (
    <div>
      <div className="space-y-2">
        {fields.map((field, i) => (
          <FieldRow
            key={i}
            field={field}
            level={0}
            onChange={(updated) => updateAt(i, updated)}
            onRemove={() => removeAt(i)}
          />
        ))}
      </div>
      <button
        onClick={add}
        className="flex items-center gap-1 text-sm text-primary hover:text-primary-hover py-2"
      >
        <Plus size={16} />
        フィールドを追加
      </button>
    </div>
  );
};

export default SchemaFieldsEditor;

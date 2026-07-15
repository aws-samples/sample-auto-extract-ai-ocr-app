import React, { useState, useEffect, useRef } from 'react';
import { Trash2 } from 'lucide-react';
import { Field } from '../../types/app-schema';
import { Suggestion } from '../../types/agent';
import { Button } from '../../components/ui';

interface ExtractedInfoDisplayProps {
  extractedInfo: Record<string, any>;
  fields: Field[];
  editMode: boolean;
  onHighlightField: (field: string, stayOnExtractionView?: boolean) => void;
  onHighlightCell: (fieldName: string, rowIndex: number, columnName: string) => void;
  onUpdateExtractedInfo: (info: Record<string, any>) => void;
  agentSuggestions?: Suggestion[];
  onAcceptSuggestion?: (suggestion: Suggestion) => void;
  onRejectSuggestion?: (suggestion: Suggestion) => void;
  onEnterEditMode?: () => void;
}

const ExtractedInfoDisplay: React.FC<ExtractedInfoDisplayProps> = ({
  extractedInfo,
  fields,
  editMode,
  onHighlightField,
  onHighlightCell,
  onUpdateExtractedInfo,
  agentSuggestions: externalSuggestions = [],
  onAcceptSuggestion,
  onRejectSuggestion,
  onEnterEditMode,
}) => {
  const [editedInfo, setEditedInfo] = useState<Record<string, any>>(extractedInfo);
  const [expandedSuggestionRow, setExpandedSuggestionRow] = useState<string | null>(null);
  const [editingCell, setEditingCell] = useState<string | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!editMode) {
      setEditedInfo({ ...extractedInfo });
    }
  }, [extractedInfo, editMode]);

  const updateFieldValue = (fieldName: string, value: any) => {
    setEditedInfo(prev => {
      const next = { ...prev, [fieldName]: value };
      onUpdateExtractedInfo(next);
      return next;
    });
  };


  const updateListItemProperty = (fieldName: string, itemIndex: number, propertyName: string, value: any) => {
    setEditedInfo(prev => {
      const currentList = [...(prev[fieldName] || [])];
      currentList[itemIndex] = { ...(currentList[itemIndex] || {}), [propertyName]: value };
      const next = { ...prev, [fieldName]: currentList };
      onUpdateExtractedInfo(next);
      return next;
    });
  };

  const setNestedValue = (obj: any, path: string, value: any): any => {
    const tokens: (string | number)[] = [];
    const re = /(\w+)|\[(\d+)\]/g;
    let m: RegExpExecArray | null;
    while ((m = re.exec(path)) !== null) {
      tokens.push(m[2] !== undefined ? parseInt(m[2], 10) : m[1]);
    }
    if (tokens.length === 0) return obj;

    const root = Array.isArray(obj) ? [...obj] : { ...obj };
    let cur: any = root;
    for (let i = 0; i < tokens.length - 1; i++) {
      const key = tokens[i];
      const nextKey = tokens[i + 1];
      const nextIsArray = typeof nextKey === 'number';
      if (Array.isArray(cur[key])) {
        cur[key] = [...cur[key]];
      } else if (cur[key] && typeof cur[key] === 'object') {
        cur[key] = { ...cur[key] };
      } else {
        cur[key] = nextIsArray ? [] : {};
      }
      cur = cur[key];
    }
    cur[tokens[tokens.length - 1]] = value;
    return root;
  };

  const handleAcceptSuggestion = (suggestion: Suggestion) => {
    if (!editMode && onEnterEditMode) {
      onEnterEditMode();
    }
    const newInfo = setNestedValue(editedInfo, suggestion.field, suggestion.suggested_value);
    setEditedInfo(newInfo);
    onUpdateExtractedInfo(newInfo);
    setExpandedSuggestionRow(null);
    onAcceptSuggestion?.(suggestion);
  };

  const handleRejectSuggestion = (suggestion: Suggestion) => {
    if (!editMode && onEnterEditMode) {
      onEnterEditMode();
    }
    setExpandedSuggestionRow(null);
    onRejectSuggestion?.(suggestion);
  };

  const getSuggestionForField = (fieldName: string) =>
    externalSuggestions.find(s => s.field === fieldName);

  const getSuggestionsForListRow = (listFieldName: string, rowIndex: number) =>
    externalSuggestions.filter(s => {
      const prefix = `${listFieldName}[${rowIndex}]`;
      return s.field === prefix || s.field.startsWith(`${prefix}.`);
    });

  const renderSuggestion = (suggestion: Suggestion) => (
    <div className="mt-2 p-3 bg-warning-light border border-warning-border rounded">
      <div className="text-sm mb-2">
        <div className="mb-1">{suggestion.reason}</div>
        <div className="text-xs text-neutral-600">
          「{suggestion.original_value}」→「{suggestion.suggested_value}」
          {suggestion.tool_used && <span className="ml-2">({suggestion.tool_used})</span>}
        </div>
      </div>
      <div className="flex gap-2">
        <Button variant="primary" size="sm" onClick={() => handleAcceptSuggestion(suggestion)}>採用する</Button>
        <Button variant="outline" size="sm" onClick={() => handleRejectSuggestion(suggestion)}>却下</Button>
      </div>
    </div>
  );

  const renderStringField = (field: Field) => {
    const rawValue = editMode ? editedInfo[field.name] : extractedInfo[field.name];
    // スキーマ変更で旧値が object/array の場合、React child に描画できず落ちるため文字列に丸める
    const value = (rawValue !== null && typeof rawValue === 'object') ? JSON.stringify(rawValue) : rawValue;
    const suggestion = getSuggestionForField(field.name);
    return (
      <div key={field.name} className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <label className="block text-sm font-medium text-neutral-700">
            {field.display_name} {suggestion && <span className="text-warning-text">⚠</span>}
          </label>
        </div>
        {editMode ? (
          <input
            type="text"
            value={value || ''}
            onChange={(e) => updateFieldValue(field.name, e.target.value)}
            onFocus={() => onHighlightField(field.name, true)}
            className="w-full p-2 border border-neutral-300 rounded"
          />
        ) : (
          <div
            className="p-2 bg-neutral-50 border border-neutral-200 rounded cursor-pointer hover:bg-neutral-100 min-h-[2.5rem]"
            onClick={() => onHighlightField(field.name, true)}
          >
            {value || ''}
          </div>
        )}
        {suggestion && renderSuggestion(suggestion)}
      </div>
    );
  };

  const getValueAtPath = (path: string): any => {
    const source = editMode ? editedInfo : extractedInfo;
    return path.split('.').reduce((cur, key) => cur?.[key], source);
  };

  const setValueAtPath = (path: string, value: any) => {
    const newInfo = setNestedValue(editedInfo, path, value);
    setEditedInfo(newInfo);
    onUpdateExtractedInfo(newInfo);
  };

  const renderMapField = (field: Field, parentPath?: string) => {
    if (!field.fields) return null;
    const basePath = parentPath ? `${parentPath}.${field.name}` : field.name;
    const rawMapValue = getValueAtPath(basePath);
    // スキーマ変更で旧値が map と合わない（例: 文字列）場合のフォールバック
    const mapValue = (rawMapValue && typeof rawMapValue === 'object' && !Array.isArray(rawMapValue)) ? rawMapValue : {};

    return (
      <div key={field.name} className="mb-6">
        <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
        <div className="pl-4 border-l-2 border-neutral-200 space-y-3">
          {field.fields.map(subField => {
            const fieldPath = `${basePath}.${subField.name}`;

            if (subField.type === 'map' && subField.fields) {
              return renderMapField(subField, basePath);
            }
            if (subField.type === 'list' && subField.items) {
              return renderNestedListField(subField, basePath);
            }

            const suggestion = getSuggestionForField(fieldPath);
            return (
              <div key={subField.name} className="mb-3">
                <div className="flex justify-between items-center mb-1">
                  <label className="block text-sm font-medium text-neutral-700">
                    {subField.display_name} {suggestion && <span className="text-warning-text">⚠</span>}
                  </label>
                </div>
                {editMode ? (
                  <input
                    type="text"
                    value={mapValue[subField.name] || ''}
                    onChange={(e) => setValueAtPath(fieldPath, e.target.value)}
                    onFocus={() => onHighlightField(fieldPath, true)}
                    className="w-full p-2 border border-neutral-300 rounded"
                  />
                ) : (
                  <div
                    className="p-2 bg-neutral-50 border border-neutral-200 rounded cursor-pointer hover:bg-neutral-100 min-h-[2.5rem]"
                    onClick={() => onHighlightField(fieldPath, true)}
                  >
                    {mapValue[subField.name] || ''}
                  </div>
                )}
                {suggestion && renderSuggestion(suggestion)}
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  const renderNestedListField = (field: Field, parentPath: string) => {
    if (!field.items) return null;
    const basePath = `${parentPath}.${field.name}`;
    const rawListData = getValueAtPath(basePath);
    const listData: any[] = Array.isArray(rawListData) ? rawListData : [];

    return (
      <div key={field.name} className="mb-4">
        <label className="block text-sm font-medium text-neutral-700 mb-1">{field.display_name}</label>
        <ul className="list-disc pl-5">
          {listData.map((item: any, i: number) => (
            <li key={i} className="mb-1">
              {editMode ? (
                <input
                  type="text"
                  value={item || ''}
                  onChange={(e) => {
                    const updated = [...listData];
                    updated[i] = e.target.value;
                    setValueAtPath(basePath, updated);
                  }}
                  className="w-full p-1 border border-neutral-300 rounded text-sm"
                />
              ) : (
                <span className="text-sm">{item || ''}</span>
              )}
            </li>
          ))}
        </ul>
      </div>
    );
  };

  const renderListField = (field: Field) => {
    if (!field.items) return null;
    const rawListData = editMode ? editedInfo[field.name] : extractedInfo[field.name];
    // スキーマ変更で旧値の型が list と合わない（例: 文字列）場合に .map で落ちるのを防ぐ
    const listData: any[] = Array.isArray(rawListData) ? rawListData : [];

    if (field.items.type === 'map' && field.items.fields) {
      const itemFields = field.items.fields;
      const colCount = itemFields.length;

      return (
        <div key={field.name} className="mb-6">
          <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
          <div className="overflow-x-auto">
            {/*
              table-fixed + w-full + colgroup(%) は列数が多い表で
              各列が極狭になり文字が縦に潰れるため使わない。
              min-w-full + table-auto にして内容に応じた自然な列幅とし、
              列数が多い場合は親の overflow-x-auto で横スクロールさせる。
            */}
            <table className="min-w-full table-auto divide-y divide-gray-200">
              <thead className="bg-neutral-50">
                <tr>
                  {itemFields.map(itemField => (
                    <th key={itemField.name} className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider whitespace-nowrap">
                      {itemField.display_name}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="bg-bg divide-y divide-gray-200">
                {listData.map((rawItem: any, itemIndex: number) => {
                  // 旧データの行が null/非オブジェクトでもセル参照で落ちないようにする
                  const item = (rawItem !== null && typeof rawItem === 'object') ? rawItem : {};
                  const rowSuggestions = getSuggestionsForListRow(field.name, itemIndex);
                  const rowKey = `${field.name}[${itemIndex}]`;
                  const isExpanded = expandedSuggestionRow === rowKey;

                  return (
                    <React.Fragment key={itemIndex}>
                      <tr className={`group ${rowSuggestions.length > 0 ? 'bg-warning-light/30' : ''}`}>
                        {itemFields.map((itemField, colIdx) => {
                          const cellPath = `${field.name}[${itemIndex}].${itemField.name}`;
                          const cellSuggestion = getSuggestionForField(cellPath);
                          const isEditing = editingCell === cellPath;
                          const isLastCol = colIdx === itemFields.length - 1;

                          return (
                            <td key={itemField.name} className="px-3 py-1.5 relative">
                              {editMode && isEditing ? (
                                <input
                                  ref={editInputRef}
                                  type="text"
                                  value={item[itemField.name] || ''}
                                  onChange={(e) => updateListItemProperty(field.name, itemIndex, itemField.name, e.target.value)}
                                  onFocus={() => onHighlightCell(field.name, itemIndex, itemField.name)}
                                  onBlur={() => setEditingCell(null)}
                                  onKeyDown={(e) => { if (e.key === 'Escape' || e.key === 'Enter') setEditingCell(null); }}
                                  className={`w-full px-2 py-1 border rounded text-sm focus:outline-none focus:ring-1 focus:ring-info ${cellSuggestion ? 'border-warning-border bg-warning-light/50' : 'border-neutral-300'}`}
                                />
                              ) : (
                                <div
                                  // 空セル（行追加直後など）でも最低1行分の高さを確保し、
                                  // 極薄の行になって気づけない問題を防ぐ（min-h のみ、幅・枠は変えない）
                                  className={`text-sm text-neutral-900 px-2 py-1 rounded max-w-xs break-words cursor-pointer min-h-[1.5rem] ${cellSuggestion ? 'text-warning-text font-medium hover:bg-warning-light/50' : 'hover:bg-info-light'}`}
                                  onClick={() => {
                                    onHighlightCell(field.name, itemIndex, itemField.name);
                                    if (editMode) {
                                      setEditingCell(cellPath);
                                      setTimeout(() => editInputRef.current?.focus(), 0);
                                    }
                                    if (rowSuggestions.length > 0) {
                                      setExpandedSuggestionRow(isExpanded ? null : rowKey);
                                    }
                                  }}
                                >
                                  {item[itemField.name] || ''}
                                  {cellSuggestion && <span className="ml-1">⚠</span>}
                                </div>
                              )}
                              {editMode && isLastCol && (
                                <button
                                  type="button"
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    const updated = [...listData];
                                    updated.splice(itemIndex, 1);
                                    updateFieldValue(field.name, updated);
                                  }}
                                  className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity text-neutral-400 hover:text-danger p-1"
                                  title="行を削除"
                                >
                                  <Trash2 size={14} />
                                </button>
                              )}
                            </td>
                          );
                        })}
                      </tr>
                      {isExpanded && rowSuggestions.length > 0 && (
                        <tr>
                          <td colSpan={colCount} className="px-3 py-2 bg-warning-light/20">
                            <div className="space-y-2">
                              {rowSuggestions.map((suggestion, i) => (
                                <div key={i} className="p-3 bg-warning-light border border-warning-border rounded">
                                  <div className="text-sm mb-2">
                                    <div className="mb-1">{suggestion.reason}</div>
                                    <div className="text-xs text-neutral-600">
                                      「{suggestion.original_value}」→「{suggestion.suggested_value}」
                                      {suggestion.tool_used && <span className="ml-2">({suggestion.tool_used})</span>}
                                    </div>
                                  </div>
                                  <div className="flex gap-2">
                                    <Button variant="primary" size="sm" onClick={() => handleAcceptSuggestion(suggestion)}>採用する</Button>
                                    <Button variant="outline" size="sm" onClick={() => handleRejectSuggestion(suggestion)}>却下</Button>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          {editMode && (
            <button type="button" onClick={() => {
              const newItem: Record<string, string> = {};
              field.items!.fields!.forEach(f => { newItem[f.name] = ''; });
              updateFieldValue(field.name, [...listData, newItem]);
            }} className="mt-2 text-info hover:text-info-text text-sm">+ 行を追加</button>
          )}
        </div>
      );
    }

    return (
      <div key={field.name} className="mb-6">
        <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
        <ul className="list-disc pl-5">
          {listData.map((item: any, i: number) => (
            <li key={i} className="mb-2">
              {editMode ? (
                <input type="text" value={item || ''} onChange={(e) => {
                  const updated = [...listData];
                  updated[i] = e.target.value;
                  updateFieldValue(field.name, updated);
                }} className="w-full p-1 border border-neutral-300 rounded" />
              ) : (
                <div className="p-1">{item || ''}</div>
              )}
            </li>
          ))}
        </ul>
        {editMode && (
          <button type="button" onClick={() => updateFieldValue(field.name, [...listData, ''])} className="mt-2 text-info hover:text-info-text">+ 項目を追加</button>
        )}
      </div>
    );
  };

  const renderField = (field: Field) => {
    if (field.type === 'map' && field.fields) return renderMapField(field);
    if (field.type === 'list' && field.items) return renderListField(field);
    return renderStringField(field);
  };

  return (
    <div className="bg-bg rounded-lg border border-neutral-200 p-4">
      <div className="space-y-4">
        {fields.map(field => renderField(field))}
      </div>
    </div>
  );
};

export default ExtractedInfoDisplay;

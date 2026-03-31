import React, { useState, useEffect } from 'react';
import { Field } from '../../types/app-schema';
import { Suggestion, Tool } from '../../types/agent';
import { isAgentEnabled } from '../../config';
import { Modal, Button } from '../../components/ui';

interface ExtractedInfoDisplayProps {
  extractedInfo: Record<string, any>;
  fields: Field[];
  editMode: boolean;
  onHighlightField: (field: string, stayOnExtractionView?: boolean) => void;
  onHighlightCell: (fieldName: string, rowIndex: number, columnName: string) => void;
  onUpdateExtractedInfo: (info: Record<string, any>) => void;
  onRunAgent?: () => Promise<Suggestion[]>;
  agentStatus?: 'idle' | 'running' | 'completed';
  onGetTools?: () => Promise<Tool[]>;
}

const ExtractedInfoDisplay: React.FC<ExtractedInfoDisplayProps> = ({
  extractedInfo,
  fields,
  editMode,
  onHighlightField,
  onHighlightCell,
  onUpdateExtractedInfo,
  onRunAgent,
  agentStatus = 'idle',
  onGetTools,
}) => {
  const [editedInfo, setEditedInfo] = useState<Record<string, any>>(extractedInfo);
  const [agentSuggestions, setAgentSuggestions] = useState<Suggestion[]>([]);
  const [showToolsModal, setShowToolsModal] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);

  useEffect(() => {
    if (!editMode) {
      setEditedInfo({ ...extractedInfo });
    }
  }, [extractedInfo, editMode]);

  useEffect(() => {
    if (onGetTools) {
      onGetTools().then(setTools).catch(() => {});
    }
  }, [onGetTools]);

  const updateFieldValue = (fieldName: string, value: any) => {
    setEditedInfo(prev => {
      const next = { ...prev, [fieldName]: value };
      onUpdateExtractedInfo(next);
      return next;
    });
  };

  const updateMapFieldValue = (fieldName: string, subFieldName: string, value: any) => {
    setEditedInfo(prev => {
      const next = {
        ...prev,
        [fieldName]: { ...(prev[fieldName] || {}), [subFieldName]: value }
      };
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

  const handleRunAgent = async () => {
    if (!onRunAgent) return;
    try {
      const suggestions = await onRunAgent();
      setAgentSuggestions(suggestions);
    } catch (error) {
      console.error('エージェント実行エラー:', error);
    }
  };

  const handleAcceptSuggestion = (suggestion: Suggestion) => {
    let newInfo = { ...editedInfo };
    const fieldPath = suggestion.field.split('.');
    if (fieldPath.length === 1) {
      newInfo[fieldPath[0]] = suggestion.suggested_value;
    } else if (fieldPath.length === 2) {
      newInfo[fieldPath[0]] = { ...(newInfo[fieldPath[0]] || {}), [fieldPath[1]]: suggestion.suggested_value };
    }
    setEditedInfo(newInfo);
    onUpdateExtractedInfo(newInfo);
    setAgentSuggestions(prev => prev.filter(s => s.field !== suggestion.field));
  };

  const handleRejectSuggestion = (suggestion: Suggestion) => {
    setAgentSuggestions(prev => prev.filter(s => s.field !== suggestion.field));
  };

  const getSuggestionForField = (fieldName: string) => agentSuggestions.find(s => s.field === fieldName);

  const renderSuggestion = (suggestion: Suggestion) => (
    <div className="mt-2 p-3 bg-warning-light border border-warning-border rounded">
      <div className="text-sm mb-2">
        {suggestion.tool_used && <div className="font-semibold text-warning-text mb-1">{suggestion.tool_used}経由で確認済み</div>}
        <div className="mb-1">「{suggestion.original_value}」→「{suggestion.suggested_value}」の表記ゆれを検出</div>
      </div>
      <div className="flex gap-2">
        <Button variant="primary" size="sm" onClick={() => handleAcceptSuggestion(suggestion)}>採用する</Button>
        <Button variant="secondary" size="sm" onClick={() => handleRejectSuggestion(suggestion)}>却下</Button>
      </div>
    </div>
  );

  const renderStringField = (field: Field) => {
    const value = editMode ? editedInfo[field.name] : extractedInfo[field.name];
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
            className="p-2 bg-neutral-50 border border-neutral-200 rounded cursor-pointer hover:bg-neutral-100"
            onClick={() => onHighlightField(field.name, true)}
          >
            {value || '(抽出されませんでした)'}
          </div>
        )}
        {suggestion && renderSuggestion(suggestion)}
      </div>
    );
  };

  const renderMapField = (field: Field) => {
    if (!field.fields) return null;
    const mapValue = editMode ? editedInfo[field.name] || {} : extractedInfo[field.name] || {};
    return (
      <div key={field.name} className="mb-6">
        <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
        <div className="pl-4 border-l-2 border-neutral-200 space-y-3">
          {field.fields.map(subField => {
            const fieldPath = `${field.name}.${subField.name}`;
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
                    onChange={(e) => updateMapFieldValue(field.name, subField.name, e.target.value)}
                    onFocus={() => onHighlightField(fieldPath, true)}
                    className="w-full p-2 border border-neutral-300 rounded"
                  />
                ) : (
                  <div
                    className="p-2 bg-neutral-50 border border-neutral-200 rounded cursor-pointer hover:bg-neutral-100"
                    onClick={() => onHighlightField(fieldPath, true)}
                  >
                    {mapValue[subField.name] || '(抽出されませんでした)'}
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

  const renderListField = (field: Field) => {
    if (!field.items) return null;
    const listData = editMode ? editedInfo[field.name] || [] : extractedInfo[field.name] || [];

    if (field.items.type === 'map' && field.items.fields) {
      return (
        <div key={field.name} className="mb-6">
          <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-neutral-50">
                <tr>
                  {field.items.fields.map(itemField => (
                    <th key={itemField.name} className={`text-left text-xs font-medium text-neutral-500 uppercase tracking-wider ${editMode ? 'px-3 py-2' : 'px-6 py-3'}`}>
                      {itemField.display_name}
                    </th>
                  ))}
                  {editMode && (
                    <th className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">操作</th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-bg divide-y divide-gray-200">
                {listData.map((item: any, itemIndex: number) => (
                  <tr key={itemIndex}>
                    {field.items!.fields!.map(itemField => (
                      <td key={itemField.name} className={editMode ? "px-3 py-2" : "px-6 py-4 whitespace-nowrap"}>
                        {editMode ? (
                          <input
                            type="text"
                            value={item[itemField.name] || ''}
                            onChange={(e) => updateListItemProperty(field.name, itemIndex, itemField.name, e.target.value)}
                            onFocus={() => onHighlightCell(field.name, itemIndex, itemField.name)}
                            className="w-full p-1 border border-neutral-300 rounded"
                          />
                        ) : (
                          <div
                            className="text-sm text-neutral-900 cursor-pointer hover:bg-info-light p-1 rounded"
                            onClick={() => onHighlightCell(field.name, itemIndex, itemField.name)}
                          >
                            {item[itemField.name] || ''}
                          </div>
                        )}
                      </td>
                    ))}
                    {editMode && (
                      <td className="px-3 py-2 whitespace-nowrap">
                        <button type="button" onClick={() => {
                          const updated = [...listData];
                          updated.splice(itemIndex, 1);
                          updateFieldValue(field.name, updated);
                        }} className="text-danger hover:text-danger-text">削除</button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {editMode && (
            <button type="button" onClick={() => {
              const newItem: Record<string, string> = {};
              field.items!.fields!.forEach(f => { newItem[f.name] = ''; });
              updateFieldValue(field.name, [...listData, newItem]);
            }} className="mt-2 text-info hover:text-info-text">+ 行を追加</button>
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
      {/* エージェント */}
      {onRunAgent && isAgentEnabled() && (
        <div className="flex items-center justify-end mb-4 gap-2">
          <Button variant="secondary" size="sm" onClick={() => setShowToolsModal(true)}>登録ツール</Button>
          <Button variant="secondary" size="sm" onClick={handleRunAgent} disabled={agentStatus === 'running'}>
            {agentStatus === 'running' ? '検証中...' : 'エージェント検証'}
          </Button>
        </div>
      )}

      <div className="space-y-4">
        {fields.map(field => renderField(field))}
      </div>

      <Modal isOpen={showToolsModal} onClose={() => setShowToolsModal(false)} className="max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6">
        <h2 className="text-lg font-semibold mb-4">登録ツール一覧</h2>
        <div className="space-y-3">
          {tools.map((tool, index) => (
            <div key={index} className="p-3 border border-neutral-200 rounded">
              <div className="font-semibold text-neutral-800">{tool.name}</div>
              <div className="text-sm text-neutral-600 mt-1">{tool.description}</div>
            </div>
          ))}
        </div>
        <div className="flex justify-end mt-4">
          <Button variant="secondary" size="sm" onClick={() => setShowToolsModal(false)}>閉じる</Button>
        </div>
      </Modal>
    </div>
  );
};

export default ExtractedInfoDisplay;

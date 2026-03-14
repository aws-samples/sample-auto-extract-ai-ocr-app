import React, { useState, useEffect } from 'react';
import { Field } from '../types/app-schema';
import { Suggestion, Tool } from '../types/agent';
import { isAgentEnabled } from '../config';
import { Modal } from './ui';

interface ExtractedInfoDisplayProps {
  extractedInfo: Record<string, any>;
  fields: Field[];
  onSave: (data?: Record<string, any>) => void;
  onHighlightField: (field: string, stayOnExtractionView?: boolean) => void;
  onHighlightCell: (fieldName: string, rowIndex: number, columnName: string) => void;
  onUpdateExtractedInfo: (info: Record<string, any>) => void;
  onRunAgent?: () => Promise<Suggestion[]>;
  agentStatus?: 'idle' | 'running' | 'completed';
  onGetTools?: () => Promise<Tool[]>;
  activeView?: 'ocr' | 'extraction';
  onBackToExtraction?: () => void;
  onViewOcr?: () => void;
  isOcrEnabled?: boolean;
  verificationCompleted?: boolean;
  onVerificationChange?: (completed: boolean) => void;
}

const ExtractedInfoDisplay: React.FC<ExtractedInfoDisplayProps> = ({
  extractedInfo,
  fields,
  onSave,
  onHighlightField,
  onHighlightCell,
  onUpdateExtractedInfo,
  onRunAgent,
  agentStatus = 'idle',
  onGetTools,
  activeView = 'extraction',
  onBackToExtraction,
  onViewOcr,
  isOcrEnabled = false,
  verificationCompleted = false,
  onVerificationChange,
}) => {
  const [editMode, setEditMode] = useState(false);
  const [editedInfo, setEditedInfo] = useState<Record<string, any>>(extractedInfo);
  const [originalInfo, setOriginalInfo] = useState<Record<string, any>>(extractedInfo);
  const [agentSuggestions, setAgentSuggestions] = useState<Suggestion[]>([]);
  const [showToolsModal, setShowToolsModal] = useState(false);
  const [tools, setTools] = useState<Tool[]>([]);

  // 外部からのextractedInfoの変更を監視して、編集中でなければ更新
  useEffect(() => {
    if (!editMode) {
      setEditedInfo({...extractedInfo});
      setOriginalInfo({...extractedInfo});
    }
  }, [extractedInfo, editMode]);

  // ページ読み込み時にツール一覧を取得
  useEffect(() => {
    if (onGetTools) {
      handleShowToolsLoad();
    }
  }, [onGetTools]);

  const handleShowToolsLoad = async () => {
    if (!onGetTools) return;
    
    try {
      const toolsList = await onGetTools();
      setTools(toolsList);
    } catch (error) {
      console.error('ツール取得エラー:', error);
    }
  };

  // 編集モードの切り替え
  const toggleEditMode = () => {
    if (editMode) {
      // 編集モードを終了して変更を保存
      // 親コンポーネントに編集後のデータを渡す
      console.log("編集後のデータ:", editedInfo);
      onUpdateExtractedInfo(editedInfo);
      onSave(editedInfo);
    }
    setEditMode(!editMode);
  };

  // エージェント実行
  const handleRunAgent = async () => {
    if (!onRunAgent) return;
    
    try {
      const suggestions = await onRunAgent();
      setAgentSuggestions(suggestions);
    } catch (error) {
      console.error('エージェント実行エラー:', error);
    }
  };

  // ツール一覧を表示
  const handleShowTools = () => {
    setShowToolsModal(true);
  };

  // 提案を採用
  const handleAcceptSuggestion = (suggestion: Suggestion) => {
    // 編集モードでない場合は確認ダイアログを表示
    if (!editMode) {
      const confirmed = window.confirm('編集画面に移動して修正を適用しますか？');
      if (!confirmed) return;
      setEditMode(true);
    }
    
    let newInfo = { ...editedInfo };
    
    // フィールドパスを解析（例: "client_info.address" -> ["client_info", "address"]）
    const fieldPath = suggestion.field.split('.');
    
    if (fieldPath.length === 1) {
      // トップレベルフィールド
      newInfo[fieldPath[0]] = suggestion.suggested_value;
    } else if (fieldPath.length === 2) {
      // ネストされたフィールド（例: client_info.address）
      const [parentField, childField] = fieldPath;
      newInfo[parentField] = {
        ...(newInfo[parentField] || {}),
        [childField]: suggestion.suggested_value
      };
    }
    
    setEditedInfo(newInfo);
    onUpdateExtractedInfo(newInfo);
    
    // 採用した提案を削除
    setAgentSuggestions(prev => 
      prev.filter(s => s.field !== suggestion.field)
    );
  };

  // 提案を却下
  const handleRejectSuggestion = (suggestion: Suggestion) => {
    setAgentSuggestions(prev => 
      prev.filter(s => s.field !== suggestion.field)
    );
  };

  // 編集をキャンセルして元に戻す
  const cancelEdit = () => {
    setEditedInfo({...originalInfo});
    setEditMode(false);
  };

  // フィールド値の更新
  const updateFieldValue = (fieldName: string, value: any) => {
    const newEditedInfo = {
      ...editedInfo,
      [fieldName]: value
    };
    setEditedInfo(newEditedInfo);
    
    // 即時に親コンポーネントにも通知
    // これにより、編集中のデータがリアルタイムで親コンポーネントに反映される
    onUpdateExtractedInfo(newEditedInfo);
  };

  // マップフィールドの値を更新
  const updateMapFieldValue = (fieldName: string, subFieldName: string, value: any) => {
    const currentMap = editedInfo[fieldName] || {};
    const updatedMap = {
      ...currentMap,
      [subFieldName]: value
    };
    
    setEditedInfo(prev => ({
      ...prev,
      [fieldName]: updatedMap
    }));
  };

  // リストフィールドのアイテムを更新
  const updateListItem = (fieldName: string, itemIndex: number, itemValue: any) => {
    const currentList = [...(editedInfo[fieldName] || [])];
    currentList[itemIndex] = itemValue;
    
    setEditedInfo(prev => ({
      ...prev,
      [fieldName]: currentList
    }));
  };

  // リストフィールドのアイテムのプロパティを更新
  const updateListItemProperty = (fieldName: string, itemIndex: number, propertyName: string, value: any) => {
    const currentList = [...(editedInfo[fieldName] || [])];
    if (!currentList[itemIndex]) {
      currentList[itemIndex] = {};
    }
    
    currentList[itemIndex] = {
      ...currentList[itemIndex],
      [propertyName]: value
    };
    
    setEditedInfo(prev => ({
      ...prev,
      [fieldName]: currentList
    }));
  };

  // フィールドの表示
  const renderField = (field: Field) => {
    if (field.type === 'string') {
      return renderStringField(field);
    }
    else if (field.type === 'map' && field.fields) {
      return renderMapField(field);
    }
    else if (field.type === 'list' && field.items) {
      return renderListField(field);
    }
    // デフォルトはシンプルなテキストフィールドとして表示
    else {
      return renderStringField(field);
    }
  };

  // フィールドに対応する修正提案を取得
  const getSuggestionForField = (fieldName: string) => {
    return agentSuggestions.find(s => s.field === fieldName);
  };

  // 修正提案の表示
  const renderSuggestion = (suggestion: Suggestion) => {
    return (
      <div className="mt-2 p-3 bg-warning-light border border-warning-border rounded">
        <div className="text-sm mb-2">
          <div className="font-semibold text-warning-text mb-1">
            {suggestion.tool_used && `${suggestion.tool_used}経由で確認済み`}
          </div>
          <div className="mb-1">
            「{suggestion.original_value}」→「{suggestion.suggested_value}」の表記ゆれを検出
          </div>
          <div className="text-neutral-700">
            提案値: {suggestion.suggested_value}
          </div>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => handleAcceptSuggestion(suggestion)}
            className="px-3 py-1 bg-primary text-on-primary text-sm rounded hover:bg-primary-hover"
          >
            採用する
          </button>
          <button
            onClick={() => handleRejectSuggestion(suggestion)}
            className="px-3 py-1 bg-neutral-300 text-neutral-700 text-sm rounded hover:bg-neutral-400"
          >
            却下
          </button>
        </div>
      </div>
    );
  };

  // 文字列フィールドの表示
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
          <div className="relative">
            <input
              type="text"
              value={value || ''}
              onChange={(e) => updateFieldValue(field.name, e.target.value)}
              className="w-full p-2 border border-neutral-300 rounded"
              onFocus={() => onHighlightField(field.name, true)}
            />
            <button
              type="button"
              onClick={() => onHighlightField(field.name, true)}
              className="absolute right-2 top-2 text-primary hover:text-primary-hover"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </button>
          </div>
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

  // マップフィールドの表示
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
                  <div className="relative">
                    <input
                      type="text"
                      value={mapValue[subField.name] || ''}
                      onChange={(e) => updateMapFieldValue(field.name, subField.name, e.target.value)}
                      className="w-full p-2 border border-neutral-300 rounded"
                      onFocus={() => onHighlightField(fieldPath, true)}
                    />
                    <button
                      type="button"
                      onClick={() => onHighlightField(fieldPath, true)}
                      className="absolute right-2 top-2 text-primary hover:text-primary-hover"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                      </svg>
                    </button>
                  </div>
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

  // リストフィールドの表示
  const renderListField = (field: Field) => {
    if (!field.items) return null;
    
    const listData = editMode ? editedInfo[field.name] || [] : extractedInfo[field.name] || [];
    
    // マップ型のリストの場合
    if (field.items.type === 'map' && field.items.fields) {
      return (
        <div key={field.name} className="mb-6">
          <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
          
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-neutral-50">
                <tr>
                  {field.items.fields.map((itemField) => (
                    <th
                      key={itemField.name}
                      scope="col"
                      className={`text-left text-xs font-medium text-neutral-500 uppercase tracking-wider ${editMode ? 'px-3 py-2' : 'px-6 py-3'}`}
                    >
                      {itemField.display_name}
                    </th>
                  ))}
                  {editMode && (
                    <th scope="col" className="px-3 py-2 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                      操作
                    </th>
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
                            className="w-full p-1 border border-neutral-300 rounded"
                            onFocus={() => onHighlightCell(field.name, itemIndex, itemField.name)}
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
                        <button
                          type="button"
                          onClick={() => {
                            const updatedList = [...listData];
                            updatedList.splice(itemIndex, 1);
                            updateFieldValue(field.name, updatedList);
                          }}
                          className="text-danger hover:text-danger-text"
                        >
                          削除
                        </button>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          {editMode && (
            <button
              type="button"
              onClick={() => {
                const newItem: Record<string, string> = {};
                field.items!.fields!.forEach(itemField => {
                  newItem[itemField.name] = '';
                });
                updateFieldValue(field.name, [...listData, newItem]);
              }}
              className="mt-2 text-info hover:text-info-text"
            >
              + 行を追加
            </button>
          )}
        </div>
      );
    }
    
    // 単純なリストの場合
    return (
      <div key={field.name} className="mb-6">
        <h3 className="text-lg font-medium mb-2">{field.display_name}</h3>
        <ul className="list-disc pl-5">
          {listData.map((item: any, itemIndex: number) => (
            <li key={itemIndex} className="mb-2">
              {editMode ? (
                <input
                  type="text"
                  value={item || ''}
                  onChange={(e) => updateListItem(field.name, itemIndex, e.target.value)}
                  className="w-full p-1 border border-neutral-300 rounded"
                />
              ) : (
                <div className="p-1">{item || ''}</div>
              )}
            </li>
          ))}
        </ul>
        
        {editMode && (
          <button
            type="button"
            onClick={() => {
              updateFieldValue(field.name, [...listData, '']);
            }}
            className="mt-2 text-info hover:text-info-text"
          >
            + 項目を追加
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="bg-bg rounded-lg border border-neutral-200 p-4">
      <div className="mb-4">
        {activeView === 'ocr' ? (
          /* OCRビュー時: 戻るボタンのみ */
          onBackToExtraction && (
            <button
              onClick={onBackToExtraction}
              className="px-4 py-2 rounded bg-neutral-500 hover:bg-neutral-600 text-on-primary"
            >
              抽出画面へ戻る
            </button>
          )
        ) : editMode ? (
          /* 編集モード時 */
          <div className="flex gap-2">
            <button
              onClick={cancelEdit}
              className="px-4 py-2 rounded bg-neutral-500 hover:bg-neutral-600 text-on-primary"
            >
              キャンセル
            </button>
            <button
              onClick={toggleEditMode}
              className="px-4 py-2 rounded bg-success hover:bg-success-hover text-on-primary"
            >
              保存
            </button>
          </div>
        ) : (
          /* 抽出ビュー時: 通常のボタン */
          <div className="flex items-center justify-between flex-wrap gap-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <button
                onClick={toggleEditMode}
                className="px-4 py-2 rounded bg-primary hover:bg-primary-hover text-on-primary"
              >
                編集
              </button>
              {isOcrEnabled && onViewOcr && (
                <button
                  onClick={onViewOcr}
                  className="px-4 py-2 rounded bg-indigo-500 hover:bg-indigo-600 text-on-primary"
                >
                  OCR結果を確認
                </button>
              )}
              
              {/* 区切り線 */}
              {onRunAgent && isAgentEnabled() && (
                <>
                  <div className="h-8 w-px bg-neutral-300"></div>
                  
                  {/* 高度な機能 */}
                  <button
                    onClick={handleShowTools}
                    className="px-3 py-2 rounded border border-neutral-300 hover:bg-neutral-50 text-neutral-700 text-sm"
                  >
                    登録ツール一覧
                  </button>
                  <button
                    onClick={handleRunAgent}
                    disabled={agentStatus === 'running'}
                    className={`px-3 py-2 rounded border text-sm ${
                      agentStatus === 'running'
                        ? 'border-neutral-300 bg-neutral-100 text-neutral-400 cursor-not-allowed'
                        : 'border-accent-border hover:bg-accent-light text-accent-text'
                    }`}
                  >
                    {agentStatus === 'running' ? '検証中...' : 'エージェントで検証'}
                  </button>
                </>
              )}
            </div>
            
            {/* 確認完了チェックボックス */}
            <div className="flex items-center gap-2 whitespace-nowrap">
              <input
                type="checkbox"
                id="verification-complete"
                checked={verificationCompleted}
                onChange={(e) => onVerificationChange?.(e.target.checked)}
                className="w-4 h-4 text-info rounded focus:ring-primary"
              />
              <label htmlFor="verification-complete" className="text-sm text-neutral-700 cursor-pointer">
                確認完了
              </label>
            </div>
          </div>
        )}
      </div>
      
      <div className="space-y-4">
        {fields.map(field => renderField(field))}
      </div>

      <Modal isOpen={showToolsModal} onClose={() => setShowToolsModal(false)} className="max-w-2xl w-full max-h-[80vh] overflow-y-auto p-6">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-bold">登録ツール一覧</h2>
              <button
                onClick={() => setShowToolsModal(false)}
                className="text-neutral-500 hover:text-neutral-700"
              >
                ✕
              </button>
            </div>
            <div className="space-y-3">
              {tools.map((tool, index) => (
                <div key={index} className="p-3 border border-neutral-200 rounded">
                  <div className="font-semibold text-neutral-800">{tool.name}</div>
                  <div className="text-sm text-neutral-600 mt-1">{tool.description}</div>
                </div>
              ))}
            </div>
      </Modal>
    </div>
  );
};

export default ExtractedInfoDisplay;

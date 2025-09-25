import React, { useState, useEffect } from 'react';
import { Field } from '../types/app-schema';

interface ExtractedInfoDisplayProps {
  extractedInfo: Record<string, any>;
  fields: Field[];
  appDisplayName: string;
  onSave: () => void;
  onHighlightField: (field: string, stayOnExtractionView?: boolean) => void;
  onHighlightCell: (fieldName: string, rowIndex: number, columnName: string) => void;
  onUpdateExtractedInfo: (info: Record<string, any>) => void;
}

const ExtractedInfoDisplay: React.FC<ExtractedInfoDisplayProps> = ({
  extractedInfo,
  fields,
  appDisplayName,
  onSave,
  onHighlightField,
  onHighlightCell,
  onUpdateExtractedInfo
}) => {
  const [editMode, setEditMode] = useState(false);
  const [editedInfo, setEditedInfo] = useState<Record<string, any>>(extractedInfo);
  const [originalInfo, setOriginalInfo] = useState<Record<string, any>>(extractedInfo);

  // 外部からのextractedInfoの変更を監視して、編集中でなければ更新
  useEffect(() => {
    if (!editMode) {
      setEditedInfo({...extractedInfo});
      setOriginalInfo({...extractedInfo});
    }
  }, [extractedInfo, editMode]);

  // 編集モードの切り替え
  const toggleEditMode = () => {
    if (editMode) {
      // 編集モードを終了して変更を保存
      // 親コンポーネントに編集後のデータを渡す
      console.log("編集後のデータ:", editedInfo);
      onUpdateExtractedInfo(editedInfo);
      
      // 少し遅延させてから保存処理を実行（状態更新が反映されるのを待つ）
      setTimeout(() => {
        onSave();
      }, 100);
    }
    setEditMode(!editMode);
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

  // 文字列フィールドの表示
  const renderStringField = (field: Field) => {
    const value = editMode ? editedInfo[field.name] : extractedInfo[field.name];
    
    return (
      <div key={field.name} className="mb-4">
        <div className="flex justify-between items-center mb-1">
          <label className="block text-sm font-medium text-gray-700">
            {field.display_name}
          </label>
        </div>
        
        {editMode ? (
          <div className="relative">
            <input
              type="text"
              value={value || ''}
              onChange={(e) => updateFieldValue(field.name, e.target.value)}
              className="w-full p-2 border border-gray-300 rounded"
              onFocus={() => onHighlightField(field.name, true)}
            />
            <button
              type="button"
              onClick={() => onHighlightField(field.name, true)}
              className="absolute right-2 top-2 text-blue-500 hover:text-blue-700"
            >
              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </button>
          </div>
        ) : (
          <div 
            className="p-2 bg-gray-50 border border-gray-200 rounded cursor-pointer hover:bg-gray-100"
            onClick={() => onHighlightField(field.name, true)}
          >
            {value || '(抽出されませんでした)'}
          </div>
        )}
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
        <div className="pl-4 border-l-2 border-gray-200 space-y-3">
          {field.fields.map(subField => (
            <div key={subField.name} className="mb-3">
              <div className="flex justify-between items-center mb-1">
                <label className="block text-sm font-medium text-gray-700">
                  {subField.display_name}
                </label>
              </div>
              
              {editMode ? (
                <div className="relative">
                  <input
                    type="text"
                    value={mapValue[subField.name] || ''}
                    onChange={(e) => updateMapFieldValue(field.name, subField.name, e.target.value)}
                    className="w-full p-2 border border-gray-300 rounded"
                    onFocus={() => onHighlightField(`${field.name}.${subField.name}`, true)}
                  />
                  <button
                    type="button"
                    onClick={() => onHighlightField(`${field.name}.${subField.name}`, true)}
                    className="absolute right-2 top-2 text-blue-500 hover:text-blue-700"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </button>
                </div>
              ) : (
                <div 
                  className="p-2 bg-gray-50 border border-gray-200 rounded cursor-pointer hover:bg-gray-100"
                  onClick={() => onHighlightField(`${field.name}.${subField.name}`, true)}
                >
                  {mapValue[subField.name] || '(抽出されませんでした)'}
                </div>
              )}
            </div>
          ))}
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
              <thead className="bg-gray-50">
                <tr>
                  {field.items.fields.map((itemField) => (
                    <th 
                      key={itemField.name}
                      scope="col" 
                      className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider"
                    >
                      {itemField.display_name}
                    </th>
                  ))}
                  {editMode && (
                    <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      操作
                    </th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {listData.map((item: any, itemIndex: number) => (
                  <tr key={itemIndex}>
                    {field.items!.fields!.map(itemField => (
                      <td key={itemField.name} className="px-6 py-4 whitespace-nowrap">
                        {editMode ? (
                          <input
                            type="text"
                            value={item[itemField.name] || ''}
                            onChange={(e) => updateListItemProperty(field.name, itemIndex, itemField.name, e.target.value)}
                            className="w-full p-1 border border-gray-300 rounded"
                            onFocus={() => onHighlightCell(field.name, itemIndex, itemField.name)}
                          />
                        ) : (
                          <div 
                            className="text-sm text-gray-900 cursor-pointer hover:bg-blue-50 p-1 rounded"
                            onClick={() => onHighlightCell(field.name, itemIndex, itemField.name)}
                          >
                            {item[itemField.name] || ''}
                          </div>
                        )}
                      </td>
                    ))}
                    {editMode && (
                      <td className="px-6 py-4 whitespace-nowrap">
                        <button
                          type="button"
                          onClick={() => {
                            const updatedList = [...listData];
                            updatedList.splice(itemIndex, 1);
                            updateFieldValue(field.name, updatedList);
                          }}
                          className="text-red-600 hover:text-red-900"
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
              className="mt-2 text-blue-600 hover:text-blue-800"
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
                  className="w-full p-1 border border-gray-300 rounded"
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
            className="mt-2 text-blue-600 hover:text-blue-800"
          >
            + 項目を追加
          </button>
        )}
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">{appDisplayName}の抽出結果</h2>
        <div className="flex space-x-2">
          {editMode ? (
            <>
              <button
                onClick={cancelEdit}
                className="px-4 py-2 rounded bg-gray-500 hover:bg-gray-600 text-white"
              >
                キャンセル
              </button>
              <button
                onClick={toggleEditMode}
                className="px-4 py-2 rounded bg-green-500 hover:bg-green-600 text-white"
              >
                保存
              </button>
            </>
          ) : (
            <button
              onClick={toggleEditMode}
              className="px-4 py-2 rounded bg-blue-500 hover:bg-blue-600 text-white"
            >
              編集
            </button>
          )}
        </div>
      </div>
      
      <div className="space-y-4">
        {fields.map(field => renderField(field))}
      </div>
    </div>
  );
};

export default ExtractedInfoDisplay;

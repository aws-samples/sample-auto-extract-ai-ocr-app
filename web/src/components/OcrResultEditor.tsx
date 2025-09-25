import React, { useState, useEffect, useRef } from 'react';
import { OcrResultData } from '../types/ocr';

interface OcrResultEditorProps {
  ocrResults: OcrResultData[];
  selectedIndex: number | null;
  onUpdateOcrResults: (results: OcrResultData[]) => void;
  onStartExtraction: () => void;
  onSelectIndex?: (index: number) => void;
}

const OcrResultEditor: React.FC<OcrResultEditorProps> = ({
  ocrResults,
  selectedIndex,
  onUpdateOcrResults,
  onStartExtraction,
  onSelectIndex
}) => {
  const [editMode, setEditMode] = useState(false);
  const [editedResults, setEditedResults] = useState<OcrResultData[]>(ocrResults);
  const [originalResults, setOriginalResults] = useState<OcrResultData[]>(ocrResults);
  const selectedRowRef = useRef<HTMLTableRowElement>(null);

  // 選択されたインデックスが変更されたらその行までスクロール（初期表示時を除く）
  const [initialRender, setInitialRender] = useState(true);
  
  useEffect(() => {
    if (selectedIndex !== null && selectedRowRef.current && !initialRender) {
      selectedRowRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'center'
      });
    }
    
    // 初期レンダリングフラグをリセット
    if (initialRender) {
      setInitialRender(false);
    }
  }, [selectedIndex, initialRender]);

  // 編集モードに入る時に元のデータを保存
  useEffect(() => {
    if (!editMode) {
      setOriginalResults([...ocrResults]);
    }
    setEditedResults([...ocrResults]);
  }, [editMode, ocrResults]);

  // テキストクリック時のハンドラ
  const handleTextClick = (index: number) => {
    if (onSelectIndex) {
      onSelectIndex(index);
    }
  };

  // 編集モードの切り替え
  const toggleEditMode = () => {
    if (editMode) {
      // 編集モードを終了して変更を保存
      onUpdateOcrResults(editedResults);
    }
    setEditMode(!editMode);
  };

  // 編集をキャンセル
  const cancelEdit = () => {
    setEditedResults([...originalResults]);
    setEditMode(false);
  };

  // テキスト内容の更新
  const updateTextContent = (index: number, content: string) => {
    const updatedResults = [...editedResults];
    updatedResults[index] = {
      ...updatedResults[index],
      content: content
    };
    setEditedResults(updatedResults);
  };

  // テキストエリアの自動リサイズ
  const handleTextareaInput = (e: React.FormEvent<HTMLTextAreaElement>) => {
    const target = e.target as HTMLTextAreaElement;
    target.style.height = 'auto';
    target.style.height = `${Math.min(target.scrollHeight, 100)}px`;
  };

  return (
    <div className="ocr-result-editor overflow-y-auto" style={{ height: "100%" }}>
      <div className="flex justify-end items-center mb-4 p-2">
        <div className="flex space-x-2">
          {editMode ? (
            <>
              <button
                onClick={cancelEdit}
                className="px-4 py-2 rounded bg-gray-500 hover:bg-gray-600 text-white text-sm"
              >
                キャンセル
              </button>
              <button
                onClick={toggleEditMode}
                className="px-4 py-2 rounded bg-green-500 hover:bg-green-600 text-white text-sm"
              >
                保存
              </button>
            </>
          ) : (
            <>
              <button
                onClick={onStartExtraction}
                className="px-4 py-2 rounded bg-yellow-500 hover:bg-yellow-600 text-white text-sm"
              >
                OCR結果に基づいて再度情報抽出
              </button>
              <button
                onClick={toggleEditMode}
                className="px-4 py-2 rounded bg-blue-500 hover:bg-blue-600 text-white text-sm ml-2"
              >
                編集
              </button>
            </>
          )}
        </div>
      </div>

      <div className="border rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50 sticky top-0 z-10">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider w-16">
                #
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                テキスト
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {editedResults.map((result, index) => (
              <tr 
                key={index}
                ref={selectedIndex === index ? selectedRowRef : null}
                className={`${selectedIndex === index ? 'bg-blue-100' : index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}`}
              >
                <td className="px-6 py-2 text-sm text-gray-500 w-16">
                  {index}
                </td>
                <td className="px-6 py-2 text-sm text-gray-900">
                  {editMode ? (
                    <textarea
                      value={result.content}
                      onChange={(e) => updateTextContent(index, e.target.value)}
                      onInput={handleTextareaInput}
                      className="w-full p-1 border border-gray-300 rounded"
                      rows={1}
                      style={{ resize: "none", minHeight: "22px", maxHeight: "100px", overflow: "hidden" }}
                    />
                  ) : (
                    <span 
                      className="cursor-pointer" 
                      onClick={() => {
                        if (selectedIndex !== index) {
                          handleTextClick(index);
                        }
                      }}
                    >
                      {result.content}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default OcrResultEditor;

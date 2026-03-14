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
  onSelectIndex
}) => {
  const [editedResults] = useState<OcrResultData[]>(ocrResults);
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

  // テキストクリック時のハンドラ
  const handleTextClick = (index: number) => {
    if (onSelectIndex) {
      onSelectIndex(index);
    }
  };

  return (
    <div className="ocr-result-editor overflow-y-auto" style={{ height: "100%" }}>
      <div className="border rounded-lg overflow-hidden">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-neutral-50 sticky top-0 z-10">
            <tr>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider w-16">
                #
              </th>
              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-neutral-500 uppercase tracking-wider">
                テキスト
              </th>
            </tr>
          </thead>
          <tbody className="bg-bg divide-y divide-gray-200">
            {editedResults.map((result, index) => (
              <tr 
                key={index}
                ref={selectedIndex === index ? selectedRowRef : null}
                className={`${selectedIndex === index ? 'bg-info-light' : index % 2 === 0 ? 'bg-neutral-50' : 'bg-bg'}`}
              >
                <td className="px-6 py-2 text-sm text-neutral-500 w-16">
                  {index}
                </td>
                <td className="px-6 py-2 text-sm text-neutral-900">
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

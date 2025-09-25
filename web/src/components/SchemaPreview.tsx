import React from 'react';
import { Field } from '../types/app-schema';

interface SchemaPreviewProps {
  schema: {
    fields: Field[];
  } | null;
}

const SchemaPreview: React.FC<SchemaPreviewProps> = ({ schema }) => {
  if (!schema || !schema.fields) {
    return (
      <div className="text-gray-500 text-center py-4">
        有効なスキーマがありません
      </div>
    );
  }

  const getTypeClass = (type: string) => {
    switch (type) {
      case 'string':
        return 'bg-blue-100 text-blue-800';
      case 'number':
        return 'bg-green-100 text-green-800';
      case 'map':
        return 'bg-purple-100 text-purple-800';
      case 'list':
        return 'bg-orange-100 text-orange-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const renderFields = (fields: Field[], level = 0) => {
    return fields.map((field, index) => (
      <div 
        key={`${field.name}-${index}`} 
        className={`border rounded-md p-3 mb-3 ${level > 0 ? 'ml-4 border-l-2 border-gray-200' : ''}`}
      >
        <div className="flex justify-between items-center">
          <div>
            <span className="font-medium">{field.display_name}</span>
            <span className="ml-2 text-sm text-gray-500">({field.name})</span>
          </div>
          <span className={`px-2 py-1 text-xs rounded-full ${getTypeClass(field.type)}`}>
            {field.type}
          </span>
        </div>
        
        {/* ネストされたフィールド（map型の場合） */}
        {field.type === 'map' && field.fields && field.fields.length > 0 && (
          <div className="mt-2 pl-4 border-l-2 border-gray-200">
            {renderFields(field.fields, level + 1)}
          </div>
        )}
        
        {/* リスト型の場合 */}
        {field.type === 'list' && field.items && (
          <div className="mt-2 pl-4 border-l-2 border-gray-200">
            <div className="flex justify-between items-center">
              <div>
                <span className="font-medium">リスト項目</span>
              </div>
              <span className={`px-2 py-1 text-xs rounded-full ${getTypeClass(field.items.type)}`}>
                {field.items.type}
              </span>
            </div>
            
            {/* リスト内のマップ型 */}
            {field.items.type === 'map' && field.items.fields && field.items.fields.length > 0 && (
              <div className="mt-2 pl-4 border-l-2 border-gray-200">
                {renderFields(field.items.fields, level + 2)}
              </div>
            )}
          </div>
        )}
      </div>
    ));
  };

  return (
    <div>
      <div className="space-y-2">
        {renderFields(schema.fields)}
      </div>
    </div>
  );
};

export default SchemaPreview;

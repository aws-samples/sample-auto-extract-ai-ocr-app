import { Field } from './app-schema';

export interface ExtractionResponse {
  extracted_info: Record<string, any>;
  mapping: ExtractionMapping;
  status: string;
  app_name: string;
  app_display_name: string;
  fields: Field[];
}

// 階層構造に対応したマッピング型
export type ExtractionMapping = {
  [key: string]: any;
};

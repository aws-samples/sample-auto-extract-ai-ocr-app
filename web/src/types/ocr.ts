export interface ImageFile {
  id: string;
  name: string;
  s3_key: string;
  uploadTime: string;
  status: 'uploading' | 'converting' | 'pending' | 'ocr' | 'extracting' | 'processing' | 'completed' | 'failed';
  jobId?: string;
  appName?: string;
  pageProcessingMode?: 'combined' | 'individual';
  pageNumber?: number;
  totalPages?: number;
  parentDocumentId?: string;
  verificationCompleted?: boolean;
  uploaded_by?: string;
  uploaded_by_email?: string;
  verified_by?: string;
  verified_by_email?: string;
  agentStatus?: 'processing' | 'completed' | 'failed' | 'skipped';
  agentSuggestionsCount?: number;
}

export interface OcrWord {
  content: string;
  rec_score?: number;
  points?: number[][];
  id?: number;
  page?: number;  // ページ番号を追加（マルチページ対応）
}

export interface OcrResultData extends OcrWord {
  // 拡張プロパティがあれば追加
}

export interface OcrBoundingBox {
  id: number;
  top: number;
  left: number;
  width: number;
  height: number;
  text: string;
}

export interface OcrResponse {
  filename: string;
  s3_key: string;
  uploadTime: string;
  status: string;
  ocrResult: {
    words: OcrWord[];
  };
  imageUrl: string;
  app_name?: string;
}

export interface PresignedDownloadUrlResponse {
  presigned_url: string;
  presigned_urls: Array<{
    page: number;
    presigned_url: string;
    s3_key: string;
  }>;
  total_pages: number;
  is_multipage: boolean;
  content_type: string;
  filename: string;
  is_converted: boolean;
}

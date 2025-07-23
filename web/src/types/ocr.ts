export interface ImageFile {
  id: string;
  name: string;
  s3_key: string;
  uploadTime: string;
  status: 'uploading' | 'pending' | 'processing' | 'completed' | 'failed';
  jobId?: string;
  appName?: string;
}

export interface OcrWord {
  content: string;
  rec_score?: number;
  det_score?: number;
  points?: number[][];
  id?: number;
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
}

export interface OcrStartResponse {
  jobId: string;
}

export interface PresignedUrlResponse {
  presigned_url: string;
  s3_key: string;
  image_id: string;
}

export interface UploadCompleteResponse {
  status: string;
  message: string;
  image_id: string;
}
export interface OcrStatusResponse {
  status: string;
  images: {
    id: string;
    filename: string;
    status: string;
  }[];
}

export interface PresignedDownloadUrlResponse {
  presigned_url: string;
  content_type: string;
  filename: string;
  is_converted: boolean;
}

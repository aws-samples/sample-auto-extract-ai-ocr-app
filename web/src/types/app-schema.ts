export interface Field {
  name: string;
  display_name: string;
  type: 'string' | 'number' | 'map' | 'list';
  fields?: Field[];    // map型のフィールド用
  items?: {           // list型のフィールド用
    type: 'string' | 'number' | 'map' | 'list';
    fields?: Field[];
  };
}

export interface InputMethods {
  file_upload: boolean;
  s3_sync: boolean;
  s3_uri?: string;
}

export interface AppSchema {
  name: string;
  display_name: string;
  description?: string;
  fields: Field[];
  input_methods?: InputMethods;
  permission?: string;
  sample_image_s3_key?: string;
  sample_image_filename?: string;
  agent_enabled?: boolean;
  agent_auto_run?: boolean;
}

export interface S3SyncFile {
  key: string;
  size: number;
  last_modified: string;
  filename: string;
  bucket?: string;
  is_existing?: boolean;
}

export interface S3ImportResponse {
  status: string;
  message: string;
  image_id: string;
}

export interface Column {
  name: string;
  display_name: string;
  type: string;
}

export interface Field {
  name: string;
  display_name: string;
  type: string;
  fields?: Field[];    // map型のフィールド用
  items?: {           // list型のフィールド用
    type: string;
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
}

export interface AppSchemaResponse {
  apps: AppSchema[];
}

export interface S3SyncFile {
  key: string;
  size: number;
  last_modified: string;
  filename: string;
  bucket?: string;
  is_existing?: boolean;
}

export interface S3SyncResponse {
  app_name: string;
  bucket: string;
  prefix: string;
  structure: FolderTree;
  files: S3SyncFile[];
}

export interface FolderTree {
  [key: string]: {
    type: "folder" | "file";
    children?: FolderTree;
    data?: S3SyncFile & { relative_path: string };
  };
}

export interface S3ImportResponse {
  status: string;
  message: string;
  image_id: string;
  is_converting: boolean;
}

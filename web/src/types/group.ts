import type { SearchUser } from './user';

export interface Group {
  id: string;
  name: string;
  description?: string;
  source: string;
  created_at: string;
  member_count: number;
}

// GET /user/search の検索結果に含まれるグループ要素
export interface SearchGroup {
  id: string;
  name: string;
  description?: string;
}

// GET /user/search のレスポンス
export interface SearchResult {
  users: SearchUser[];
  groups: SearchGroup[];
}

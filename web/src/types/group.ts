export interface Group {
  id: string;
  name: string;
  description?: string;
  source: string;
  created_at: string;
  member_count: number;
}

export interface SearchGroup {
  id: string;
  name: string;
  description?: string;
}

export interface SearchResult {
  users: { id: string; email: string; display_name?: string }[];
  groups: { id: string; name: string }[];
}

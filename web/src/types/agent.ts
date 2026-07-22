export interface Tool {
  id: string;
  name: string;
  description: string;
  is_active: boolean;
}

export interface Suggestion {
  field: string;
  original_value: string;
  suggested_value: string;
  reason: string;
  confidence: string;
  tool_used?: string;
  index: number;  // original index in backend suggestions array
}

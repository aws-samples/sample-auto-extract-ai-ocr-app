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

// GET /images/{id}/agent/tools のレスポンス
export interface ToolsResponse {
  status: string;
  tools: Tool[];
}

// GET /images/{id}/agent のレスポンス（画像の最新エージェントジョブ）
export interface AgentJobResponse {
  status: string;
  suggestions: Suggestion[];
  job_id?: string;
  total_suggestions_count?: number;
  error?: string;
}

// runAgent / pollAgentJobStatus が整形して返す結果
export interface AgentRunResult {
  status: string;
  suggestions: Suggestion[];
}

/** RBAC 管理対象のツール（agent.ts の Tool とは別管理） */
export interface ManagedTool {
  id: string;
  name: string;
  description?: string;
  is_active: boolean;
}

export interface ToolPermissions {
  users: { id: string; email: string; display_name?: string }[];
  groups: { id: string; name: string }[];
  usecases: { id: string; app_name: string }[];
}

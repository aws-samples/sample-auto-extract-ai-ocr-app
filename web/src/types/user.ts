export interface User {
  id: string;
  cognito_sub: string;
  email: string;
  display_name?: string;
  department?: string;
  role: string;
  is_active: boolean;
  created_at: string;
  groups?: Array<{ name: string; source: string }>;
}

export interface SearchUser {
  id: string;
  email: string;
  display_name?: string;
}

export interface GroupMember {
  id: string;
  email: string;
  display_name?: string;
  role: string;
  source: string;
}

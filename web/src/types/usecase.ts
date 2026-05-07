export interface Usecase {
  id: string;
  app_name: string;
  created_at: string;
  created_by_email?: string;
  owner_emails?: string[];
}

export interface UsecaseUserPermission {
  id: string;
  email: string;
  display_name?: string;
  permission: string;
}

export interface UsecaseGroupPermission {
  id: string;
  name: string;
  permission: string;
}

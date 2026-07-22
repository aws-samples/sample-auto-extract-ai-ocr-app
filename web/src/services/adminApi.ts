import api from './api';

// Users
export const getUsers = () => api.get('/admin/users').then(r => r.data);
export const updateUserRole = (userId: string, role: string) =>
  api.patch(`/admin/users/${userId}/role`, { role });

// Groups
export const getGroups = () => api.get('/admin/groups').then(r => r.data);
export const createGroup = (name: string, description?: string) =>
  api.post('/admin/groups', { name, description });
export const deleteGroup = (groupId: string) =>
  api.delete(`/admin/groups/${groupId}`);
export const updateGroup = (groupId: string, data: { name?: string; description?: string }) =>
  api.patch(`/admin/groups/${groupId}`, data);
export const getGroupMembers = (groupId: string) =>
  api.get(`/admin/groups/${groupId}/members`).then(r => r.data);
export const updateGroupMembers = (groupId: string, userIds: string[]) =>
  api.put(`/admin/groups/${groupId}/members`, { user_ids: userIds });

// Usecases
export const getUsecases = () => api.get('/admin/usecases').then(r => r.data);
export const getUsecasePermissions = (appName: string) =>
  api.get(`/admin/usecases/${appName}/permissions`).then(r => r.data);

// Tools
export const getTools = () => api.get('/admin/tools').then(r => r.data);
export const getToolPermissions = (toolId: string) =>
  api.get(`/admin/tools/${toolId}/permissions`).then(r => r.data);
export const updateTool = (toolId: string, data: { name?: string; description?: string; is_active?: boolean }) =>
  api.patch(`/admin/tools/${toolId}`, data);

// Images (admin)
export const getAllImages = (appName?: string) =>
  api.get('/admin/images', { params: appName ? { app_name: appName } : {} }).then(r => r.data);

// Tool permissions
export const addToolUser = (toolId: string, userId: string) =>
  api.post(`/admin/tools/${toolId}/users`, { user_id: userId });
export const removeToolUser = (toolId: string, userId: string) =>
  api.delete(`/admin/tools/${toolId}/users/${userId}`);
export const addToolGroup = (toolId: string, groupId: string) =>
  api.post(`/admin/tools/${toolId}/groups`, { group_id: groupId });
export const removeToolGroup = (toolId: string, groupId: string) =>
  api.delete(`/admin/tools/${toolId}/groups/${groupId}`);

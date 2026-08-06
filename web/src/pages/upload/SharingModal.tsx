import { useState, useEffect } from 'react';
import api from '../../services/api';
import { PermissionModal } from '../../components/shared/PermissionModal';
import type { Owner, PermissionUser, PermissionGroup } from '../../components/shared/PermissionModal';

interface SharingModalProps {
  isOpen: boolean;
  onClose: () => void;
  appName: string;
  appDisplayName: string;
  currentUserId?: string;
  onPermissionLost?: () => void;
}

const USER_PERM_LEVELS = [
  { value: 'viewer', label: '閲覧者' },
  { value: 'editor', label: '編集者' },
  { value: 'owner', label: 'オーナー' },
];

const GROUP_PERM_LEVELS = [
  { value: 'viewer', label: '閲覧者' },
  { value: 'editor', label: '編集者' },
];

export default function SharingModal({ isOpen, onClose, appName, appDisplayName, currentUserId, onPermissionLost }: SharingModalProps) {
  const [owners, setOwners] = useState<Owner[]>([]);
  const [users, setUsers] = useState<PermissionUser[]>([]);
  const [groups, setGroups] = useState<PermissionGroup[]>([]);
  const [isPublic, setIsPublic] = useState(false);

  const loadSharing = async () => {
    try {
      const r = await api.get(`/apps/${appName}/sharing`);
      setOwners(r.data.owners || []);
      setUsers(r.data.users || []);
      const grps = r.data.groups || [];
      setGroups(grps);
      setIsPublic(grps.some((g: PermissionGroup) => g.name === 'all'));
    } catch {}
  };

  useEffect(() => {
    if (isOpen) loadSharing();
  }, [isOpen, appName]);

  const togglePublic = async (checked: boolean) => {
    if (checked) {
      await api.post(`/apps/${appName}/sharing/all`);
    } else {
      const allGroup = groups.find((g) => g.name === 'all');
      if (allGroup) await api.delete(`/apps/${appName}/sharing/groups/${allGroup.id}`);
    }
    setIsPublic(checked);
    await loadSharing();
  };

  return (
    <PermissionModal
      isOpen={isOpen}
      onClose={onClose}
      title={`「${appDisplayName}」を共有`}
      subtitle="ユーザーやグループを追加してアクセス権を管理します"
      owners={owners}
      users={users}
      groups={groups}
      isPublic={isPublic}
      userPermissionLevels={USER_PERM_LEVELS}
      groupPermissionLevels={GROUP_PERM_LEVELS}
      onAddUser={async (userId) => {
        await api.post(`/apps/${appName}/sharing/users`, { user_id: userId, permission: 'viewer' });
        await loadSharing();
      }}
      onRemoveUser={async (userId) => {
        if (userId === currentUserId && !window.confirm('自分のアクセス権を削除すると、このユースケースにアクセスできなくなる可能性があります。続行しますか？')) return;
        await api.delete(`/apps/${appName}/sharing/users/${userId}`);
        if (userId === currentUserId) { onPermissionLost?.(); return; }
        await loadSharing();
      }}
      onUpdateUserPermission={async (userId, perm) => {
        const currentPerm = users.find((u) => u.id === userId)?.permission;
        if (userId === currentUserId && currentPerm === 'owner' && perm !== 'owner' && !window.confirm('自分のオーナー権限を変更すると、共有設定を編集できなくなる可能性があります。続行しますか？')) return;
        await api.post(`/apps/${appName}/sharing/users`, { user_id: userId, permission: perm });
        if (userId === currentUserId && currentPerm === 'owner' && perm !== 'owner') { onPermissionLost?.(); return; }
        await loadSharing();
      }}
      onAddGroup={async (groupId) => {
        await api.post(`/apps/${appName}/sharing/groups`, { group_id: groupId, permission: 'viewer' });
        await loadSharing();
      }}
      onRemoveGroup={async (groupId) => {
        await api.delete(`/apps/${appName}/sharing/groups/${groupId}`);
        await loadSharing();
      }}
      onUpdateGroupPermission={async (groupId, perm) => {
        await api.post(`/apps/${appName}/sharing/groups`, { group_id: groupId, permission: perm });
        await loadSharing();
      }}
      onTogglePublic={togglePublic}
    />
  );
}

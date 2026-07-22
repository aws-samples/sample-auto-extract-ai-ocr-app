import { useState, useCallback, useMemo } from 'react';
import { Layers } from 'lucide-react';
import { Table, Thead, Tbody, Button, usePagination, Pagination, SearchBox, CardTable, EmptyState, TableSkeleton } from '../../components/ui';
import api from '../../services/api';
import { useFetch } from '../../hooks/useFetch';
import * as adminApi from '../../services/adminApi';
import { PermissionModal } from '../../components/shared/PermissionModal';
import { AdminToolbar } from './AdminToolbar';
import type { Usecase, UsecaseUserPermission, UsecaseGroupPermission } from '../../types/usecase';

const USER_PERM_LEVELS = [
  { value: 'viewer', label: '閲覧者' },
  { value: 'editor', label: '編集者' },
  { value: 'owner', label: 'オーナー' },
];

const GROUP_PERM_LEVELS = [
  { value: 'viewer', label: '閲覧者' },
  { value: 'editor', label: '編集者' },
];

export default function UsecasesTab() {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<Usecase | null>(null);
  const [perms, setPerms] = useState<{ users: UsecaseUserPermission[]; groups: UsecaseGroupPermission[] }>({ users: [], groups: [] });
  const [isPublic, setIsPublic] = useState(false);

  const fetchUsecases = useCallback(async () => (await adminApi.getUsecases()).usecases || [], []);
  const { data: usecases, loading } = useFetch<Usecase[]>(fetchUsecases, []);

  const filtered = useMemo(() => {
    if (!search) return usecases;
    const q = search.toLowerCase();
    return usecases.filter((uc) =>
      uc.app_name?.toLowerCase().includes(q) ||
      uc.created_by_email?.toLowerCase().includes(q) ||
      uc.owner_emails?.some((e: string) => e.toLowerCase().includes(q))
    );
  }, [usecases, search]);

  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } = usePagination(filtered);

  const loadPerms = async (uc: Usecase) => {
    const data = await adminApi.getUsecasePermissions(uc.app_name);
    setPerms(data);
    setIsPublic(data.groups.some((g: { name?: string }) => g.name === 'all'));
  };

  const openPerms = async (uc: Usecase) => {
    setSelected(uc);
    await loadPerms(uc);
  };

  const togglePublic = async (checked: boolean) => {
    if (!selected) return;
    if (checked) {
      await api.post(`/apps/${selected.app_name}/sharing/all`);
    } else {
      const allGroup = perms.groups.find((g) => g.name === 'all');
      if (allGroup) await api.delete(`/apps/${selected.app_name}/sharing/groups/${allGroup.id}`);
    }
    setIsPublic(checked);
    await loadPerms(selected);
  };

  if (loading) return <TableSkeleton rows={5} cols={5} />;

  return (
    <>
      <AdminToolbar
        left={<SearchBox value={search} onChange={(v: string) => { setSearch(v); setPage(0); }} placeholder="アプリ名、作成者で検索..." />}
      />
      {filtered.length === 0 ? (
        <EmptyState icon={Layers} message="ユースケースが見つかりません" />
      ) : (
        <CardTable>
          <Table>
            <Thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">アプリ名</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">作成者</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">オーナー</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">作成日</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">操作</th>
              </tr>
            </Thead>
            <Tbody>
              {paged.map((uc) => (
                <tr key={uc.id}>
                  <td className="px-4 py-3 text-sm">{uc.app_name}</td>
                  <td className="px-4 py-3 text-sm">{uc.created_by_email || '-'}</td>
                  <td className="px-4 py-3 text-sm">{uc.owner_emails?.join(', ') || '-'}</td>
                  <td className="px-4 py-3 text-sm">{new Date(uc.created_at).toLocaleDateString('ja-JP')}</td>
                  <td className="px-4 py-3">
                    <Button variant="ghost" size="sm" onClick={() => openPerms(uc)}>権限</Button>
                  </td>
                </tr>
              ))}
            </Tbody>
          </Table>
        </CardTable>
      )}
      <Pagination page={page} total={total} setPage={setPage} pageSize={pageSize} totalItems={totalItems} onPageSizeChange={changePageSize} />

      <PermissionModal
        isOpen={!!selected}
        onClose={() => setSelected(null)}
        title={`${selected?.app_name || ''} - 権限管理`}
        subtitle="ユーザーやグループを追加してアクセス権を管理します"
        users={perms.users}
        groups={perms.groups}
        isPublic={isPublic}
        userPermissionLevels={USER_PERM_LEVELS}
        groupPermissionLevels={GROUP_PERM_LEVELS}
        onAddUser={async (userId) => { if (selected) { await api.post(`/apps/${selected.app_name}/sharing/users`, { user_id: userId, permission: 'viewer' }); await loadPerms(selected); } }}
        onRemoveUser={async (userId) => { if (selected) { await api.delete(`/apps/${selected.app_name}/sharing/users/${userId}`); await loadPerms(selected); } }}
        onUpdateUserPermission={async (userId, perm) => { if (selected) { await api.post(`/apps/${selected.app_name}/sharing/users`, { user_id: userId, permission: perm }); await loadPerms(selected); } }}
        onAddGroup={async (groupId) => { if (selected) { await api.post(`/apps/${selected.app_name}/sharing/groups`, { group_id: groupId, permission: 'viewer' }); await loadPerms(selected); } }}
        onRemoveGroup={async (groupId) => { if (selected) { await api.delete(`/apps/${selected.app_name}/sharing/groups/${groupId}`); await loadPerms(selected); } }}
        onUpdateGroupPermission={async (groupId, perm) => { if (selected) { await api.post(`/apps/${selected.app_name}/sharing/groups`, { group_id: groupId, permission: perm }); await loadPerms(selected); } }}
        onTogglePublic={togglePublic}
      />
    </>
  );
}

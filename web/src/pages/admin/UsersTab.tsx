import { useState, useEffect, useCallback, useMemo } from 'react';
import { Users } from 'lucide-react';
import { Table, Thead, Tbody, Badge, usePagination, Pagination, SearchBox, CardTable, EmptyState, TableSkeleton } from '../../components/ui';
import * as adminApi from '../../services/adminApi';
import { AdminToolbar } from './AdminToolbar';
import type { User } from '../../types/user';
import UserDetailModal from './UserDetailModal';

const ROLE_OPTIONS = [
  { value: '', label: 'すべてのロール' },
  { value: 'admin', label: 'Admin' },
  { value: 'author', label: 'Author' },
  { value: 'reader', label: 'Reader' },
];

const ROLE_BADGE_COLOR = {
  admin: 'red',
  author: 'blue',
  reader: 'gray',
} as const;

export default function UsersTab() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [roleFilter, setRoleFilter] = useState('');
  const [selectedUser, setSelectedUser] = useState<User | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await adminApi.getUsers();
      setUsers(data.users || []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const filtered = useMemo(() => {
    return users.filter((u) => {
      if (roleFilter && u.role !== roleFilter) return false;
      if (!search) return true;
      const q = search.toLowerCase();
      return u.email?.toLowerCase().includes(q) ||
        u.display_name?.toLowerCase().includes(q) ||
        u.groups?.some((g) => g.name.toLowerCase().includes(q));
    });
  }, [users, search, roleFilter]);

  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } = usePagination(filtered);

  const handleUpdated = (userId: string, newRole: string) => {
    setUsers(prev => prev.map(u => u.id === userId ? { ...u, role: newRole } : u));
  };

  if (loading) return <TableSkeleton rows={5} cols={5} />;

  return (
    <>
      <AdminToolbar
        left={
          <>
            <SearchBox value={search} onChange={(v: string) => { setSearch(v); setPage(0); }} placeholder="メール、名前、グループで検索..." />
            <select
              value={roleFilter}
              onChange={(e) => { setRoleFilter(e.target.value); setPage(0); }}
              className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg"
            >
              {ROLE_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </>
        }
      />
      {filtered.length === 0 ? (
        <EmptyState icon={Users} message="ユーザーが見つかりません" />
      ) : (
        <CardTable>
          <Table>
            <Thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">メール</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">表示名</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">グループ</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ロール</th>
              </tr>
            </Thead>
            <Tbody>
              {paged.map((u) => (
                <tr key={u.id} className="cursor-pointer" onClick={() => setSelectedUser(u)}>
                  <td className="px-4 py-3 text-sm">{u.email}</td>
                  <td className="px-4 py-3 text-sm">{u.display_name || '-'}</td>
                  <td className="px-4 py-3 text-sm">
                    {u.groups && u.groups.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {u.groups.slice(0, 2).map((g) => (
                          <Badge key={g.name} color={g.source === 'idp' ? 'blue' : 'gray'} className="text-xs">{g.name}</Badge>
                        ))}
                        {u.groups.length > 2 && (
                          <span className="text-xs text-muted" title={u.groups.slice(2).map((g) => g.name).join(', ')}>+{u.groups.length - 2}</span>
                        )}
                      </div>
                    ) : '-'}
                  </td>
                  <td className="px-4 py-3">
                    <Badge color={ROLE_BADGE_COLOR[u.role as keyof typeof ROLE_BADGE_COLOR] || 'gray'}>
                      {u.role}
                    </Badge>
                  </td>
                </tr>
              ))}
            </Tbody>
          </Table>
        </CardTable>
      )}
      <Pagination
        page={page} total={total} setPage={setPage}
        pageSize={pageSize} totalItems={totalItems} onPageSizeChange={changePageSize}
      />

      <UserDetailModal
        user={selectedUser}
        onClose={() => setSelectedUser(null)}
        onUpdated={handleUpdated}
      />
    </>
  );
}

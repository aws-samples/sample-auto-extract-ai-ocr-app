import { useState, useEffect, useCallback, useMemo } from 'react';
import { Wrench } from 'lucide-react';
import { Button, Table, Thead, Tbody, Toggle, usePagination, Pagination, SearchBox, CardTable, EmptyState, TableSkeleton } from '../../components/ui';
import * as adminApi from '../../services/adminApi';
import { PermissionModal } from '../../components/shared/PermissionModal';
import { AdminToolbar } from './AdminToolbar';
import type { ManagedTool, ToolPermissions } from '../../types/tool';

export default function ToolsTab() {
  const [tools, setTools] = useState<ManagedTool[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [selectedTool, setSelectedTool] = useState<ManagedTool | null>(null);
  const [perms, setPerms] = useState<ToolPermissions>({ users: [], groups: [], usecases: [] });
  const [isPublic, setIsPublic] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await adminApi.getTools();
      setTools(data.tools || []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { load(); }, [load]);

  const filtered = useMemo(() => {
    if (!search) return tools;
    const q = search.toLowerCase();
    return tools.filter((t) => t.name?.toLowerCase().includes(q) || t.tool_name?.toLowerCase().includes(q) || t.description?.toLowerCase().includes(q));
  }, [tools, search]);

  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } = usePagination(filtered);

  const toggleActive = async (tool: ManagedTool) => {
    try {
      await adminApi.updateTool(tool.id, { is_active: !tool.is_active });
      await load();
    } catch {
      setTools((prev) => prev.map((t) => t.id === tool.id ? { ...t, is_active: !t.is_active } : t));
    }
  };

  const loadPerms = async (tool: ManagedTool) => {
    try {
      const data = await adminApi.getToolPermissions(tool.id);
      setPerms(data);
      setIsPublic(data.groups.some((g: { name: string }) => g.name === 'all'));
    } catch {
      setPerms({ users: [], groups: [], usecases: [] });
      setIsPublic(false);
    }
  };

  const openPerms = async (tool: ManagedTool) => {
    setSelectedTool(tool);
    await loadPerms(tool);
  };

  const togglePublic = async (checked: boolean) => {
    if (!selectedTool) return;
    const groupsData = await adminApi.getGroups();
    const allGroup = (groupsData.groups || []).find((g: { name: string; source: string }) => g.name === 'all' && g.source === 'auto');
    if (!allGroup) return;
    if (checked) {
      await adminApi.addToolGroup(selectedTool.id, allGroup.id);
    } else {
      await adminApi.removeToolGroup(selectedTool.id, allGroup.id);
    }
    setIsPublic(checked);
    await loadPerms(selectedTool);
  };

  if (loading) return <TableSkeleton rows={5} cols={4} />;

  return (
    <>
      <AdminToolbar
        left={<SearchBox value={search} onChange={(v: string) => { setSearch(v); setPage(0); }} placeholder="ツール名、説明で検索..." />}
      />
      {filtered.length === 0 ? (
        <EmptyState icon={Wrench} message="ツールが見つかりません" />
      ) : (
        <CardTable>
          <Table>
            <Thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ツール名</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">説明</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">状態</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">操作</th>
              </tr>
            </Thead>
            <Tbody>
              {paged.map((t) => (
                <tr key={t.id}>
                  <td className="px-4 py-3">
                    <div className="text-sm font-medium text-default">{t.name}</div>
                    <div className="text-xs text-muted font-mono">{t.tool_name}</div>
                  </td>
                  <td className="px-4 py-3 text-sm text-default">{t.description || '-'}</td>
                  <td className="px-4 py-3">
                    <Toggle checked={t.is_active} onChange={() => toggleActive(t)} />
                  </td>
                  <td className="px-4 py-3">
                    <Button variant="ghost" size="sm" onClick={() => openPerms(t)}>権限</Button>
                  </td>
                </tr>
              ))}
            </Tbody>
          </Table>
        </CardTable>
      )}
      <Pagination page={page} total={total} setPage={setPage} pageSize={pageSize} totalItems={totalItems} onPageSizeChange={changePageSize} />

      <PermissionModal
        isOpen={!!selectedTool}
        onClose={() => setSelectedTool(null)}
        title={`${selectedTool?.name || ''} - 権限管理`}
        subtitle="ユーザーやグループを追加してアクセス権を管理します"
        users={perms.users}
        groups={perms.groups}
        isPublic={isPublic}
        onAddUser={async (userId) => { if (selectedTool) { await adminApi.addToolUser(selectedTool.id, userId); await loadPerms(selectedTool); } }}
        onRemoveUser={async (userId) => { if (selectedTool) { await adminApi.removeToolUser(selectedTool.id, userId); await loadPerms(selectedTool); } }}
        onAddGroup={async (groupId) => { if (selectedTool) { await adminApi.addToolGroup(selectedTool.id, groupId); await loadPerms(selectedTool); } }}
        onRemoveGroup={async (groupId) => { if (selectedTool) { await adminApi.removeToolGroup(selectedTool.id, groupId); await loadPerms(selectedTool); } }}
        onTogglePublic={togglePublic}
      />
    </>
  );
}

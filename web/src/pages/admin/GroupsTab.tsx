import { useState, useEffect, useCallback, useMemo } from 'react';
import { Users, Trash2, Plus, X, Search, User } from 'lucide-react';
import { Button, Table, Thead, Tbody, Badge, Modal, Input, usePagination, Pagination, SearchBox, CardTable, EmptyState, TableSkeleton } from '../../components/ui';
import * as adminApi from '../../services/adminApi';
import api from '../../services/api';
import { AdminToolbar } from './AdminToolbar';
import type { Group } from '../../types/group';
import type { GroupMember, SearchUser } from '../../types/user';

const MEMBER_PAGE_SIZE = 10;

export default function GroupsTab() {
  const [groups, setGroups] = useState<Group[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDesc, setNewDesc] = useState('');
  const [editGroup, setEditGroup] = useState<Group | null>(null);
  const [editName, setEditName] = useState('');
  const [editDesc, setEditDesc] = useState('');
  const [memberGroup, setMemberGroup] = useState<Group | null>(null);
  const [members, setMembers] = useState<GroupMember[]>([]);
  const [memberPage, setMemberPage] = useState(0);
  const [memberSearch, setMemberSearch] = useState('');
  const [addSearch, setAddSearch] = useState('');
  const [addResults, setAddResults] = useState<SearchUser[]>([]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const data = await adminApi.getGroups();
      setGroups(data.groups || []);
    } finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);

  const visible = useMemo(() => groups.filter((g) => g.source !== 'auto'), [groups]);
  const filtered = useMemo(() => {
    if (!search) return visible;
    const q = search.toLowerCase();
    return visible.filter((g) => g.name?.toLowerCase().includes(q) || g.description?.toLowerCase().includes(q));
  }, [visible, search]);
  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } = usePagination(filtered);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    await adminApi.createGroup(newName, newDesc || undefined);
    setShowCreate(false); setNewName(''); setNewDesc('');
    await load();
  };

  const openEdit = (group: Group) => {
    setEditGroup(group);
    setEditName(group.name || '');
    setEditDesc(group.description || '');
  };

  const handleSaveEdit = async () => {
    if (!editGroup) return;
    const updates: { name?: string; description?: string } = {};
    if (editName !== editGroup.name) updates.name = editName;
    if (editDesc !== (editGroup.description || '')) updates.description = editDesc;
    if (Object.keys(updates).length === 0) { setEditGroup(null); return; }
    await adminApi.updateGroup(editGroup.id, updates);
    setEditGroup(null);
    await load();
  };

  const handleDelete = async (groupId: string) => {
    if (!confirm('このグループを削除しますか？')) return;
    await adminApi.deleteGroup(groupId);
    setEditGroup(null);
    await load();
  };

  const openMembers = async (group: Group, e?: React.MouseEvent) => {
    e?.stopPropagation();
    setMemberGroup(group);
    setMemberPage(0); setMemberSearch(''); setAddSearch(''); setAddResults([]);
    const data = await adminApi.getGroupMembers(group.id);
    setMembers(data.members || []);
  };

  const removeMember = async (userId: string) => {
    if (!memberGroup) return;
    const newIds = members.filter((m) => m.source === 'manual' && m.id !== userId).map((m) => m.id);
    await adminApi.updateGroupMembers(memberGroup.id, newIds);
    await openMembers(memberGroup);
    await load();
  };

  useEffect(() => {
    if (!addSearch || addSearch.length < 2) { setAddResults([]); return; }
    const timer = setTimeout(async () => {
      try {
        const r = await api.get(`/user/search?q=${encodeURIComponent(addSearch)}`);
        setAddResults(r.data.users || []);
      } catch { setAddResults([]); }
    }, 300);
    return () => clearTimeout(timer);
  }, [addSearch]);

  const addMember = async (userId: string) => {
    if (!memberGroup) return;
    const currentIds = members.filter((m) => m.source === 'manual').map((m) => m.id);
    if (!currentIds.includes(userId)) {
      await adminApi.updateGroupMembers(memberGroup.id, [...currentIds, userId]);
      await openMembers(memberGroup);
      await load();
    }
    setAddSearch(''); setAddResults([]);
  };

  const memberIds = new Set(members.map((m) => m.id));
  const addableResults = addResults.filter((u) => !memberIds.has(u.id));
  const filteredMembers = useMemo(() => {
    if (!memberSearch) return members;
    const q = memberSearch.toLowerCase();
    return members.filter((m) => m.email?.toLowerCase().includes(q) || m.display_name?.toLowerCase().includes(q));
  }, [members, memberSearch]);
  const memberTotal = Math.ceil(filteredMembers.length / MEMBER_PAGE_SIZE);
  const pagedMembers = filteredMembers.slice(memberPage * MEMBER_PAGE_SIZE, (memberPage + 1) * MEMBER_PAGE_SIZE);

  if (loading) return <TableSkeleton rows={5} cols={5} />;

  const isEditable = (g: Group) => g.source !== 'auto' && g.source !== 'idp';

  return (
    <>
      <AdminToolbar
        left={<SearchBox value={search} onChange={(v: string) => { setSearch(v); setPage(0); }} placeholder="グループ名で検索..." />}
        right={<Button size="sm" onClick={() => setShowCreate(true)}><Plus size={14} className="mr-1 inline" />グループ作成</Button>}
      />
      {filtered.length === 0 ? (
        <EmptyState icon={Users} message="グループが見つかりません" action={
          <Button size="sm" onClick={() => setShowCreate(true)}><Plus size={14} className="mr-1 inline" />グループ作成</Button>
        } />
      ) : (
        <CardTable>
          <Table>
            <Thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">名前</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">説明</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ソース</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">メンバー数</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">操作</th>
              </tr>
            </Thead>
            <Tbody>
              {paged.map((g) => (
                <tr key={g.id} className={isEditable(g) ? 'cursor-pointer' : ''} onClick={() => isEditable(g) && openEdit(g)}>
                  <td className="px-4 py-3 text-sm">{g.name}</td>
                  <td className="px-4 py-3 text-sm text-muted">{g.description || '-'}</td>
                  <td className="px-4 py-3"><Badge color={g.source === 'idp' ? 'blue' : 'gray'}>{g.source}</Badge></td>
                  <td className="px-4 py-3 text-sm">{g.member_count}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-1" onClick={(e) => e.stopPropagation()}>
                      <Button variant="ghost" size="sm" onClick={(e) => openMembers(g, e)}>メンバー管理</Button>
                      {isEditable(g) && (
                        <button className="p-1 text-danger hover:text-danger-hover" onClick={() => handleDelete(g.id)} title="削除"><Trash2 size={16} /></button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </Tbody>
          </Table>
        </CardTable>
      )}
      <Pagination page={page} total={total} setPage={setPage} pageSize={pageSize} totalItems={totalItems} onPageSizeChange={changePageSize} />

      {/* 作成モーダル */}
      <Modal isOpen={showCreate} onClose={() => setShowCreate(false)} className="p-6 w-96">
        <h2 className="text-lg font-bold mb-4">グループ作成</h2>
        <form onSubmit={(e) => e.preventDefault()} className="space-y-3">
          <Input placeholder="グループ名" value={newName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewName(e.target.value)} />
          <Input placeholder="説明（任意）" value={newDesc} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewDesc(e.target.value)} />
          <div className="flex gap-2 justify-end">
            <Button variant="secondary" size="sm" onClick={() => setShowCreate(false)}>キャンセル</Button>
            <Button size="sm" onClick={handleCreate}>作成</Button>
          </div>
        </form>
      </Modal>

      {/* 編集モーダル */}
      <Modal isOpen={!!editGroup} onClose={() => setEditGroup(null)} className="p-6 w-96">
        <h2 className="text-lg font-bold mb-4">グループ編集</h2>
        <form onSubmit={(e) => e.preventDefault()} className="space-y-3">
          <div>
            <label className="text-xs font-medium text-muted">グループ名</label>
            <Input value={editName} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditName(e.target.value)} />
          </div>
          <div>
            <label className="text-xs font-medium text-muted">説明</label>
            <Input value={editDesc} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setEditDesc(e.target.value)} placeholder="説明（任意）" />
          </div>
          <div className="flex gap-2 justify-end">
            <Button variant="secondary" size="sm" onClick={() => setEditGroup(null)}>キャンセル</Button>
            <Button size="sm" onClick={handleSaveEdit}>保存</Button>
          </div>
        </form>
      </Modal>

      {/* メンバー管理モーダル */}
      <Modal isOpen={!!memberGroup} onClose={() => setMemberGroup(null)} className="p-6 w-[500px] max-h-[80vh] overflow-y-auto">
        <h2 className="text-lg font-bold mb-3">{memberGroup?.name} - メンバー</h2>
        {memberGroup?.source === 'idp' && (
          <p className="text-xs text-muted mb-3 bg-surface rounded-lg px-3 py-2">このグループは IdP（AD）から同期されています。メンバーの追加・削除は IdP 側で管理してください。</p>
        )}
        {memberGroup?.source !== 'idp' && (
          <>
            <div className="relative mb-3">
              <Search size={16} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-neutral-400" />
              <input type="text" value={addSearch} onChange={(e) => setAddSearch(e.target.value)} placeholder="ユーザーを検索して追加..." className="w-full pl-8 pr-3 py-2 border border-default rounded-lg text-sm bg-bg" />
            </div>
            {addSearch.length >= 2 && addableResults.length > 0 && (
              <div className="border border-default rounded-lg mb-3 max-h-40 overflow-y-auto">
                {addableResults.map((u) => (
                  <button key={u.id} onClick={() => addMember(u.id)} className="flex items-center justify-between w-full px-3 py-2 hover:bg-surface text-left text-sm">
                    <div className="flex items-center gap-2"><User size={14} className="text-muted" /><span>{u.email}</span>{u.display_name && <span className="text-xs text-muted ml-1">{u.display_name}</span>}</div>
                    <Plus size={14} className="text-neutral-400" />
                  </button>
                ))}
              </div>
            )}
            {addSearch.length >= 2 && addableResults.length === 0 && (
              <p className="text-xs text-muted mb-3">該当するユーザーが見つかりません</p>
            )}
          </>
        )}
        <div className="border-t border-default pt-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-muted">メンバー一覧（{filteredMembers.length}人）</span>
            {members.length > MEMBER_PAGE_SIZE && (
              <div className="relative">
                <Search size={14} className="absolute left-2 top-1/2 -translate-y-1/2 text-neutral-400" />
                <input type="text" value={memberSearch} onChange={(e) => { setMemberSearch(e.target.value); setMemberPage(0); }} placeholder="絞り込み..." className="pl-7 pr-2 py-1 border border-default rounded text-xs w-40 bg-bg" />
              </div>
            )}
          </div>
          {pagedMembers.length === 0 ? (
            <p className="text-sm text-muted py-2">メンバーがいません</p>
          ) : (
            <div className="space-y-1">
              {pagedMembers.map((m) => (
                <div key={m.id} className="flex items-center justify-between p-2 rounded hover:bg-surface">
                  <div className="flex items-center gap-2">
                    <User size={14} className="text-muted" />
                    <span className="text-sm">{m.email}</span>
                    {m.display_name && <span className="text-xs text-muted">{m.display_name}</span>}
                  </div>
                  {memberGroup?.source !== 'idp' && (
                    <button className="p-1 text-danger hover:text-danger-hover" onClick={() => removeMember(m.id)} title="除外"><X size={14} /></button>
                  )}
                </div>
              ))}
            </div>
          )}
          {memberTotal > 1 && (
            <div className="flex items-center gap-2 mt-2 justify-end text-xs">
              <button disabled={memberPage === 0} onClick={() => setMemberPage(memberPage - 1)} className="px-2 py-0.5 rounded border border-default disabled:opacity-40">前</button>
              <span>{memberPage + 1} / {memberTotal}</span>
              <button disabled={memberPage >= memberTotal - 1} onClick={() => setMemberPage(memberPage + 1)} className="px-2 py-0.5 rounded border border-default disabled:opacity-40">次</button>
            </div>
          )}
        </div>
        <div className="mt-4 flex justify-end">
          <Button variant="secondary" size="sm" onClick={() => setMemberGroup(null)}>閉じる</Button>
        </div>
      </Modal>
    </>
  );
}

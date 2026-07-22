import { useState, useEffect, useMemo, useRef } from 'react';
import { Search, Plus, X, User, Users, Crown, Eye, Pencil, Building2 } from 'lucide-react';
import { Button, Badge, Modal } from '../ui';
import api from '../../services/api';
import type { SearchResult } from '../../types/group';

interface PermissionUser {
  id: string;
  email: string;
  display_name?: string;
  permission?: string;
}

interface PermissionGroup {
  id: string;
  name: string;
  permission?: string;
}

interface Owner {
  id: string;
  email: string;
  display_name?: string;
}

interface PermissionLevel {
  value: string;
  label: string;
}

interface PermissionModalProps {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  owners?: Owner[];
  users: PermissionUser[];
  groups: PermissionGroup[];
  isPublic: boolean;
  userPermissionLevels?: PermissionLevel[] | null;
  groupPermissionLevels?: PermissionLevel[] | null;
  onAddUser: (userId: string) => Promise<void>;
  onRemoveUser: (userId: string) => Promise<void>;
  onUpdateUserPermission?: (userId: string, permission: string) => Promise<void>;
  onAddGroup: (groupId: string) => Promise<void>;
  onRemoveGroup: (groupId: string) => Promise<void>;
  onUpdateGroupPermission?: (groupId: string, permission: string) => Promise<void>;
  onTogglePublic: (checked: boolean) => Promise<void>;
}

const PERM_CONFIG: Record<string, { icon: typeof Eye; color: string; label: string }> = {
  viewer: { icon: Eye, color: 'text-neutral-500', label: '閲覧者' },
  editor: { icon: Pencil, color: 'text-info', label: '編集者' },
  owner: { icon: Crown, color: 'text-warning', label: 'オーナー' },
};

/** アイコンのみ表示 + ホバーでツールチップ + クリックでドロップダウン */
function PermSelector({ value, levels, onChange }: {
  value: string;
  levels: PermissionLevel[];
  onChange: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const config = PERM_CONFIG[value] || PERM_CONFIG.viewer;
  const Icon = config.icon;

  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 px-2 py-1 rounded hover:bg-neutral-100 transition-colors text-sm"
        title={config.label}
      >
        <Icon size={14} className={config.color} />
        <span className="text-xs text-muted">{config.label}</span>
      </button>
      {open && (
        <div className="absolute right-0 top-full mt-1 z-30 bg-bg border border-default rounded-lg shadow-lg py-1 min-w-[180px]">
          {levels.map((l) => {
            const lc = PERM_CONFIG[l.value] || PERM_CONFIG.viewer;
            const LIcon = lc.icon;
            const selected = value === l.value;
            return (
              <button
                key={l.value}
                onClick={() => { onChange(l.value); setOpen(false); }}
                className={`flex items-center gap-2.5 w-full px-3 py-2 text-sm transition-colors
                  ${selected ? 'bg-surface font-medium' : 'hover:bg-neutral-50'}`}
              >
                <LIcon size={15} className={lc.color} />
                <span className="text-default">{l.label}</span>
                {selected && <span className="ml-auto text-primary text-xs">✓</span>}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

type MemberTab = 'users' | 'groups';

export function PermissionModal({
  isOpen, onClose, title, subtitle,
  owners,
  users, groups, isPublic,
  userPermissionLevels, groupPermissionLevels,
  onAddUser, onRemoveUser, onUpdateUserPermission,
  onAddGroup, onRemoveGroup, onUpdateGroupPermission,
  onTogglePublic,
}: PermissionModalProps) {
  const [search, setSearch] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult>({ users: [], groups: [] });
  const [memberTab, setMemberTab] = useState<MemberTab>('users');

  useEffect(() => {
    if (isOpen) { setSearch(''); setSearchResults({ users: [], groups: [] }); setMemberTab('users'); }
  }, [isOpen]);

  useEffect(() => {
    if (!search || search.length < 2) { setSearchResults({ users: [], groups: [] }); return; }
    const timer = setTimeout(async () => {
      try {
        const r = await api.get<SearchResult>(`/user/search?q=${encodeURIComponent(search)}`);
        setSearchResults({ users: r.data.users || [], groups: r.data.groups || [] });
      } catch { setSearchResults({ users: [], groups: [] }); }
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  const existingUserIds = new Set(users.map((u) => u.id));
  const existingGroupIds = new Set(groups.map((g) => g.id));
  const ownerIds = new Set((owners || []).map((o) => o.id));
  const visibleGroups = groups.filter((g) => g.name !== 'all');
  const visibleUsers = users.filter((u) => !ownerIds.has(u.id));
  const allGroup = groups.find((g) => g.name === 'all');

  const filteredSearchUsers = useMemo(
    () => searchResults.users.filter((u) => !existingUserIds.has(u.id) && !ownerIds.has(u.id)),
    [searchResults.users, existingUserIds, ownerIds]
  );
  const filteredSearchGroups = useMemo(
    () => searchResults.groups.filter((g) => !existingGroupIds.has(g.id)),
    [searchResults.groups, existingGroupIds]
  );

  const handleAddUser = async (userId: string) => {
    await onAddUser(userId);
    setSearch('');
    setSearchResults({ users: [], groups: [] });
  };

  const handleAddGroup = async (groupId: string) => {
    await onAddGroup(groupId);
    setSearch('');
    setSearchResults({ users: [], groups: [] });
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="p-6 w-[520px] max-h-[80vh] overflow-y-auto">
      <h2 className="text-lg font-bold mb-1">{title}</h2>
      {subtitle && <p className="text-sm text-muted mb-4">{subtitle}</p>}
      {!subtitle && <div className="mb-4" />}

      {/* 検索 */}
      <div className="relative mb-2">
        <Search size={16} className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-400" />
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="ユーザーまたはグループを追加..."
          className="w-full pl-9 pr-3 py-2 border border-default rounded-lg text-sm bg-bg"
        />
      </div>

      {/* 検索結果 */}
      {search.length >= 2 && (filteredSearchUsers.length > 0 || filteredSearchGroups.length > 0) && (
        <div className="border border-default rounded-lg mb-4 max-h-48 overflow-y-auto">
          {filteredSearchUsers.map((u) => (
            <button key={`u-${u.id}`} onClick={() => handleAddUser(u.id)} className="flex items-center justify-between w-full px-3 py-2 hover:bg-surface text-left text-sm">
              <span className="flex items-center gap-2">
                <User size={14} className="text-muted" />
                {u.email}
                {u.display_name && <span className="text-xs text-muted">({u.display_name})</span>}
              </span>
              <Plus size={14} className="text-neutral-400" />
            </button>
          ))}
          {filteredSearchGroups.map((g) => (
            <button key={`g-${g.id}`} onClick={() => handleAddGroup(g.id)} className="flex items-center justify-between w-full px-3 py-2 hover:bg-surface text-left text-sm">
              <span className="flex items-center gap-2">
                <Users size={14} className="text-info" />
                {g.name}
              </span>
              <Plus size={14} className="text-neutral-400" />
            </button>
          ))}
        </div>
      )}
      {search.length >= 2 && filteredSearchUsers.length === 0 && filteredSearchGroups.length === 0 && (
        <p className="text-sm text-muted mb-4">該当するユーザー・グループが見つかりません</p>
      )}

      {/* 全員公開 */}
      {isPublic && (
        <div className="border-b border-default pb-3 mb-3">
          <div className="flex items-center justify-between py-1.5">
            <div className="flex items-center gap-2.5">
              <div className="w-7 h-7 rounded-full bg-info-light flex items-center justify-center">
                <Building2 size={14} className="text-info" />
              </div>
              <div>
                <span className="text-sm font-medium text-default">全員に共有</span>
                <span className="text-xs text-muted ml-1.5">
                  ({allGroup?.permission === 'editor' ? '編集者' : '閲覧者'})
                </span>
              </div>
            </div>
            <button onClick={() => onTogglePublic(false)} className="p-1.5 rounded hover:bg-neutral-100 text-neutral-400 hover:text-danger transition-colors" title="共有を解除">
              <X size={14} />
            </button>
          </div>
        </div>
      )}

      {/* オーナー */}
      {owners && owners.length > 0 && (
        <div className="mb-3">
          {owners.map((o) => (
            <div key={o.id} className="flex items-center gap-2.5 py-1.5">
              <div className="w-7 h-7 rounded-full bg-warning-light flex items-center justify-center">
                <Crown size={14} className="text-warning" />
              </div>
              <div className="flex-1 min-w-0">
                <span className="text-sm font-medium truncate">{o.email}</span>
                {o.display_name && <span className="text-xs text-muted ml-1.5">({o.display_name})</span>}
              </div>
              <span className="text-xs text-muted">オーナー</span>
            </div>
          ))}
        </div>
      )}

      {/* メンバータブ */}
      <div className="flex gap-1 border-b border-default mb-3">
        <button
          onClick={() => setMemberTab('users')}
          className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors relative
            ${memberTab === 'users' ? 'text-primary' : 'text-muted hover:text-default'}`}
        >
          <User size={14} />
          ユーザー
          {visibleUsers.length > 0 && <Badge color="gray" className="text-xs ml-0.5">{visibleUsers.length}</Badge>}
          {memberTab === 'users' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-t" />}
        </button>
        <button
          onClick={() => setMemberTab('groups')}
          className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors relative
            ${memberTab === 'groups' ? 'text-primary' : 'text-muted hover:text-default'}`}
        >
          <Users size={14} />
          グループ
          {visibleGroups.length > 0 && <Badge color="gray" className="text-xs ml-0.5">{visibleGroups.length}</Badge>}
          {memberTab === 'groups' && <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-t" />}
        </button>
      </div>

      {/* ユーザータブ */}
      {memberTab === 'users' && (
        <div className="space-y-0.5 mb-3 min-h-[60px]">
          {visibleUsers.length === 0 ? (
            <p className="text-sm text-muted py-4 text-center">ユーザーが追加されていません</p>
          ) : (
            visibleUsers.map((u) => (
              <div key={u.id} className="flex items-center justify-between py-1.5 px-1 rounded hover:bg-surface transition-colors">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-7 h-7 rounded-full bg-neutral-100 flex items-center justify-center shrink-0">
                    <User size={14} className="text-muted" />
                  </div>
                  <span className="text-sm truncate">{u.email}</span>
                </div>
                <div className="flex items-center gap-0.5 shrink-0">
                  {userPermissionLevels && onUpdateUserPermission ? (
                    <PermSelector value={u.permission || 'viewer'} levels={userPermissionLevels} onChange={(v) => onUpdateUserPermission(u.id, v)} />
                  ) : (
                    <span className="flex items-center gap-1.5 px-2 py-1 text-xs text-muted"><Eye size={14} className="text-neutral-400" />閲覧者</span>
                  )}
                  <button onClick={() => onRemoveUser(u.id)} className="p-1.5 rounded hover:bg-neutral-100 text-neutral-400 hover:text-danger transition-colors" title="削除"><X size={14} /></button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* グループタブ */}
      {memberTab === 'groups' && (
        <div className="space-y-0.5 mb-3 min-h-[60px]">
          {visibleGroups.length === 0 ? (
            <p className="text-sm text-muted py-4 text-center">グループが追加されていません</p>
          ) : (
            visibleGroups.map((g) => (
              <div key={g.id} className="flex items-center justify-between py-1.5 px-1 rounded hover:bg-surface transition-colors">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="w-7 h-7 rounded-full bg-info-light flex items-center justify-center shrink-0">
                    <Users size={14} className="text-info" />
                  </div>
                  <span className="text-sm truncate">{g.name}</span>
                </div>
                <div className="flex items-center gap-0.5 shrink-0">
                  {groupPermissionLevels && onUpdateGroupPermission ? (
                    <PermSelector value={g.permission || 'viewer'} levels={groupPermissionLevels} onChange={(v) => onUpdateGroupPermission(g.id, v)} />
                  ) : (
                    <span className="flex items-center gap-1.5 px-2 py-1 text-xs text-muted"><Eye size={14} className="text-neutral-400" />閲覧者</span>
                  )}
                  <button onClick={() => onRemoveGroup(g.id)} className="p-1.5 rounded hover:bg-neutral-100 text-neutral-400 hover:text-danger transition-colors" title="削除"><X size={14} /></button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* 全ユーザーに公開する（未公開時のみ、一番下） */}
      {!isPublic && (
        <div className="border-t border-default mt-3 pt-3">
          <button onClick={() => onTogglePublic(true)} className="flex items-center gap-2.5 w-full py-2 px-3 text-sm text-muted hover:text-info hover:bg-info-light rounded-lg transition-colors">
            <Building2 size={16} />
            <span>全ユーザーに公開する</span>
          </button>
        </div>
      )}

      <div className="mt-4 flex justify-end">
        <Button variant="secondary" size="sm" onClick={onClose}>閉じる</Button>
      </div>
    </Modal>
  );
}

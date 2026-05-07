import { useState, useEffect, useCallback } from 'react';
import { User, Users, Layers, Wrench, Image } from 'lucide-react';
import { useAppContext } from '../contexts/AppContext';
import { Tabs } from '../components/ui';
import type { TabItem } from '../components/ui';
import { StatsCards } from './admin/StatsCards';
import NotFound from './NotFound';
import UsersTab from './admin/UsersTab';
import GroupsTab from './admin/GroupsTab';
import UsecasesTab from './admin/UsecasesTab';
import ToolsTab from './admin/ToolsTab';
import ImagesTab from './admin/ImagesTab';
import * as adminApi from '../services/adminApi';

type Tab = 'users' | 'groups' | 'usecases' | 'tools' | 'images';

const TAB_ITEMS: TabItem<Tab>[] = [
  { value: 'users', label: 'ユーザー', icon: User },
  { value: 'groups', label: 'グループ', icon: Users },
  { value: 'usecases', label: 'ユースケース', icon: Layers },
  { value: 'tools', label: 'ツール', icon: Wrench },
  { value: 'images', label: '全履歴', icon: Image },
];

interface Stats {
  users: number | null;
  groups: number | null;
  usecases: number | null;
  tools: number | null;
  images: number | null;
}

export default function Admin() {
  const { isAdmin } = useAppContext();
  const [tab, setTab] = useState<Tab>('users');
  const [stats, setStats] = useState<Stats>({
    users: null, groups: null, usecases: null, tools: null, images: null,
  });

  const loadStats = useCallback(async () => {
    const [usersRes, groupsRes, usecasesRes, toolsRes, imagesRes] = await Promise.allSettled([
      adminApi.getUsers(),
      adminApi.getGroups(),
      adminApi.getUsecases(),
      adminApi.getTools(),
      adminApi.getAllImages(),
    ]);
    setStats({
      users: usersRes.status === 'fulfilled' ? (usersRes.value.users?.length ?? 0) : null,
      groups: groupsRes.status === 'fulfilled' ? (groupsRes.value.groups?.filter((g: { source?: string }) => g.source !== 'auto')?.length ?? 0) : null,
      usecases: usecasesRes.status === 'fulfilled' ? (usecasesRes.value.usecases?.length ?? 0) : null,
      tools: toolsRes.status === 'fulfilled' ? (toolsRes.value.tools?.length ?? 0) : null,
      images: imagesRes.status === 'fulfilled' ? (imagesRes.value.images?.length ?? 0) : null,
    });
  }, []);

  useEffect(() => { loadStats(); }, [loadStats]);

  if (!isAdmin) return <NotFound />;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-default">管理者ページ</h1>
        <p className="text-sm text-muted mt-1">ユーザー、グループ、ユースケースの管理</p>
      </div>

      <StatsCards {...stats} onTabChange={(t) => setTab(t as Tab)} />

      <Tabs items={TAB_ITEMS} value={tab} onChange={setTab} />

      <div className="mt-6">
        {tab === 'users' && <UsersTab />}
        {tab === 'groups' && <GroupsTab />}
        {tab === 'usecases' && <UsecasesTab />}
        {tab === 'tools' && <ToolsTab />}
        {tab === 'images' && <ImagesTab />}
      </div>
    </div>
  );
}

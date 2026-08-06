import { User, Users, Layers, Wrench, Image } from 'lucide-react';
import { Skeleton } from '../../components/ui';
import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  icon: LucideIcon;
  label: string;
  value: number | null;
  onClick?: () => void;
}

function StatCard({ icon: Icon, label, value, onClick }: StatCardProps) {
  return (
    <div
      className={`rounded-xl border border-default shadow-sm p-4 bg-bg transition-colors ${onClick ? 'cursor-pointer hover:border-primary/30 hover:shadow-md' : ''}`}
      onClick={onClick}
    >
      <div className="flex items-center gap-3">
        <div className="p-2 rounded-lg bg-surface">
          <Icon size={20} className="text-muted" />
        </div>
        <div>
          <p className="text-xs text-muted">{label}</p>
          {value !== null ? (
            <p className="text-xl font-semibold text-default">{value.toLocaleString()}</p>
          ) : (
            <Skeleton className="h-6 w-12 mt-0.5" />
          )}
        </div>
      </div>
    </div>
  );
}

interface StatsCardsProps {
  users: number | null;
  groups: number | null;
  usecases: number | null;
  tools: number | null;
  images: number | null;
  onTabChange?: (tab: string) => void;
}

export function StatsCards({ users, groups, usecases, tools, images, onTabChange }: StatsCardsProps) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-6">
      <StatCard icon={User} label="ユーザー" value={users} onClick={() => onTabChange?.('users')} />
      <StatCard icon={Users} label="グループ" value={groups} onClick={() => onTabChange?.('groups')} />
      <StatCard icon={Layers} label="ユースケース" value={usecases} onClick={() => onTabChange?.('usecases')} />
      <StatCard icon={Wrench} label="ツール" value={tools} onClick={() => onTabChange?.('tools')} />
      <StatCard icon={Image} label="画像" value={images} onClick={() => onTabChange?.('images')} />
    </div>
  );
}

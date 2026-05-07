import type { LucideIcon } from 'lucide-react';
import { Inbox } from 'lucide-react';

interface EmptyStateProps {
  icon?: LucideIcon;
  message: string;
  action?: React.ReactNode;
}

export function EmptyState({ icon: Icon = Inbox, message, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-muted">
      <Icon size={40} strokeWidth={1.5} className="mb-3 text-neutral-300" />
      <p className="text-sm mb-3">{message}</p>
      {action}
    </div>
  );
}

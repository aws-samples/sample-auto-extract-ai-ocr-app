import type { LucideIcon } from 'lucide-react';

export interface TabItem<T extends string = string> {
  value: T;
  label: string;
  icon?: LucideIcon;
}

interface TabsProps<T extends string = string> {
  items: TabItem<T>[];
  value: T;
  onChange: (value: T) => void;
}

export function Tabs<T extends string>({ items, value, onChange }: TabsProps<T>) {
  return (
    <div className="flex gap-1 border-b border-default">
      {items.map((item) => {
        const active = value === item.value;
        const Icon = item.icon;
        return (
          <button
            key={item.value}
            onClick={() => onChange(item.value)}
            className={`flex items-center gap-1.5 px-3 py-2 text-sm font-medium transition-colors relative
              ${active ? 'text-primary' : 'text-muted hover:text-default'}`}
          >
            {Icon && <Icon size={16} />}
            {item.label}
            {active && (
              <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary rounded-t" />
            )}
          </button>
        );
      })}
    </div>
  );
}

interface AvatarProps {
  name?: string;
  size?: 'sm' | 'md' | 'lg';
  role?: string;
  className?: string;
}

const sizeMap = { sm: 'w-7 h-7 text-xs', md: 'w-9 h-9 text-sm', lg: 'w-12 h-12 text-default' };

const roleColorMap: Record<string, string> = {
  admin: 'bg-red-500 text-white',
  author: 'bg-blue-500 text-white',
  reader: 'bg-neutral-400 text-white',
};

export function Avatar({ name = '', size = 'md', role, className = '' }: AvatarProps) {
  const initial = name.charAt(0).toUpperCase() || '?';
  const color = role ? (roleColorMap[role] || roleColorMap.reader) : 'bg-primary text-on-primary';
  return (
    <div className={`rounded-full flex items-center justify-center font-medium ${sizeMap[size]} ${color} ${className}`}>
      {initial}
    </div>
  );
}

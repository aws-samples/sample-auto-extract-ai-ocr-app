interface AvatarProps {
  name?: string;
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeMap = { sm: 'w-7 h-7 text-xs', md: 'w-9 h-9 text-sm', lg: 'w-12 h-12 text-default' };

export function Avatar({ name = '', size = 'md', className = '' }: AvatarProps) {
  const initial = name.charAt(0).toUpperCase() || '?';
  return (
    <div
      className={`rounded-full bg-primary text-on-primary flex items-center justify-center font-medium ${sizeMap[size]} ${className}`}
    >
      {initial}
    </div>
  );
}

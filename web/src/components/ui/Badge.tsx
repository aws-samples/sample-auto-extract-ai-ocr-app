import { HTMLAttributes } from 'react';

const colorMap = {
  blue: 'bg-info-light text-info-text',
  green: 'bg-success-light text-success-text',
  red: 'bg-danger-light text-danger-text',
  yellow: 'bg-warning-light text-warning-text',
  gray: 'bg-surface-alt text-default',
  purple: 'bg-accent-light text-accent-text',
} as const;

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  color?: keyof typeof colorMap;
}

export function Badge({ color = 'gray', className = '', ...props }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${colorMap[color]} ${className}`}
      {...props}
    />
  );
}

import { HTMLAttributes } from 'react';

const typeStyles = {
  error: 'bg-danger-light border border-danger-border text-danger-text',
  success: 'bg-success-light border border-success-border text-success-text',
  warning: 'bg-warning-light border border-warning-border text-warning-text',
  info: 'bg-info-light border border-info-border text-info-text',
} as const;

export interface AlertProps extends HTMLAttributes<HTMLDivElement> {
  type?: keyof typeof typeStyles;
}

export function Alert({ type = 'info', className = '', ...props }: AlertProps) {
  return (
    <div
      role="alert"
      className={`px-4 py-3 rounded ${typeStyles[type]} ${className}`}
      {...props}
    />
  );
}

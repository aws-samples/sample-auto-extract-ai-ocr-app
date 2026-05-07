import { ButtonHTMLAttributes, forwardRef } from 'react';

const variants = {
  primary: 'bg-primary hover:bg-primary-hover text-on-primary',
  secondary: 'bg-neutral-200 hover:bg-neutral-300 text-neutral-800',
  danger: 'bg-danger hover:bg-danger-hover text-on-primary',
  success: 'bg-success hover:bg-success-hover text-on-primary',
  ghost: 'text-primary hover:text-primary-hover bg-transparent',
  outline: 'border border-default bg-transparent hover:bg-neutral-100 text-neutral-700',
} as const;

const sizes = {
  sm: 'px-2 py-1 text-sm',
  md: 'px-4 py-2',
  lg: 'px-6 py-3 text-lg',
} as const;

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: keyof typeof variants;
  size?: keyof typeof sizes;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ variant = 'primary', size = 'md', className = '', disabled, type = 'button', ...props }, ref) => (
    <button
      ref={ref}
      type={type}
      className={`inline-flex items-center justify-center rounded font-medium transition-colors ${variants[variant]} ${sizes[size]} ${disabled ? 'opacity-50 cursor-not-allowed' : ''} ${className}`}
      disabled={disabled}
      {...props}
    />
  ),
);

Button.displayName = 'Button';

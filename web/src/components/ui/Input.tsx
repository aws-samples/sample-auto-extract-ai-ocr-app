import { InputHTMLAttributes, TextareaHTMLAttributes, forwardRef } from 'react';

const base = 'border border-default rounded px-3 py-2 w-full focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent';

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(
  ({ className = '', ...props }, ref) => (
    <input ref={ref} className={`${base} ${className}`} {...props} />
  ),
);
Input.displayName = 'Input';

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(
  ({ className = '', ...props }, ref) => (
    <textarea ref={ref} className={`${base} ${className}`} {...props} />
  ),
);
Textarea.displayName = 'Textarea';

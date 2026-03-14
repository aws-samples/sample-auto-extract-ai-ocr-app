import { ReactNode, TableHTMLAttributes } from 'react';

interface TableProps extends TableHTMLAttributes<HTMLTableElement> {
  children: ReactNode;
}

export function Table({ className = '', children, ...props }: TableProps) {
  return (
    <table className={`min-w-full divide-y divide-default ${className}`} {...props}>
      {children}
    </table>
  );
}

export function Thead({ className = '', children, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <thead className={`bg-surface ${className}`} {...props}>{children}</thead>;
}

export function Tbody({ className = '', children, ...props }: React.HTMLAttributes<HTMLTableSectionElement>) {
  return <tbody className={`bg-bg divide-y divide-default ${className}`} {...props}>{children}</tbody>;
}

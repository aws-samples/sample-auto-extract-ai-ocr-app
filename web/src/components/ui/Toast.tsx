import React, { useEffect, useRef } from 'react';

interface ToastProps {
  message: string;
  type: 'success' | 'error' | 'info';
  show: boolean;
  onClose: () => void;
  duration?: number;
}

const Toast: React.FC<ToastProps> = ({
  message,
  type = 'info',
  show,
  onClose,
  duration = 3000
}) => {
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (show) {
      const timer = setTimeout(() => {
        onCloseRef.current();
      }, duration);

      return () => clearTimeout(timer);
    }
  }, [show, duration]);

  if (!show) return null;

  const bgColor = type === 'success' ? 'bg-success' : 
                 type === 'error' ? 'bg-danger' : 
                 'bg-primary';

  return (
    <div 
      className={`fixed top-4 right-4 px-6 py-3 rounded-md shadow-lg z-[9999] ${bgColor} text-on-primary font-medium animate-fade-in-down`}
      style={{
        minWidth: '250px',
        textAlign: 'center',
        animation: 'fadeIn 0.3s, fadeOut 0.3s 2.7s'
      }}
    >
      <div className="flex items-center justify-between">
        <span>{message}</span>
        <button 
          onClick={onClose}
          className="ml-4 text-on-primary hover:text-neutral-200 focus:outline-none"
        >
          ×
        </button>
      </div>
    </div>
  );
};

export default Toast;

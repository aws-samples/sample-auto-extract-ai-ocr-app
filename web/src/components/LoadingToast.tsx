import React from 'react';

interface LoadingToastProps {
  message: string;
  show: boolean;
}

const LoadingToast: React.FC<LoadingToastProps> = ({ message, show }) => {
  if (!show) return null;

  return (
    <div 
      className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 px-8 py-6 rounded-lg shadow-2xl z-[9999] bg-bg border-2 border-info"
      style={{
        minWidth: '400px',
        maxWidth: '600px',
      }}
    >
      <div className="flex flex-col items-center space-y-4">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-info"></div>
        <p className="text-neutral-800 text-center font-medium whitespace-pre-line">{message}</p>
      </div>
    </div>
  );
};

export default LoadingToast;

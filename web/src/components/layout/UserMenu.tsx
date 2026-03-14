import { useState, useRef, useEffect } from 'react';
import { useAuthenticator } from '@aws-amplify/ui-react';
import { useNavigate } from 'react-router-dom';
import { Avatar } from '../ui';
import { Button } from '../ui';

export function UserMenu() {
  const { user, signOut } = useAuthenticator();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const displayName = user?.signInDetails?.loginId ?? user?.username ?? '';

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen(!open)} className="focus:outline-none">
        <Avatar name={displayName} size="sm" />
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-48 bg-bg rounded-md shadow-lg py-1 z-50">
          <div className="px-4 py-2 text-sm text-neutral-700 border-b border-default">
            {displayName}
          </div>
          <button
            onClick={() => { navigate('/admin'); setOpen(false); }}
            className="block w-full text-left px-4 py-2 text-sm text-neutral-700 hover:bg-surface-alt"
          >
            管理画面
          </button>
          <Button
            variant="danger"
            size="sm"
            onClick={() => signOut()}
            className="w-full mt-1 mx-0 rounded-none rounded-b-md"
          >
            ログアウト
          </Button>
        </div>
      )}
    </div>
  );
}

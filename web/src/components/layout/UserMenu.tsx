import { useState, useRef, useEffect } from 'react';
import { useAuthenticator } from '@aws-amplify/ui-react';
import { useNavigate } from 'react-router-dom';
import { Shield, LogOut, User } from 'lucide-react';
import { useAppContext } from '../../contexts/AppContext';
import { Avatar, Modal, Input, Button } from '../ui';

function ProfileModal({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { currentUser, updateDisplayName } = useAppContext();
  const [value, setValue] = useState('');
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (isOpen) setValue(currentUser?.display_name || '');
  }, [isOpen, currentUser?.display_name]);

  const save = async () => {
    const trimmed = value.trim();
    if (!trimmed || trimmed === currentUser?.display_name) { onClose(); return; }
    setSaving(true);
    try {
      await updateDisplayName(trimmed);
      onClose();
    } finally {
      setSaving(false);
    }
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} className="w-80">
      <div className="p-5">
        <h3 className="text-sm font-semibold text-default mb-3">表示名の編集</h3>
        <Input
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter') save(); }}
          placeholder="表示名を入力"
        />
        <div className="flex justify-end gap-2 mt-4">
          <Button variant="secondary" onClick={onClose}>キャンセル</Button>
          <Button onClick={save} disabled={saving}>{saving ? '保存中...' : '保存'}</Button>
        </div>
      </div>
    </Modal>
  );
}

export function UserMenu() {
  const { user, signOut } = useAuthenticator();
  const { isAdmin, currentUser } = useAppContext();
  const [open, setOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();

  const displayName = currentUser?.display_name || user?.signInDetails?.loginId || user?.username || '';
  const email = currentUser?.email || user?.signInDetails?.loginId || '';

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
        <Avatar name={displayName} size="sm" role={currentUser?.role} />
      </button>
      {open && (
        <div className="absolute right-0 mt-2 w-56 bg-bg rounded-xl border border-default shadow-lg py-1 z-50">
          {/* ユーザー情報（クリックで編集モーダル） */}
          <button
            onClick={() => { setOpen(false); setProfileOpen(true); }}
            className="w-full text-left px-4 py-3 border-b border-default hover:bg-neutral-50 transition-colors"
          >
            <div className="flex items-center gap-2">
              <User size={14} className="text-muted flex-shrink-0" />
              <div className="min-w-0">
                <p className="text-sm font-medium text-default truncate">{displayName}</p>
                {email && email !== displayName && (
                  <p className="text-xs text-muted truncate">{email}</p>
                )}
              </div>
            </div>
          </button>

          {/* メニュー項目 */}
          {isAdmin && (
            <>
              <button
                onClick={() => { navigate('/admin'); setOpen(false); }}
                className="flex items-center gap-2 w-full text-left px-4 py-2 text-sm text-default hover:bg-neutral-50 transition-colors"
              >
                <Shield size={14} className="text-muted" />
                管理画面
              </button>
              <div className="border-t border-default my-1" />
            </>
          )}

          {/* ログアウト */}
          <button
            onClick={() => signOut()}
            className="flex items-center gap-2 w-full text-left px-4 py-2 text-sm text-danger hover:bg-danger-light transition-colors"
          >
            <LogOut size={14} />
            ログアウト
          </button>
        </div>
      )}
      <ProfileModal isOpen={profileOpen} onClose={() => setProfileOpen(false)} />
    </div>
  );
}

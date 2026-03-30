import { useState, useEffect } from 'react';
import { Shield } from 'lucide-react';
import { Modal, Badge, Button } from '../../components/ui';
import ConfirmModal from '../../components/shared/ConfirmModal';
import * as adminApi from '../../services/adminApi';
import type { User } from '../../types/user';

interface Props {
  user: User | null;
  onClose: () => void;
  onUpdated: (userId: string, newRole: string) => void;
}

export default function UserDetailModal({ user, onClose, onUpdated }: Props) {
  const [editRole, setEditRole] = useState('');
  const [saving, setSaving] = useState(false);
  const [showUnsavedConfirm, setShowUnsavedConfirm] = useState(false);

  useEffect(() => {
    if (user) setEditRole(user.role);
  }, [user]);

  const isDirty = user ? editRole !== user.role : false;

  const handleClose = () => {
    if (isDirty) {
      setShowUnsavedConfirm(true);
    } else {
      onClose();
    }
  };

  const handleSave = async () => {
    if (!user || !isDirty) return;
    setSaving(true);
    try {
      await adminApi.updateUserRole(user.id, editRole);
      onClose();
      onUpdated(user.id, editRole);
    } finally {
      setSaving(false);
    }
  };

  const handleDiscardAndClose = () => {
    setShowUnsavedConfirm(false);
    setEditRole(user?.role || '');
    onClose();
  };

  return (
    <>
      <Modal isOpen={!!user} onClose={handleClose} className="max-w-md w-full mx-4 p-6">
        {user && (
          <>
            <div className="space-y-4">
              {/* 表示名 + メール */}
              <div>
                <span className="text-base font-semibold text-default">{user.display_name || user.email}</span>
                {user.display_name && (
                  <span className="text-sm text-muted ml-2">{user.email}</span>
                )}
              </div>

              {/* 部署 */}
              {user.department && (
                <p className="text-xs text-muted -mt-2">{user.department}</p>
              )}

              {/* グループ */}
              <div>
                <label className="block text-xs font-medium text-muted mb-1">グループ</label>
                {user.groups && user.groups.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {user.groups.map((g) => (
                      <Badge key={g.name} color={g.source === 'idp' ? 'blue' : 'gray'}>{g.name}</Badge>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted">所属グループなし</p>
                )}
              </div>

              {/* ロール（一番下） */}
              <div>
                <label className="block text-xs font-medium text-muted mb-1">
                  <Shield size={14} className="inline-block mr-1 -mt-0.5" />ロール
                </label>
                <select
                  value={editRole}
                  onChange={(e) => setEditRole(e.target.value)}
                  className="border border-default rounded-lg px-3 py-1.5 text-sm bg-bg w-full"
                >
                  {(['admin', 'author', 'reader'] as const).map((r) => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
              </div>
            </div>

            <div className="flex justify-end gap-2 mt-6">
              <Button variant="secondary" size="sm" onClick={handleClose}>閉じる</Button>
              {isDirty && (
                <Button variant="primary" size="sm" onClick={handleSave} disabled={saving}>
                  {saving ? '保存中...' : '保存'}
                </Button>
              )}
            </div>
          </>
        )}
      </Modal>

      {/* 未保存確認 */}
      <ConfirmModal
        isOpen={showUnsavedConfirm}
        onClose={() => setShowUnsavedConfirm(false)}
        onConfirm={handleDiscardAndClose}
        title="未保存の変更"
        message="変更が保存されていません。破棄して閉じますか？"
        confirmText="破棄して閉じる"
        cancelText="戻る"
      />
    </>
  );
}

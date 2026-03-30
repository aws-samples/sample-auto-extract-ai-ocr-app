import { NavLink } from 'react-router-dom';
import { Home, Star, Clock, Shield } from 'lucide-react';
import { useAppContext } from '../../contexts/AppContext';

const items = [
  { to: '/', label: 'ホーム', icon: Home },
  { to: '/stars', label: 'お気に入り', icon: Star },
  { to: '/history', label: '処理履歴', icon: Clock },
];

export function Sidebar() {
  const { isAdmin } = useAppContext();

  return (
    <nav className="w-12 bg-bg border-r border-default flex flex-col items-center py-3 gap-1 shrink-0">
      {items.map(({ to, label, icon: Icon }) => (
        <NavLink
          key={to}
          to={to}
          end={to === '/'}
          className={({ isActive }) =>
            `relative group p-2 rounded-lg transition-colors ${
              isActive ? 'bg-primary/10 text-primary' : 'text-muted hover:text-default hover:bg-surface-alt'
            }`
          }
        >
          <Icon size={18} />
          <span className="absolute left-full ml-2 px-2 py-1 text-xs bg-neutral-800 text-white rounded whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
            {label}
          </span>
        </NavLink>
      ))}

      {isAdmin && (
        <>
          <div className="w-6 border-t border-default my-1" />
          <NavLink
            to="/admin"
            className={({ isActive }) =>
              `relative group p-2 rounded-lg transition-colors ${
                isActive ? 'bg-primary/10 text-primary' : 'text-muted hover:text-default hover:bg-surface-alt'
              }`
            }
          >
            <Shield size={18} />
            <span className="absolute left-full ml-2 px-2 py-1 text-xs bg-neutral-800 text-white rounded whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
              管理画面
            </span>
          </NavLink>
        </>
      )}
    </nav>
  );
}

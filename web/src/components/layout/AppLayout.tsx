import { Suspense } from 'react';
import { Outlet, Link, useParams } from 'react-router-dom';
import { useAppContext } from '../../contexts/AppContext';
import { UserMenu } from './UserMenu';
import { Sidebar } from './Sidebar';

export function AppLayout() {
  const params = useParams();
  const { apps: availableApps } = useAppContext();

  const currentAppName = params.appName || params['*']?.split('/')[0] || '';
  const currentAppDisplayName =
    availableApps?.find((a) => a.name === currentAppName)?.display_name || currentAppName;

  return (
    <div className="min-h-screen bg-surface flex flex-col w-full">
      <header
        className="bg-header text-on-primary py-2 flex justify-between items-center px-4"
        style={{ position: 'sticky', top: 0, zIndex: 1000 }}
      >
        <div className="flex items-center gap-2">
          <Link to="/" className="font-semibold text-sm hover:text-neutral-200">AutoExtract</Link>
          {currentAppName && (
            <>
              <span className="text-neutral-300 text-sm">/</span>
              <span className="text-sm">{currentAppDisplayName}</span>
            </>
          )}
        </div>
        <UserMenu />
      </header>

      <div className="flex flex-grow">
        <Sidebar />
        <main className="flex-grow overflow-auto">
          <Suspense fallback={<div className="flex justify-center items-center h-full py-20"><div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-info" /></div>}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  );
}

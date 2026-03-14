import { Outlet, Link, useParams } from 'react-router-dom';
import { useAppContext } from '../../contexts/AppContext';
import { UserMenu } from './UserMenu';

export function AppLayout() {
  const params = useParams();
  const { apps: availableApps } = useAppContext();

  const currentAppName = params.appName || params['*']?.split('/')[0] || '';
  const currentAppDisplayName =
    availableApps?.find((a) => a.name === currentAppName)?.display_name || currentAppName;

  return (
    <div className="min-h-screen bg-surface flex flex-col w-full">
      <header
        className="bg-header text-on-primary text-center py-4 flex justify-between items-center px-4"
        style={{ position: 'sticky', top: 0, zIndex: 1000 }}
      >
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center hover:text-neutral-200">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            <span className="ml-2 font-semibold">ホーム</span>
          </Link>
          {currentAppName && (
            <div className="flex items-center">
              <span className="mx-2 text-neutral-300">/</span>
              <span className="font-medium">{currentAppDisplayName}</span>
            </div>
          )}
        </div>
        <UserMenu />
      </header>

      <main className="flex-grow flex w-full">
        <Outlet />
      </main>
    </div>
  );
}

import { ReactNode } from 'react';
import { Link, useParams } from 'react-router-dom';
import { useAuthenticator } from '@aws-amplify/ui-react';
import { useAppContext } from './components/AppContext';

interface AppProps {
  children: ReactNode;
}

function App({ children }: AppProps) {
  const { signOut } = useAuthenticator();
  const params = useParams();
  const { apps: availableApps } = useAppContext();

  // Current app name
  const currentAppName = params.appName || '';

  // Current app display name
  const currentAppDisplayName = availableApps?.find(a => a.name === currentAppName)?.display_name || currentAppName;

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col w-full">
      <header className="bg-blue-600 text-white text-center py-4 flex justify-between items-center px-4" style={{ position: 'sticky', top: 0, zIndex: 1000 }}>
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center hover:text-gray-200">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
            <span className="ml-2 font-semibold">ホーム</span>
          </Link>
          
          {/* App name display (only on app pages) */}
          {currentAppName && (
            <div className="flex items-center">
              <span className="mx-2 text-gray-300">/</span>
              <span className="font-medium">{currentAppDisplayName}</span>
            </div>
          )}
        </div>
        <button onClick={() => signOut()} className="bg-red-500 text-white px-4 py-2 rounded">ログアウト</button>
      </header>

      <main className="flex-grow flex w-full">
        {children}
      </main>
    </div>
  );
}

export default App;

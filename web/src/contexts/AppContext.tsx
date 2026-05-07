import { createContext, useContext, ReactNode, useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import { AppSchema } from '../types/app-schema';

interface CurrentUser {
  id: string;
  email: string;
  display_name: string | null;
  department: string | null;
  role: string;
}

interface AppContextType {
  apps: AppSchema[];
  loading: boolean;
  error: string | null;
  refreshApps: () => Promise<void>;
  currentUser: CurrentUser | null;
  userLoaded: boolean;
  isAdmin: boolean;
  isAuthorOrAbove: boolean;
  stars: string[];
  toggleStar: (appName: string) => Promise<void>;
  updateDisplayName: (name: string) => Promise<void>;
}

const AppContext = createContext<AppContextType>({
  apps: [],
  loading: false,
  error: null,
  refreshApps: async () => {},
  currentUser: null,
  userLoaded: false,
  isAdmin: false,
  isAuthorOrAbove: false,
  stars: [],
  toggleStar: async () => {},
  updateDisplayName: async () => {},
});

export { AppContext };
export const useAppContext = () => useContext(AppContext);

export const AppProvider = ({ children }: { children: ReactNode }) => {
  const [apps, setApps] = useState<AppSchema[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [userLoaded, setUserLoaded] = useState(false);
  const [stars, setStars] = useState<string[]>([]);

  const fetchApps = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.get('/apps');
      setApps(response?.data?.apps || []);
    } catch (err) {
      console.error('Failed to fetch apps:', err);
      setError('アプリケーション情報の取得に失敗しました');
      setApps([]);
    } finally {
      setLoading(false);
    }
  };

  const toggleStar = useCallback(async (appName: string) => {
    const isStarred = stars.includes(appName);
    // 楽観的更新
    setStars(prev => isStarred ? prev.filter(s => s !== appName) : [...prev, appName]);
    try {
      if (isStarred) {
        await api.delete(`/user/stars/${appName}`);
      } else {
        await api.put(`/user/stars/${appName}`);
      }
    } catch {
      // ロールバック
      setStars(prev => isStarred ? [...prev, appName] : prev.filter(s => s !== appName));
    }
  }, [stars]);

  const updateDisplayName = useCallback(async (name: string) => {
    await api.patch('/user/me', { display_name: name });
    setCurrentUser(prev => prev ? { ...prev, display_name: name } : prev);
  }, []);

  useEffect(() => {
    fetchApps();
    api.get('/user/me')
      .then(r => setCurrentUser(r.data.user || null))
      .catch(() => setCurrentUser(null))
      .finally(() => setUserLoaded(true));
    api.get('/user/stars')
      .then(r => setStars(r.data.stars || []))
      .catch(() => {});
  }, []);

  return (
    <AppContext.Provider
      value={{
        apps,
        loading,
        error,
        refreshApps: fetchApps,
        currentUser,
        userLoaded,
        isAdmin: userLoaded && currentUser?.role === 'admin',
        isAuthorOrAbove: userLoaded && (currentUser?.role === 'admin' || currentUser?.role === 'author'),
        stars,
        toggleStar,
        updateDisplayName,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

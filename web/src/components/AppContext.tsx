import { createContext, useContext, ReactNode, useState, useEffect } from 'react';
import api from '../utils/api';
import { AppSchema } from '../types/app-schema';

interface AppContextType {
  apps: AppSchema[];
  loading: boolean;
  error: string | null;
  refreshApps: () => Promise<void>;
}

const AppContext = createContext<AppContextType>({
  apps: [],
  loading: false,
  error: null,
  refreshApps: async () => {},
});

export { AppContext };
export const useAppContext = () => useContext(AppContext);

interface AppProviderProps {
  children: ReactNode;
}

export const AppProvider = ({ children }: AppProviderProps) => {
  const [apps, setApps] = useState<AppSchema[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchApps = async () => {
    setLoading(true);
    setError(null);
    try {
      // リトライ機能を追加
      let retries = 3;
      let success = false;
      let response: any = null;
      
      while (retries > 0 && !success) {
        try {
          response = await api.get('/apps');
          success = true;
        } catch (err) {
          retries--;
          if (retries === 0) throw err;
          // リトライ前に少し待機
          await new Promise(resolve => setTimeout(resolve, 1000));
        }
      }
      
      console.log('API Response:', response);
      if (response && response.data) {
        setApps(response.data.apps || []);
      } else {
        setApps([]);
      }
    } catch (err) {
      console.error('Failed to fetch apps:', err);
      setError('アプリケーション情報の取得に失敗しました');
      // エラーが発生しても空の配列を設定して、UIが壊れないようにする
      setApps([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchApps();
  }, []);

  return (
    <AppContext.Provider
      value={{
        apps,
        loading,
        error,
        refreshApps: fetchApps,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

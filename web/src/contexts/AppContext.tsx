import { createContext, useContext, ReactNode, useState, useEffect, useCallback } from 'react';
import api from '../services/api';
import { AppSchema } from '../types/app-schema';
import { FullPageLoader } from '../components/ui';

interface CurrentUser {
  id: string;
  email: string;
  display_name: string | null;
  department: string | null;
  role: string;
}

// currentUser のローカルキャッシュキー。フォーマット変更時は suffix を v2 に上げて古い値を無効化する。
const CACHE_KEY = 'autoextract.currentUser.v1';

// localStorage からキャッシュされた currentUser を同期的に復元する。壊れた値は null 扱い。
const readUserCache = (): CurrentUser | null => {
  try {
    const raw = localStorage.getItem(CACHE_KEY);
    return raw ? (JSON.parse(raw) as CurrentUser) : null;
  } catch {
    return null;
  }
};

const writeUserCache = (user: CurrentUser) => {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(user));
  } catch {
    // ストレージ書き込み失敗（容量超過/プライベートモード等）は致命的でないため無視
  }
};

const clearUserCache = () => {
  try {
    localStorage.removeItem(CACHE_KEY);
  } catch {
    // 無視
  }
};

export { CACHE_KEY, clearUserCache };

interface AppContextType {
  apps: AppSchema[];
  loading: boolean;
  appsLoaded: boolean;
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
  appsLoaded: false,
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
  const [appsLoaded, setAppsLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(() => readUserCache());
  const [userLoaded, setUserLoaded] = useState<boolean>(() => readUserCache() !== null);
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
      setAppsLoaded(true);
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
    setCurrentUser(prev => {
      if (!prev) return prev;
      const next = { ...prev, display_name: name };
      writeUserCache(next);
      return next;
    });
  }, []);

  useEffect(() => {
    fetchApps();
    // SWR: キャッシュがあっても毎回 /user/me で再検証し、state とキャッシュを最新化する
    api.get('/user/me')
      .then(r => {
        const user: CurrentUser | null = r.data.user || null;
        setCurrentUser(user);
        if (user) writeUserCache(user);
        else clearUserCache();
      })
      .catch(err => {
        // 401: 認証切れ → キャッシュ削除して未ログイン扱い
        // それ以外（response なし=ネットワーク / 5xx 等）: キャッシュを維持し次回マウントで再検証
        if (err?.response?.status === 401) {
          clearUserCache();
          setCurrentUser(null);
        }
      })
      .finally(() => setUserLoaded(true));
    api.get('/user/stars')
      .then(r => setStars(r.data.stars || []))
      .catch(() => {});
  }, []);

  // クロスタブ同期: 別タブでキャッシュキーが削除されたら（= ログアウト）、
  // 自タブが user を保持している場合は reload して Authenticator のログイン画面に戻す。
  // setState(null) だけでは Authenticator は認証状態を見ているためログイン画面に戻らず、
  // ユーザー不在のアプリシェルが残る壊れた中間状態になるため reload にする。
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === CACHE_KEY && e.newValue === null && currentUser) {
        window.location.reload();
      }
    };
    window.addEventListener('storage', onStorage);
    return () => window.removeEventListener('storage', onStorage);
  }, [currentUser]);

  if (!userLoaded) {
    // ※ AppProvider は Authenticator の内側にあるため、未認証時はここに到達しない
    //   （未認証ユーザーには Amplify のログイン画面が表示される）。
    //   この分岐はキャッシュ無しでの初回ログイン直後など、/user/me 解決待ちの瞬間のみ。
    return <FullPageLoader />;
  }

  return (
    <AppContext.Provider
      value={{
        apps,
        loading,
        appsLoaded,
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

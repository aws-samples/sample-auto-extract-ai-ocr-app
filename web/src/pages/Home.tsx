import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { Alert } from '../components/ui';
import { Button } from '../components/ui';

function Home() {
  const navigate = useNavigate();
  const { apps: availableApps, loading, error } = useAppContext();
  
  // アプリ選択処理
  const selectApp = (appName: string) => {
    navigate(`/app/${appName}`);
  };
  
  // スキーマ生成ページへ遷移する関数を追加
  const navigateToSchemaGenerator = () => {
    navigate('/schema-generator');
  };
  
  // フォールバック用のデフォルト説明を取得
  const getDefaultDescription = () => {
    return '文書からの情報抽出を行います';
  };

  return (
    <div className="home-container bg-bg rounded-lg shadow-md">
      <h1 className="text-3xl font-bold mb-6 border-b pb-3 text-center text-neutral-800">アプリ一覧</h1>
      
      <div className="flex justify-between items-center mb-6 px-6">
        <p className="text-xl text-neutral-700">アプリケーションを選択してください</p>
        {/* 新規ユースケース追加ボタン */}
        <Button 
          variant="success"
          onClick={navigateToSchemaGenerator}
          className="rounded-lg flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          新規ユースケース追加
        </Button>
      </div>
      
      {error && (
        <Alert type="error" className="mb-4 mx-6">
          <span className="block sm:inline">{error}</span>
        </Alert>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 px-6 pb-6">
        {availableApps && availableApps.map(app => (
          <div 
            key={app.name}
            className="app-card bg-neutral-50 border border-neutral-200 rounded-lg p-6 hover:shadow-md transition-shadow cursor-pointer hover:border-info-border"
            onClick={() => selectApp(app.name)}
          >
            <div className="app-icon mb-4 bg-info-light text-info rounded-full w-16 h-16 flex items-center justify-center mx-auto">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold mb-2 text-center text-neutral-800">{app.display_name}</h2>
            <div className="text-sm text-neutral-600 mb-4 text-center">
              {app.description || getDefaultDescription()}
            </div>
            <div className="mt-4 text-center">
              <Button 
                variant="primary"
                className="rounded-lg"
                onClick={(e) => {
                  e.stopPropagation();
                  selectApp(app.name);
                }}
              >
                選択する
              </Button>
            </div>
          </div>
        ))}
      </div>
      
      {loading && (
        <div className="text-center py-10">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-info mx-auto mb-4"></div>
          <p className="text-neutral-700">アプリケーション情報を読み込み中...</p>
        </div>
      )}
      
      {!loading && availableApps && availableApps.length === 0 && !error && (
        <div className="text-center py-10">
          <p className="text-neutral-700">利用可能なアプリケーションがありません</p>
        </div>
      )}
    </div>
  );
}

export default Home;

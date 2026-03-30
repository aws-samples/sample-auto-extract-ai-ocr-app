import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { Alert } from '../components/ui';
import { Button } from '../components/ui';
import { Star, Plus, FileText } from 'lucide-react';

function Home() {
  const navigate = useNavigate();
  const { apps: availableApps, loading, error, stars, toggleStar, isAuthorOrAbove } = useAppContext();
  
  const selectApp = (appName: string) => {
    navigate(`/app/${appName}`);
  };
  
  const navigateToSchemaGenerator = () => {
    navigate('/schema-generator');
  };
  
  const getDefaultDescription = () => {
    return '文書からの情報抽出を行います';
  };

  return (
    <div className="p-6 w-full">
      <h1 className="text-2xl font-bold mb-6 text-neutral-800">アプリ一覧</h1>
      
      <div className="flex justify-between items-center mb-6">
        <p className="text-xl text-neutral-700">アプリケーションを選択してください</p>
        {isAuthorOrAbove && (
          <Button 
            variant="success"
            onClick={navigateToSchemaGenerator}
            className="rounded-lg flex items-center"
          >
            <Plus size={20} className="mr-2" />
            新規ユースケース追加
          </Button>
        )}
      </div>
      
      {error && (
        <Alert type="error" className="mb-4">
          <span className="block sm:inline">{error}</span>
        </Alert>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {availableApps && availableApps.map(app => {
          const starred = stars.includes(app.name);
          return (
            <div 
              key={app.name}
              className="app-card bg-neutral-50 border border-neutral-200 rounded-lg p-6 hover:shadow-md transition-shadow cursor-pointer hover:border-info-border relative"
              onClick={() => selectApp(app.name)}
            >
              <button
                className="absolute top-3 right-3 p-1"
                onClick={(e) => { e.stopPropagation(); toggleStar(app.name); }}
                title={starred ? 'お気に入り解除' : 'お気に入り追加'}
              >
                <Star size={18} className={starred ? 'fill-yellow-400 text-yellow-400' : 'text-neutral-300 hover:text-yellow-400'} />
              </button>
              <div className="app-icon mb-4 bg-info-light text-info rounded-full w-16 h-16 flex items-center justify-center mx-auto">
                <FileText size={32} />
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
          );
        })}
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

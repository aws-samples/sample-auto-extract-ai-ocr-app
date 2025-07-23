import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../components/AppContext';

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
  const getDefaultDescription = (appName: string) => {
    const descriptions: Record<string, string> = {
      'shipping_ocr': '輸送伝票や納品書の情報を抽出します',
      'invoice_ocr': '請求書や領収書の情報を抽出します',
      'receipt_ocr': 'レシートからの情報を抽出します',
      'inventory_management': '倉庫や店舗の在庫情報を管理します'
    };
    
    return descriptions[appName] || 'このアプリケーションでは文書からの情報抽出を行います';
  };

  return (
    <div className="home-container bg-white rounded-lg shadow-md">
      <h1 className="text-3xl font-bold mb-6 border-b pb-3 text-center text-gray-800">アプリ一覧</h1>
      
      <div className="flex justify-between items-center mb-6 px-6">
        <p className="text-xl text-gray-700">アプリケーションを選択してください</p>
        {/* 新規ユースケース追加ボタン */}
        <button 
          onClick={navigateToSchemaGenerator}
          className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors flex items-center"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
          </svg>
          新規ユースケース追加
        </button>
      </div>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative mb-4 mx-6" role="alert">
          <span className="block sm:inline">{error}</span>
        </div>
      )}
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 px-6 pb-6">
        {availableApps && availableApps.map(app => (
          <div 
            key={app.name}
            className="app-card bg-gray-50 border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow cursor-pointer hover:border-blue-300"
            onClick={() => selectApp(app.name)}
          >
            <div className="app-icon mb-4 bg-blue-100 text-blue-600 rounded-full w-16 h-16 flex items-center justify-center mx-auto">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h2 className="text-xl font-semibold mb-2 text-center text-gray-800">{app.display_name}</h2>
            <div className="text-sm text-gray-600 mb-4 text-center">
              {app.description || getDefaultDescription(app.name)}
            </div>
            <div className="mt-4 text-center">
              <button 
                className="bg-blue-500 hover:bg-blue-600 text-white py-2 px-4 rounded-lg transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  selectApp(app.name);
                }}
              >
                選択する
              </button>
            </div>
          </div>
        ))}
      </div>
      
      {loading && (
        <div className="text-center py-10">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-700">アプリケーション情報を読み込み中...</p>
        </div>
      )}
      
      {!loading && availableApps && availableApps.length === 0 && !error && (
        <div className="text-center py-10">
          <p className="text-gray-700">利用可能なアプリケーションがありません</p>
        </div>
      )}
    </div>
  );
}

export default Home;

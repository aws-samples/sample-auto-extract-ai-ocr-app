import { useNavigate } from 'react-router-dom';
import { useAppContext } from '../contexts/AppContext';
import { Button } from '../components/ui';
import { Star, FileText } from 'lucide-react';

export default function Stars() {
  const navigate = useNavigate();
  const { apps, stars, toggleStar } = useAppContext();
  const starredApps = apps.filter(app => stars.includes(app.name));

  return (
    <div className="p-6 w-full">
      <h1 className="text-2xl font-bold mb-6 text-neutral-800">お気に入り</h1>

      {starredApps.length === 0 ? (
        <p className="text-neutral-600">お気に入りに登録されたアプリはありません</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {starredApps.map(app => (
            <div
              key={app.name}
              className="app-card bg-neutral-50 border border-neutral-200 rounded-lg p-6 hover:shadow-md transition-shadow cursor-pointer hover:border-info-border relative"
              onClick={() => navigate(`/app/${app.name}`)}
            >
              <button
                className="absolute top-3 right-3 p-1"
                onClick={(e) => { e.stopPropagation(); toggleStar(app.name); }}
                title="お気に入り解除"
              >
                <Star size={18} className="fill-yellow-400 text-yellow-400" />
              </button>
              <div className="app-icon mb-4 bg-info-light text-info rounded-full w-16 h-16 flex items-center justify-center mx-auto">
                <FileText size={32} />
              </div>
              <h2 className="text-xl font-semibold mb-2 text-center text-neutral-800">{app.display_name}</h2>
              <div className="text-sm text-neutral-600 mb-4 text-center">
                {app.description || '文書からの情報抽出を行います'}
              </div>
              <div className="mt-4 text-center">
                <Button variant="primary" className="rounded-lg" onClick={(e) => { e.stopPropagation(); navigate(`/app/${app.name}`); }}>
                  選択する
                </Button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

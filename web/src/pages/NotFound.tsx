import { Link } from 'react-router-dom';
import { Button } from '../components/ui';

export default function NotFound() {
  return (
    <div className="flex flex-col items-center justify-center w-full py-20">
      <h1 className="text-6xl font-bold text-muted mb-4">404</h1>
      <p className="text-xl text-default mb-8">ページが見つかりません</p>
      <Link to="/">
        <Button variant="primary">ホームに戻る</Button>
      </Link>
    </div>
  );
}

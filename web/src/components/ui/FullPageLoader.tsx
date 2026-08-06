import { Loader2 } from 'lucide-react';

interface FullPageLoaderProps {
  message?: string;
}

/**
 * 認証情報の初回ロード中など、アプリ全体の描画をブロックして表示する全画面ローダー。
 * キャッシュがある場合は即描画されるため、これが出るのは初回ログイン直後など限定的。
 */
export function FullPageLoader({ message = '読み込み中...' }: FullPageLoaderProps) {
  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center gap-3">
      <Loader2 size={32} className="animate-spin text-primary" />
      <p className="text-sm text-muted">{message}</p>
    </div>
  );
}

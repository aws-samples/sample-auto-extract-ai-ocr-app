import { ReactNode, useEffect, useState } from 'react';
import { useAuthenticator } from '@aws-amplify/ui-react';
import { signInWithRedirect } from 'aws-amplify/auth';
import { Hub } from 'aws-amplify/utils';
import { LogIn } from 'lucide-react';
import { Alert, Button, FullPageLoader } from '../ui';
import { APP_CONFIG } from '../../config';

interface SamlSignInProps {
  children: ReactNode;
}

/** SAML 認証時のログイン画面。ボタンで IdP へリダイレクトする。 */
export function SamlSignIn({ children }: SamlSignInProps) {
  const { authStatus } = useAuthenticator((context) => [context.authStatus]);
  const [error, setError] = useState<string | null>(null);
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    return Hub.listen('auth', ({ payload }) => {
      if (payload.event === 'signInWithRedirect_failure') {
        setRedirecting(false);
        setError('ログインに失敗しました。時間をおいて再度お試しください。解決しない場合は管理者に連絡してください。');
      }
    });
  }, []);

  const signIn = async () => {
    setError(null);
    setRedirecting(true);
    try {
      await signInWithRedirect({
        provider: { custom: APP_CONFIG.samlCognitoFederatedIdentityProviderName },
      });
    } catch (err) {
      console.error('signInWithRedirect failed:', err);
      setRedirecting(false);
      setError('ログイン画面へ移動できませんでした。時間をおいて再度お試しください。');
    }
  };

  if (authStatus === 'configuring') {
    return <FullPageLoader />;
  }

  if (authStatus === 'authenticated') {
    return <>{children}</>;
  }

  return (
    <div className="min-h-screen bg-surface flex flex-col items-center justify-center gap-6 px-4">
      <h1 className="text-2xl font-semibold text-default">AutoExtract</h1>
      {error && (
        <Alert type="error" className="w-full max-w-sm text-sm">
          {error}
        </Alert>
      )}
      <Button variant="primary" onClick={signIn} disabled={redirecting} className="w-60 gap-2">
        <LogIn size={20} />
        {redirecting ? 'リダイレクト中...' : 'ログイン'}
      </Button>
    </div>
  );
}

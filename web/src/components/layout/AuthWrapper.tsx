import { ReactNode } from 'react';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import { APP_CONFIG } from '../../config';
import { SamlSignIn } from './SamlSignIn';

interface AuthWrapperProps {
  children: ReactNode;
}

export function AuthWrapper({ children }: AuthWrapperProps) {
  if (APP_CONFIG.samlAuthEnabled) {
    return (
      <Authenticator.Provider>
        <SamlSignIn>{children}</SamlSignIn>
      </Authenticator.Provider>
    );
  }

  return (
    <Authenticator
      hideSignUp={!APP_CONFIG.selfSignUpEnabled}
      variation="modal"
    >
      {() => <>{children}</>}
    </Authenticator>
  );
}

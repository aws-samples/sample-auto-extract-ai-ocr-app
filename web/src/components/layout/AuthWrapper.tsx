import { ReactNode } from 'react';
import { Authenticator } from '@aws-amplify/ui-react';
import '@aws-amplify/ui-react/styles.css';
import { APP_CONFIG } from '../../config';

interface AuthWrapperProps {
  children: ReactNode;
}

export function AuthWrapper({ children }: AuthWrapperProps) {
  return (
    <Authenticator
      hideSignUp={!APP_CONFIG.selfSignUpEnabled}
      variation="modal"
    >
      {() => <>{children}</>}
    </Authenticator>
  );
}

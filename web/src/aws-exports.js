const samlAuthEnabled = import.meta.env.VITE_APP_SAMLAUTH_ENABLED === 'true';

const awsConfig = {
  Auth: {
    Cognito: {
      userPoolId: import.meta.env.VITE_APP_USER_POOL_ID,
      userPoolClientId: import.meta.env.VITE_APP_USER_POOL_CLIENT_ID,
      region: import.meta.env.VITE_APP_REGION,
      ...(samlAuthEnabled && {
        loginWith: {
          oauth: {
            domain: import.meta.env.VITE_APP_SAML_COGNITO_DOMAIN_NAME,
            scopes: ['openid', 'email', 'profile'],
            redirectSignIn: [window.location.origin],
            redirectSignOut: [window.location.origin],
            responseType: 'code',
          },
        },
      }),
    },
  },
  API: {
    REST: {
      api: {
        endpoint: import.meta.env.VITE_API_BASE_URL,
      }
    }
  }
};

export default awsConfig;

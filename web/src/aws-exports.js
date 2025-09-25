const awsConfig = {
  Auth: {
    Cognito: {
      userPoolId: import.meta.env.VITE_APP_USER_POOL_ID,
      userPoolClientId: import.meta.env.VITE_APP_USER_POOL_CLIENT_ID,
      region: import.meta.env.VITE_APP_REGION,
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

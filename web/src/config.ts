// アプリケーション設定
export const APP_CONFIG = {
  // OCRモードの設定
  enableOcr: import.meta.env.VITE_ENABLE_OCR === 'true',

  // Cognito のセルフサインアップ許可。未設定時は true（従来の Authenticator デフォルト挙動を維持）
  selfSignUpEnabled: import.meta.env.VITE_SELF_SIGN_UP_ENABLED !== 'false',

  // その他の設定
  userPoolClientId: import.meta.env.VITE_APP_USER_POOL_CLIENT_ID,
  userPoolId: import.meta.env.VITE_APP_USER_POOL_ID,
  region: import.meta.env.VITE_APP_REGION,
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL,
} as const;

// OCRモードのチェック関数
export const isOcrEnabled = () => APP_CONFIG.enableOcr;

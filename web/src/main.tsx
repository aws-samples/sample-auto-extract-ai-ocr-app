import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { Amplify } from 'aws-amplify';
import { cognitoUserPoolsTokenProvider } from 'aws-amplify/auth/cognito';
import awsmobile from './aws-exports';
import router from './router';
import { AppProvider } from './components/AppContext';
import './index.css';
import './styles/index.css'; // アニメーション用のCSSを追加

// Configure Amplify
Amplify.configure(awsmobile);

// 認証トークンの自動更新を設定
cognitoUserPoolsTokenProvider.setKeyValueStorage({
  getItem: (key) => Promise.resolve(localStorage.getItem(key)),
  setItem: (key, value) => { localStorage.setItem(key, value); return Promise.resolve(); },
  removeItem: (key) => { localStorage.removeItem(key); return Promise.resolve(); },
  clear: () => { localStorage.clear(); return Promise.resolve(); }
});

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AppProvider>
      <RouterProvider router={router} />
    </AppProvider>
  </React.StrictMode>,
);

import { createBrowserRouter, Outlet } from 'react-router-dom';
import { Authenticator } from '@aws-amplify/ui-react';
import App from './App';
import Home from './pages/Home';
import Upload from './pages/Upload';
import OCRResult from './pages/OCRResult';
import SchemaGenerator from './pages/SchemaGenerator'; // 追加
import '@aws-amplify/ui-react/styles.css';

// Layout component that includes the App wrapper and authentication
const AppLayout = () => {
  return (
    <Authenticator>
      {() => (
        <App>
          <Outlet />
        </App>
      )}
    </Authenticator>
  );
};

// Define routes
const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      {
        path: 'app/:appName',
        element: <Upload />,
      },
      {
        path: 'ocr-result/:id',
        element: <OCRResult />,
      },
      {
        // 新規作成・編集共通のルート
        path: 'schema-generator',
        element: <SchemaGenerator mode="create" />,
      },
      {
        // 新規作成・編集共通のルート (appNameあり)
        path: 'schema-generator/:appName',
        element: <SchemaGenerator mode="edit" />,
      },
      {
        // 確認用
        path: 'apps/:appName/view',
        element: <SchemaGenerator mode="view" />,
      },
      {
        // 編集用 (schema-generatorにリダイレクト)
        path: 'apps/:appName/edit',
        element: <SchemaGenerator mode="edit" />,
      },
    ],
  },
]);

export default router;

import { createBrowserRouter } from 'react-router-dom';
import { lazy } from 'react';
import { AuthWrapper } from './components/layout/AuthWrapper';
import { AppLayout } from './components/layout/AppLayout';
import { AppProvider } from './contexts/AppContext';

// ルート単位でコード分割し、初回ロードの単一巨大チャンクを避ける。
const Home = lazy(() => import('./pages/Home'));
const Stars = lazy(() => import('./pages/Stars'));
const History = lazy(() => import('./pages/History'));
const Upload = lazy(() => import('./pages/upload/Upload'));
const OCRResult = lazy(() => import('./pages/ocr-result/OCRResult'));
const SchemaGenerator = lazy(() => import('./pages/schema/SchemaGenerator'));
const Admin = lazy(() => import('./pages/Admin'));
const NotFound = lazy(() => import('./pages/NotFound'));

const Root = () => (
  <AuthWrapper>
    <AppProvider>
      <AppLayout />
    </AppProvider>
  </AuthWrapper>
);

const router = createBrowserRouter([
  {
    path: '/',
    element: <Root />,
    children: [
      { index: true, element: <Home /> },
      { path: 'stars', element: <Stars /> },
      { path: 'history', element: <History /> },
      { path: 'app/:appName', element: <Upload /> },
      { path: 'ocr-result/:id', element: <OCRResult /> },
      { path: 'schema-generator', element: <SchemaGenerator mode="create" /> },
      { path: 'schema-generator/:appName', element: <SchemaGenerator mode="edit" /> },
      { path: 'apps/:appName/view', element: <SchemaGenerator mode="view" /> },
      { path: 'apps/:appName/edit', element: <SchemaGenerator mode="edit" /> },
      { path: 'admin', element: <Admin /> },
      { path: '*', element: <NotFound /> },
    ],
  },
]);

export default router;

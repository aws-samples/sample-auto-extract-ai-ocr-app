import { createBrowserRouter } from 'react-router-dom';
import { AuthWrapper } from './components/layout/AuthWrapper';
import { AppLayout } from './components/layout/AppLayout';
import { AppProvider } from './contexts/AppContext';
import Home from './pages/Home';
import Stars from './pages/Stars';
import History from './pages/History';
import Upload from './pages/upload/Upload';
import OCRResult from './pages/ocr-result/OCRResult';
import SchemaGenerator from './pages/schema/SchemaGenerator';
import Admin from './pages/Admin';
import NotFound from './pages/NotFound';

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

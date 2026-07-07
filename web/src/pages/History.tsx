import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, CheckCircle } from 'lucide-react';
import { Table, Thead, Tbody, usePagination, Pagination, SearchBox, CardTable, EmptyState, TableSkeleton, Tooltip } from '../components/ui';
import ProcessStatusBadge from '../components/shared/ProcessStatusBadge';
import PresenceBadge from '../components/shared/PresenceBadge';
import { usePresence, PRESENCE_LIST_MODE } from '../hooks/usePresence';
import { useAppContext } from '../contexts/AppContext';
import { formatDateTimeJST } from '../utils/dateUtils';
import api from '../services/api';

const STATUS_OPTIONS = [
  { value: '', label: 'ステータス: すべて' },
  { value: 'completed', label: 'OCR 済み' },
  { value: 'processing', label: '処理中' },
  { value: 'pending', label: '未処理' },
  { value: 'uploaded', label: '前処理中' },
  { value: 'failed', label: '失敗' },
];

interface ImageItem {
  id: string;
  status: string;
  appName?: string;
  name?: string;
  uploadTime?: string;
  verificationCompleted?: boolean;
  uploaded_by_email?: string;
  verified_by_email?: string;
  agentStatus?: string;
  agentSuggestionsCount?: number;
}

export default function History() {
  const navigate = useNavigate();
  const { apps } = useAppContext();
  const { byImageId: presenceByImageId } = usePresence({ imageId: PRESENCE_LIST_MODE });
  const [images, setImages] = useState<ImageItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [appFilter, setAppFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [verificationFilter, setVerificationFilter] = useState('');
  const [search, setSearch] = useState('');

  useEffect(() => {
    api.get('/images')
      .then(r => {
        const items: ImageItem[] = r.data.images || [];
        items.sort((a, b) => (b.uploadTime || '').localeCompare(a.uploadTime || ''));
        setImages(items);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    return images.filter(img => {
      if (appFilter && img.appName !== appFilter) return false;
      if (statusFilter && img.status !== statusFilter) return false;
      if (verificationFilter === 'verified' && !img.verificationCompleted) return false;
      if (verificationFilter === 'unverified' && img.verificationCompleted) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!img.name?.toLowerCase().includes(q) && !img.id.toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }, [images, appFilter, statusFilter, verificationFilter, search]);

  const appNames = useMemo(() => [...new Set(images.map(img => img.appName).filter(Boolean))], [images]);
  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } = usePagination(filtered);

  if (loading) return <div className="p-6 max-w-7xl mx-auto"><TableSkeleton rows={5} cols={6} /></div>;

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-default">処理履歴</h1>
        <p className="text-sm text-muted mt-1">処理済み・処理中の画像一覧</p>
      </div>

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <SearchBox value={search} onChange={v => { setSearch(v); setPage(0); }} placeholder="ファイル名で検索..." />
        <select value={appFilter} onChange={e => { setAppFilter(e.target.value); setPage(0); }} className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg">
          <option value="">ユースケース: すべて</option>
          {appNames.map(name => {
            const disp = apps.find(a => a.name === name)?.display_name;
            return <option key={name} value={name}>{disp ? `${disp}（${name}）` : name}</option>;
          })}
        </select>
        <select value={statusFilter} onChange={e => { setStatusFilter(e.target.value); setPage(0); }} className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg">
          {STATUS_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
        </select>
        <select value={verificationFilter} onChange={e => { setVerificationFilter(e.target.value); setPage(0); }} className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg">
          <option value="">確認状態: すべて</option>
          <option value="verified">確認済み</option>
          <option value="unverified">未確認</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon={FileText} message="処理履歴がありません" />
      ) : (
        <CardTable>
          <Table>
            <Thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ファイル名</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ユースケース</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">アップロード者</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">日時</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ステータス</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-muted uppercase tracking-wider">確認</th>
              </tr>
            </Thead>
            <Tbody>
              {paged.map(img => {
                const disp = apps.find(a => a.name === img.appName)?.display_name;
                return (
                  <tr key={img.id} className="cursor-pointer" onClick={() => navigate(`/ocr-result/${img.id}`)}>
                    <td className="px-4 py-3 text-sm">{img.name || img.id}</td>
                    <td className="px-4 py-3 text-sm">{disp ? <>{disp}<span className="text-neutral-400 ml-1">（{img.appName}）</span></> : img.appName || '-'}</td>
                    <td className="px-4 py-3 text-sm text-muted">{img.uploaded_by_email || '-'}</td>
                    <td className="px-4 py-3 text-sm">{formatDateTimeJST(img.uploadTime || '')}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <ProcessStatusBadge status={img.status} agentStatus={img.agentStatus} />
                        <PresenceBadge viewers={presenceByImageId[img.id] ?? []} />
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      {img.verificationCompleted ? (
                        <Tooltip content={img.verified_by_email || '確認済み'}>
                          <CheckCircle size={18} className="text-success inline-block" />
                        </Tooltip>
                      ) : (img.agentSuggestionsCount ?? 0) > 0 ? (
                        <span className="inline-flex items-center justify-center min-w-5 h-5 px-1.5 text-[10px] font-bold text-white bg-warning rounded-full">
                          {img.agentSuggestionsCount}
                        </span>
                      ) : (
                        <span className="text-neutral-300">-</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </Tbody>
          </Table>
        </CardTable>
      )}
      <Pagination page={page} total={total} setPage={setPage} pageSize={pageSize} totalItems={totalItems} onPageSizeChange={changePageSize} />
    </div>
  );
}

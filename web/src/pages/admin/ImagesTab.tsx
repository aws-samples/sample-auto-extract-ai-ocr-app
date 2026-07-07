import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Image, CheckCircle } from 'lucide-react';
import { Table, Thead, Tbody, usePagination, Pagination, SearchBox, CardTable, EmptyState, TableSkeleton, Tooltip } from '../../components/ui';
import StatusBadge from '../../components/shared/StatusBadge';
import PresenceBadge from '../../components/shared/PresenceBadge';
import { usePresence, PRESENCE_LIST_MODE } from '../../hooks/usePresence';
import { formatDateTimeJST } from '../../utils/dateUtils';
import * as adminApi from '../../services/adminApi';
import { useAppContext } from '../../contexts/AppContext';
import { AdminToolbar } from './AdminToolbar';
import type { ImageFile } from '../../types/ocr';

type AdminImage = ImageFile & { uploaded_by_email?: string; verified_by_email?: string };

const STATUS_OPTIONS = [
  { value: '', label: 'ステータス: すべて' },
  { value: 'completed', label: 'OCR 済み' },
  { value: 'processing', label: '処理中' },
  { value: 'pending', label: '未処理' },
  { value: 'uploaded', label: '前処理中' },
  { value: 'failed', label: '失敗' },
];

export default function ImagesTab() {
  const navigate = useNavigate();
  const { apps } = useAppContext();
  const { byImageId: presenceByImageId } = usePresence({ imageId: PRESENCE_LIST_MODE });
  const [images, setImages] = useState<AdminImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');
  const [appFilter, setAppFilter] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const [verificationFilter, setVerificationFilter] = useState('');

  useEffect(() => {
    adminApi.getAllImages().then((data) => {
      const list: AdminImage[] = (data.images || []).sort((a: AdminImage, b: AdminImage) =>
        (b.uploadTime || '').localeCompare(a.uploadTime || '')
      );
      setImages(list);
      setLoading(false);
    });
  }, []);

  const appNames = useMemo(() => [...new Set(images.map((i) => i.appName).filter(Boolean))], [images]);

  const filtered = useMemo(() => {
    return images.filter((img) => {
      if (appFilter && img.appName !== appFilter) return false;
      if (statusFilter && img.status !== statusFilter) return false;
      if (verificationFilter === 'verified' && !img.verificationCompleted) return false;
      if (verificationFilter === 'unverified' && img.verificationCompleted) return false;
      if (!search) return true;
      const q = search.toLowerCase();
      return img.name?.toLowerCase().includes(q) ||
        img.appName?.toLowerCase().includes(q) ||
        img.uploaded_by_email?.toLowerCase().includes(q) ||
        img.status?.toLowerCase().includes(q);
    });
  }, [images, search, appFilter, statusFilter, verificationFilter]);

  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } = usePagination(filtered);

  if (loading) return <TableSkeleton rows={5} cols={5} />;

  return (
    <>
      <AdminToolbar
        left={
          <>
            <SearchBox value={search} onChange={(v: string) => { setSearch(v); setPage(0); }} placeholder="ファイル名、アップロード者で検索..." />
            <select value={appFilter} onChange={(e) => { setAppFilter(e.target.value); setPage(0); }} className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg">
              <option value="">ユースケース: すべて</option>
              {appNames.map((name) => {
                const disp = apps.find((a) => a.name === name)?.display_name;
                return <option key={name} value={name}>{disp ? `${disp}（${name}）` : name}</option>;
              })}
            </select>
            <select value={statusFilter} onChange={(e) => { setStatusFilter(e.target.value); setPage(0); }} className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg">
              {STATUS_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
            <select value={verificationFilter} onChange={(e) => { setVerificationFilter(e.target.value); setPage(0); }} className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg">
              <option value="">確認状態: すべて</option>
              <option value="verified">確認済み</option>
              <option value="unverified">未確認</option>
            </select>
          </>
        }
      />
      {filtered.length === 0 ? (
        <EmptyState icon={Image} message="画像が見つかりません" />
      ) : (
        <CardTable>
          <Table>
            <Thead>
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ファイル名</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">アプリ</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">アップロード者</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">日時</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-muted uppercase tracking-wider">ステータス</th>
                <th className="px-4 py-3 text-center text-xs font-medium text-muted uppercase tracking-wider">確認</th>
              </tr>
            </Thead>
            <Tbody>
              {paged.map((img) => {
                const disp = apps.find((a) => a.name === img.appName)?.display_name;
                return (
                  <tr key={img.id} className="cursor-pointer" onClick={() => navigate(`/ocr-result/${img.id}`)}>
                    <td className="px-4 py-3 text-sm">{img.name || img.id}</td>
                    <td className="px-4 py-3 text-sm">{disp ? <>{disp}<span className="text-neutral-400 ml-1">（{img.appName}）</span></> : img.appName || '-'}</td>
                    <td className="px-4 py-3 text-sm text-muted">{img.uploaded_by_email || img.uploaded_by || '-'}</td>
                    <td className="px-4 py-3 text-sm">{formatDateTimeJST(img.uploadTime || '')}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        <StatusBadge status={img.status} />
                        <PresenceBadge viewers={presenceByImageId[img.id] ?? []} />
                      </div>
                    </td>
                    <td className="px-4 py-3 text-center">
                      {img.verificationCompleted ? (
                        <Tooltip content={img.verified_by_email || '確認済み'}>
                          <CheckCircle size={18} className="text-success inline-block" />
                        </Tooltip>
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
    </>
  );
}

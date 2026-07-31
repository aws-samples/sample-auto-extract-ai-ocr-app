import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePagination, Pagination, SearchBox, CardTable, TableSkeleton } from '../components/ui';
import ImageListTable from '../components/shared/ImageListTable';
import { ImageListFilterTabs } from '../components/shared/ImageListFilterTabs';
import { usePresence, PRESENCE_LIST_MODE } from '../hooks/usePresence';
import { useAppContext } from '../contexts/AppContext';
import {
  applyFilter,
  filterImageFamilies,
  getTopLevelFiles,
  type FilterKey,
} from '../utils/imageListHelpers';
import type { ImageFile } from '../types/ocr';
import api from '../services/api';

type HistoryImage = ImageFile;

export default function History() {
  const navigate = useNavigate();
  const { apps } = useAppContext();
  const { byImageId: presenceByImageId } = usePresence({ imageId: PRESENCE_LIST_MODE });

  const [images, setImages] = useState<HistoryImage[]>([]);
  const [loading, setLoading] = useState(true);
  const [appFilter, setAppFilter] = useState('');
  const [filterKey, setFilterKey] = useState<FilterKey>('all');
  const [search, setSearch] = useState('');

  useEffect(() => {
    api.get('/images')
      .then((r) => {
        const items: HistoryImage[] = r.data.images || [];
        items.sort((a, b) => (b.uploadTime || '').localeCompare(a.uploadTime || ''));
        setImages(items);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, []);

  const appNames = useMemo(
    () => [...new Set(images.map((img) => img.appName).filter(Boolean))],
    [images]
  );

  // 検索で子ページだけが一致しても、表示コンテナとなる親と兄弟ページを保持する。
  const preFiltered = useMemo(() => {
    return filterImageFamilies(images, (img) => {
      if (appFilter && img.appName !== appFilter) return false;
      if (search) {
        const q = search.toLowerCase();
        if (!img.name?.toLowerCase().includes(q) && !img.id.toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }, [images, appFilter, search]);

  // 確認状態フィルタも family 単位で適用し、親子を別ページに分断しない。
  const filtered = useMemo(() => applyFilter(preFiltered, filterKey), [preFiltered, filterKey]);
  const topLevelFiles = useMemo(() => getTopLevelFiles(filtered), [filtered]);
  const { page, setPage, total, paged, pageSize, changePageSize, totalItems } =
    usePagination(topLevelFiles);
  const visibleTopLevelIds = useMemo(() => new Set(paged.map((file) => file.id)), [paged]);

  const appDisplayNameMap = useMemo(() => {
    const map: Record<string, string> = {};
    apps.forEach((a) => {
      map[a.name] = a.display_name;
    });
    return map;
  }, [apps]);

  if (loading) {
    return (
      <div className="p-6 max-w-7xl mx-auto">
        <TableSkeleton rows={5} cols={6} />
      </div>
    );
  }

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-default">処理履歴</h1>
        <p className="text-sm text-muted mt-1">処理済み・処理中の画像一覧</p>
      </div>

      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <SearchBox
          value={search}
          onChange={(v) => {
            setSearch(v);
            setPage(0);
          }}
          placeholder="ファイル名で検索..."
        />
        <select
          value={appFilter}
          onChange={(e) => {
            setAppFilter(e.target.value);
            setPage(0);
          }}
          className="border border-default rounded-lg px-2.5 py-1.5 text-sm bg-bg"
        >
          <option value="">ユースケース: すべて</option>
          {appNames.map((name) => {
            const disp = apps.find((a) => a.name === name)?.display_name;
            return (
              <option key={name} value={name}>
                {disp ? `${disp}（${name}）` : name}
              </option>
            );
          })}
        </select>
      </div>

      <div className="mb-3">
        <ImageListFilterTabs
          files={preFiltered}
          value={filterKey}
          onChange={(k) => {
            setFilterKey(k);
            setPage(0);
          }}
        />
      </div>

      <CardTable>
        <ImageListTable
          files={filtered}
          visibleTopLevelIds={visibleTopLevelIds}
          onRowClick={(file) => navigate(`/ocr-result/${file.id}`)}
          showUsecaseColumn={true}
          presenceByImageId={presenceByImageId}
          appDisplayNameMap={appDisplayNameMap}
          emptyMessage="処理履歴がありません"
        />
      </CardTable>

      <Pagination
        page={page}
        total={total}
        setPage={setPage}
        pageSize={pageSize}
        totalItems={totalItems}
        onPageSizeChange={changePageSize}
      />
    </div>
  );
}

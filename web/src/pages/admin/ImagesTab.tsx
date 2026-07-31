import { useState, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { usePagination, Pagination, SearchBox, CardTable, TableSkeleton } from '../../components/ui';
import ImageListTable from '../../components/shared/ImageListTable';
import { ImageListFilterTabs } from '../../components/shared/ImageListFilterTabs';
import { useFetch } from '../../hooks/useFetch';
import { usePresence, PRESENCE_LIST_MODE } from '../../hooks/usePresence';
import { useAppContext } from '../../contexts/AppContext';
import {
  applyFilter,
  filterImageFamilies,
  getTopLevelFiles,
  type FilterKey,
} from '../../utils/imageListHelpers';
import * as adminApi from '../../services/adminApi';
import { AdminToolbar } from './AdminToolbar';
import type { ImageFile } from '../../types/ocr';

type AdminImage = ImageFile & { uploaded_by_email?: string; verified_by_email?: string };

export default function ImagesTab() {
  const navigate = useNavigate();
  const { apps } = useAppContext();
  const { byImageId: presenceByImageId } = usePresence({ imageId: PRESENCE_LIST_MODE });

  const [search, setSearch] = useState('');
  const [appFilter, setAppFilter] = useState('');
  const [filterKey, setFilterKey] = useState<FilterKey>('all');

  const fetchImages = useCallback(async (): Promise<AdminImage[]> => {
    const data = await adminApi.getAllImages();
    return (data.images || []).sort((a: AdminImage, b: AdminImage) =>
      (b.uploadTime || '').localeCompare(a.uploadTime || '')
    );
  }, []);
  const { data: images, loading } = useFetch<AdminImage[]>(fetchImages, []);

  const appNames = useMemo(
    () => [...new Set(images.map((i) => i.appName).filter(Boolean))],
    [images]
  );

  const preFiltered = useMemo(() => {
    return filterImageFamilies(images, (img) => {
      if (appFilter && img.appName !== appFilter) return false;
      if (!search) return true;
      const q = search.toLowerCase();
      return (
        img.name?.toLowerCase().includes(q) ||
        img.appName?.toLowerCase().includes(q) ||
        img.uploaded_by_email?.toLowerCase().includes(q) ||
        img.status?.toLowerCase().includes(q)
      );
    });
  }, [images, search, appFilter]);

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

  if (loading) return <TableSkeleton rows={5} cols={6} />;

  return (
    <>
      <AdminToolbar
        left={
          <>
            <SearchBox
              value={search}
              onChange={(v: string) => {
                setSearch(v);
                setPage(0);
              }}
              placeholder="ファイル名、アップロード者で検索..."
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
          </>
        }
      />

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
          emptyMessage="画像が見つかりません"
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
    </>
  );
}

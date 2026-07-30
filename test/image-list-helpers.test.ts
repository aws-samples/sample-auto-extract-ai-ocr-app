import type { ImageFile } from '../web/src/types/ocr';
import {
  applyFilter,
  filterImageFamilies,
  getTopLevelFiles,
  groupFiles,
  normalizeDeletionTargets,
  normalizeOcrTargets,
  toggleImageSelection,
} from '../web/src/utils/imageListHelpers';

const makeImage = (
  id: string,
  overrides: Partial<ImageFile> = {}
): ImageFile => ({
  id,
  name: `${id}.png`,
  s3_key: `images/${id}`,
  uploadTime: '2026-07-30T00:00:00Z',
  status: 'completed',
  ...overrides,
});

const makeFamily = () => {
  const parent = makeImage('parent', {
    name: 'document.pdf',
    pageProcessingMode: 'individual',
    totalPages: 3,
    uploadTime: '2026-07-30T03:00:00Z',
  });
  const children = [1, 2, 3].map((pageNumber) =>
    makeImage(`child-${pageNumber}`, {
      parentDocumentId: parent.id,
      pageNumber,
      totalPages: 3,
      uploadTime: `2026-07-30T0${pageNumber}:00:00Z`,
    })
  );
  return { parent, children };
};

describe('image list family helpers', () => {
  test('keeps the entire family when only one child matches', () => {
    const { parent, children } = makeFamily();
    const standalone = makeImage('standalone');
    const result = filterImageFamilies(
      [parent, ...children, standalone],
      (file) => file.id === 'child-2'
    );

    expect(new Set(result.map((file) => file.id))).toEqual(
      new Set(['parent', 'child-1', 'child-2', 'child-3'])
    );
  });

  test('applies status filters without orphaning matching children', () => {
    const { parent, children } = makeFamily();
    children[1] = { ...children[1], status: 'failed' };
    const result = applyFilter([parent, ...children, makeImage('standalone')], 'failed');

    expect(new Set(result.map((file) => file.id))).toEqual(
      new Set(['parent', 'child-1', 'child-2', 'child-3'])
    );
  });

  test('paginates only parent families and standalone files', () => {
    const { parent, children } = makeFamily();
    const standalone = makeImage('standalone', {
      uploadTime: '2026-07-30T04:00:00Z',
    });
    const files = [parent, ...children, standalone];

    expect(getTopLevelFiles(files).map((file) => file.id)).toEqual([
      'standalone',
      'parent',
    ]);
    expect(groupFiles(files).childPages[parent.id]).toHaveLength(3);
  });

  test('keeps an orphan child visible as a standalone row', () => {
    const orphan = makeImage('orphan', { parentDocumentId: 'missing-parent' });

    expect(getTopLevelFiles([orphan]).map((file) => file.id)).toEqual(['orphan']);
  });

  test('collapses a selected parent and its children into the parent target', () => {
    const { parent, children } = makeFamily();
    const targets = normalizeDeletionTargets(
      [parent, ...children],
      new Set([parent.id, children[0].id, children[1].id])
    );

    expect(targets).toEqual([parent.id]);
  });

  test('collapses all selected children into the parent target', () => {
    const { parent, children } = makeFamily();
    const targets = normalizeDeletionTargets(
      [parent, ...children],
      new Set(children.map((child) => child.id))
    );

    expect(targets).toEqual([parent.id]);
  });

  test('keeps a partial child selection as individual targets', () => {
    const { parent, children } = makeFamily();
    const standalone = makeImage('standalone');
    const targets = normalizeDeletionTargets(
      [parent, ...children, standalone],
      new Set([children[0].id, children[1].id, standalone.id])
    );

    expect(new Set(targets)).toEqual(
      new Set([children[0].id, children[1].id, standalone.id])
    );
  });

  test('expands a selected PDF parent into pending child OCR targets', () => {
    const { parent, children } = makeFamily();
    children[0] = { ...children[0], status: 'pending' };
    children[1] = { ...children[1], status: 'processing' };
    children[2] = { ...children[2], status: 'pending' };

    expect(normalizeOcrTargets([parent, ...children], new Set([parent.id]))).toEqual([
      children[0].id,
      children[2].id,
    ]);
  });

  test('processes only explicitly selected pending children', () => {
    const { parent, children } = makeFamily();
    children[0] = { ...children[0], status: 'pending' };
    children[1] = { ...children[1], status: 'pending' };

    expect(
      normalizeOcrTargets(
        [parent, ...children],
        new Set([children[1].id, children[2].id])
      )
    ).toEqual([children[1].id]);
  });

  test('includes only selected pending standalone OCR targets', () => {
    const pending = makeImage('pending', { status: 'pending' });
    const completed = makeImage('completed');
    const orphan = makeImage('orphan', {
      parentDocumentId: 'missing-parent',
      status: 'pending',
    });

    expect(
      normalizeOcrTargets(
        [pending, completed, orphan],
        new Set([pending.id, completed.id, orphan.id, 'stale'])
      )
    ).toEqual([pending.id, orphan.id]);
  });

  test('selects and clears an entire family from the parent checkbox', () => {
    const { parent, children } = makeFamily();
    const files = [parent, ...children];

    const selected = toggleImageSelection(files, new Set(), parent.id);
    expect(selected).toEqual(new Set([parent.id, ...children.map((child) => child.id)]));

    const cleared = toggleImageSelection(files, selected, parent.id);
    expect(cleared).toEqual(new Set());
  });

  test('clears the parent selection when one selected child is cleared', () => {
    const { parent, children } = makeFamily();
    const files = [parent, ...children];
    const allSelected = new Set([parent.id, ...children.map((child) => child.id)]);

    const result = toggleImageSelection(files, allSelected, children[1].id);

    expect(result).toEqual(new Set([children[0].id, children[2].id]));
  });

  test('selects the parent when the final child is selected', () => {
    const { parent, children } = makeFamily();
    const files = [parent, ...children];
    const partiallySelected = new Set([children[0].id, children[1].id]);

    const result = toggleImageSelection(files, partiallySelected, children[2].id);

    expect(result).toEqual(new Set([parent.id, ...children.map((child) => child.id)]));
  });

  test('toggles a standalone selection without affecting other files', () => {
    const first = makeImage('first');
    const second = makeImage('second');

    expect(toggleImageSelection([first, second], new Set([second.id]), first.id)).toEqual(
      new Set([first.id, second.id])
    );
  });
});

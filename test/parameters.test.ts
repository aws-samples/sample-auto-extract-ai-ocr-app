/**
 * resolveOcrModel / getParameters の OCR 用モデル解決のテスト。
 *
 * 想定している正しい挙動:
 * - ocrEngine が "bedrock" のとき、ocrModelId 未指定なら us.openai.gpt-6-astra、
 *   ocrModelRegion 未指定なら modelRegion を使う。抽出用 modelId は OCR に流れない。
 * - 許可リスト外のモデル、またはそのモデルを呼べないリージョンなら synth 時に例外にする。
 * - ocrEngine が "bedrock" 以外、または OCR 無効なら ocrModelId / ocrModelRegion を外し、
 *   許可リスト外の値が残っていても例外にしない。
 * - 定義済みの全環境（base / dev / stg / prod）が検証を通る。
 */
import { AppParameters, getParameters, resolveOcrModel } from '../lib/parameters';

const bedrockParams = (overrides: Partial<AppParameters> = {}): AppParameters => ({
  ...getParameters('dev'),
  enableOcr: true,
  ocrEngine: 'bedrock',
  modelId: 'us.anthropic.claude-sonnet-4-6',
  modelRegion: 'us-east-1',
  ocrModelId: undefined,
  ocrModelRegion: undefined,
  ...overrides,
});

describe('resolveOcrModel', () => {
  test('defaults to Astra and modelRegion without using the extraction modelId', () => {
    const p = resolveOcrModel(bedrockParams());

    expect(p.ocrModelId).toBe('us.openai.gpt-6-astra');
    expect(p.ocrModelRegion).toBe('us-east-1');
    expect(p.modelId).toBe('us.anthropic.claude-sonnet-4-6');
  });

  test('accepts the global Astra profile from Tokyo', () => {
    const p = resolveOcrModel(
      bedrockParams({ ocrModelId: 'global.openai.gpt-6-astra', ocrModelRegion: 'ap-northeast-1' }),
    );

    expect(p.ocrModelId).toBe('global.openai.gpt-6-astra');
    expect(p.ocrModelRegion).toBe('ap-northeast-1');
  });

  test.each([
    'us.anthropic.claude-sonnet-4-6',
    'openai.gpt-6-astra',
    'us.openai.gpt-6-sol',
  ])('rejects %s as the OCR model', (ocrModelId) => {
    expect(() => resolveOcrModel(bedrockParams({ ocrModelId }))).toThrow(/Bedrock OCR に使えません/);
  });

  test('rejects the US profile from a region outside the US geography', () => {
    expect(() =>
      resolveOcrModel(bedrockParams({ ocrModelId: 'us.openai.gpt-6-astra', ocrModelRegion: 'ap-northeast-1' })),
    ).toThrow(/呼び出せません/);
  });

  test('drops OCR model settings for non-bedrock engines', () => {
    const p = resolveOcrModel(bedrockParams({ ocrEngine: 'paddle', ocrModelId: 'not-allowed' }));

    expect(p.ocrModelId).toBeUndefined();
    expect(p.ocrModelRegion).toBeUndefined();
  });

  test('drops OCR model settings when OCR is disabled', () => {
    const p = resolveOcrModel(bedrockParams({ enableOcr: false, ocrModelId: 'not-allowed' }));

    expect(p.ocrModelId).toBeUndefined();
  });
});

describe('getParameters', () => {
  test.each(['base', 'dev', 'stg', 'prod'])('%s passes OCR model validation', (env) => {
    expect(() => getParameters(env)).not.toThrow();
  });
});

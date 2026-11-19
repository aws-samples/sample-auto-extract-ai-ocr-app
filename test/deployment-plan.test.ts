/**
 * resolveDeploymentPlan のテスト。
 *
 * 想定している正しい挙動:
 * - WAF 有効時は us-east-1 の WAF スタックをアプリケーションスタックより先に並べる。
 * - ocrEngine が "textract" で OCR 有効のとき、Textract の提供リージョン以外への
 *   デプロイは例外にする。他のエンジンや OCR 無効時はリージョンを検証しない。
 */
import { resolveDeploymentPlan } from '../lib/deployment-plan';
import { getParameters } from '../lib/parameters';

describe('deployment plan', () => {
  test('includes the WAF stack in us-east-1 before the application stack', () => {
    const params = {
      ...getParameters('dev'),
      waf: { enabled: true },
    };

    const plan = resolveDeploymentPlan(
      params,
      'OcrAppStack-dev',
      'ap-northeast-1',
    );

    expect(plan.wafStack).toEqual({
      kind: 'waf',
      name: 'OcrAppStack-dev-Waf',
      region: 'us-east-1',
    });
    expect(plan.applicationStack).toEqual({
      kind: 'application',
      name: 'OcrAppStack-dev',
      region: 'ap-northeast-1',
    });
    expect(plan.stacks).toEqual([
      plan.wafStack,
      plan.applicationStack,
    ]);
  });

  test('includes only the application stack when WAF is disabled', () => {
    const params = {
      ...getParameters('dev'),
      waf: { enabled: false },
    };

    const plan = resolveDeploymentPlan(params, 'OcrAppStack-dev', 'eu-west-1');

    expect(plan.wafStack).toBeUndefined();
    expect(plan.stacks).toEqual([
      {
        kind: 'application',
        name: 'OcrAppStack-dev',
        region: 'eu-west-1',
      },
    ]);
  });
});

describe('textract region validation', () => {
  const textractParams = () => ({
    ...getParameters('dev'),
    enableOcr: true,
    ocrEngine: 'textract' as const,
  });

  test('accepts Tokyo', () => {
    expect(() =>
      resolveDeploymentPlan(textractParams(), 'OcrAppStack-dev', 'ap-northeast-1'),
    ).not.toThrow();
  });

  test('rejects a region where Textract is not offered', () => {
    expect(() =>
      resolveDeploymentPlan(textractParams(), 'OcrAppStack-dev', 'ap-northeast-3'),
    ).toThrow(/"textract" はリージョン "ap-northeast-3" で使えません/);
  });

  test('does not check the region for other engines', () => {
    const params = { ...textractParams(), ocrEngine: 'paddle' as const };
    expect(() =>
      resolveDeploymentPlan(params, 'OcrAppStack-dev', 'ap-northeast-3'),
    ).not.toThrow();
  });

  test('does not check the region when OCR is disabled', () => {
    const params = { ...textractParams(), enableOcr: false };
    expect(() =>
      resolveDeploymentPlan(params, 'OcrAppStack-dev', 'ap-northeast-3'),
    ).not.toThrow();
  });
});

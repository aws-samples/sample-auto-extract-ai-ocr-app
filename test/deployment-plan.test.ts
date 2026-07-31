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

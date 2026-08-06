import {
  AppParameters,
  getParameters,
  getStackName,
} from './parameters';

export const DEFAULT_APPLICATION_REGION = 'ap-northeast-1';

export type DeploymentStackKind = 'waf' | 'application';

export interface DeploymentStackPlan {
  kind: DeploymentStackKind;
  name: string;
  region: string;
}

export interface DeploymentPlan {
  params: AppParameters;
  applicationStack: DeploymentStackPlan;
  wafStack?: DeploymentStackPlan;
  stacks: DeploymentStackPlan[];
}

export function resolveDeploymentPlan(
  params: AppParameters,
  stackName: string,
  applicationRegion: string,
): DeploymentPlan {
  const applicationStack: DeploymentStackPlan = {
    kind: 'application',
    name: stackName,
    region: applicationRegion,
  };

  const wafStack: DeploymentStackPlan | undefined = params.waf.enabled
    ? {
        kind: 'waf',
        name: `${stackName}-Waf`,
        region: 'us-east-1',
      }
    : undefined;

  return {
    params,
    applicationStack,
    wafStack,
    stacks: wafStack ? [wafStack, applicationStack] : [applicationStack],
  };
}

export function resolveDeploymentPlanFromEnvironment(
  env?: string,
  applicationRegion = DEFAULT_APPLICATION_REGION,
): DeploymentPlan {
  const params = getParameters(env);
  return resolveDeploymentPlan(params, getStackName(env), applicationRegion);
}

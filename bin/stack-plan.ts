#!/usr/bin/env node
import { resolveDeploymentPlanFromEnvironment } from '../lib/deployment-plan';

const plan = resolveDeploymentPlanFromEnvironment(
  process.env.ENV,
  process.env.CDK_DEFAULT_REGION,
);

for (const stack of plan.stacks) {
  process.stdout.write(`${stack.kind}\t${stack.name}\t${stack.region}\n`);
}

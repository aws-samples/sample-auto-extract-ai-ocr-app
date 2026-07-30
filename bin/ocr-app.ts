#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { resolveDeploymentPlanFromEnvironment } from "../lib/deployment-plan";
import { OcrAppStack } from "../lib/ocr-app-stack";
import { WafStack } from "../lib/waf-stack";

const env = process.env.ENV;
const deploymentPlan = resolveDeploymentPlanFromEnvironment(
  env,
  process.env.CDK_DEFAULT_REGION,
);
const params = deploymentPlan.params;

const app = new cdk.App();

// WAF Stack（CloudFront 用 WAF は us-east-1 に作成する必要がある）
let webAclArn: string | undefined;
if (deploymentPlan.wafStack) {
  const wafStack = new WafStack(app, deploymentPlan.wafStack.name, {
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: deploymentPlan.wafStack.region,
    },
    crossRegionReferences: true,
    wafOptions: params.waf,
    envName: env,
  });
  webAclArn = wafStack.webAclArn;
}

new OcrAppStack(app, deploymentPlan.applicationStack.name, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: deploymentPlan.applicationStack.region,
  },
  crossRegionReferences: true,
  params,
  webAclArn,
  envName: env,
});

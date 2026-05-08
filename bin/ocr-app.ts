#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { OcrAppStack } from "../lib/ocr-app-stack";
import { WafStack } from "../lib/waf-stack";
import { getParameters, getStackName } from "../lib/parameters";

const env = process.env.ENV;
const params = getParameters(env);
const stackName = getStackName(env);

const app = new cdk.App();

// WAF Stack（CloudFront 用 WAF は us-east-1 に作成する必要がある）
let webAclArn: string | undefined;
if (params.waf.enabled) {
  const wafStack = new WafStack(app, `${stackName}-Waf`, {
    env: {
      account: process.env.CDK_DEFAULT_ACCOUNT,
      region: "us-east-1",
    },
    crossRegionReferences: true,
    wafOptions: params.waf,
    envName: env,
  });
  webAclArn = wafStack.webAclArn;
}

new OcrAppStack(app, stackName, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "ap-northeast-1",
  },
  crossRegionReferences: true,
  params,
  webAclArn,
});

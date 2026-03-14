#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { OcrAppStack } from "../lib/ocr-app-stack";
import { getParameters, getStackName } from "../lib/parameters";

const env = process.env.ENV;
const params = getParameters(env);
const stackName = getStackName(env);

const app = new cdk.App();
new OcrAppStack(app, stackName, {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "ap-northeast-1",
  },
  params,
});

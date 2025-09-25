#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { OcrAppStack } from "../lib/ocr-app-stack";

const app = new cdk.App();
new OcrAppStack(app, "OcrAppStack", {
  env: {
    account: process.env.CDK_DEFAULT_ACCOUNT,
    region: process.env.CDK_DEFAULT_REGION || "ap-northeast-1",
  },
});

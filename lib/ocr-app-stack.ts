import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";

import { Auth } from "./constructs/auth";
import { Api } from "./constructs/api";
import { Web } from "./constructs/web";
import { Database } from "./constructs/database";
import { Ocr } from "./constructs/ocr";

export class OcrAppStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // OCR有効フラグを取得（デフォルトはtrue）
    const enableOcr = this.node.tryGetContext("enable_ocr") ?? true;

    const auth = new Auth(this, "Auth");

    const database = new Database(this, "Database", {});

    // OCRが有効な場合のみSageMakerエンドポイントを作成
    let ocrEndpoint = undefined;
    if (enableOcr) {
      const ocr = new Ocr(this, "OcrEndpoint");
      ocrEndpoint = ocr;
    }

    const api = new Api(this, "Api", {
      imagesTable: database.imagesTable,
      jobsTable: database.jobsTable,
      schemasTable: database.schemasTable,
      userPoolId: auth.userPool.userPoolId,
      userPoolClientId: auth.client.userPoolClientId,
      enableOcr: enableOcr,
      sagemakerEndpointName: ocrEndpoint?.endpointName,
      sagemakerInferenceComponentName: ocrEndpoint?.inferenceComponentName,
    });

    new Web(this, "WebConstruct", {
      buildFolder: "/dist",
      userPoolId: auth.userPool.userPoolId,
      userPoolClientId: auth.client.userPoolClientId,
      apiUrl: api.apiEndpoint,
      enableOcr: enableOcr,
    });
  }
}

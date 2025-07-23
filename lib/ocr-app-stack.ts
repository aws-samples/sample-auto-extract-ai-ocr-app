import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";
import { Network } from "./constructs/network";
import { Auth } from "./constructs/auth";
import { Api } from "./constructs/api";
import { Web } from "./constructs/web";
import { Database } from "./constructs/database";
import { Ocr } from "./constructs/ocr";
import * as path from "path";

export class OcrAppStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // OCR有効フラグを取得（デフォルトはtrue）
    const enableOcr = this.node.tryGetContext("enable_ocr") ?? true;

    // const network = new Network(this, "Network");

    const auth = new Auth(this, "Auth");

    // DynamoDB に変更（VPC 不要）
    const database = new Database(this, "Database", {});

    // OCRが有効な場合のみSageMakerエンドポイントを作成
    let ocrEndpoint = undefined;
    if (enableOcr) {
      const ocr = new Ocr(this, "OcrEndpoint", {
        baseName: "yomitoku",
        containerPath: path.join(__dirname, "../container"),
        instanceType: "ml.g4dn.2xlarge",
        environment: {
          USE_GPU: "true",
          CUDA_VISIBLE_DEVICES: "0",
          PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512",
        },
      });
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

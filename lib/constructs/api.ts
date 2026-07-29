import { Duration, RemovalPolicy, CfnOutput } from "aws-cdk-lib";
import { Construct } from "constructs";
import {
  BlockPublicAccess,
  Bucket,
  BucketEncryption,
  HttpMethods,
} from "aws-cdk-lib/aws-s3";
import {
  PolicyStatement,
  Role,
  ServicePrincipal,
  ManagedPolicy,
} from "aws-cdk-lib/aws-iam";
import { DockerImageCode, DockerImageFunction } from "aws-cdk-lib/aws-lambda";
import * as apigateway from "aws-cdk-lib/aws-apigateway";
import {
  RestApi,
  LambdaIntegration,
  Cors,
  AuthorizationType,
  CognitoUserPoolsAuthorizer,
  PassthroughBehavior,
} from "aws-cdk-lib/aws-apigateway";
import { UserPool } from "aws-cdk-lib/aws-cognito";
import { Platform } from "aws-cdk-lib/aws-ecr-assets";
import { Table } from "aws-cdk-lib/aws-dynamodb";

export interface ApiProps {
  imagesTable: Table;
  jobsTable: Table;
  schemasTable: Table;
  userPreferencesTable: Table;
  // toolsTable removed — tools now managed via AgentCore Gateway + DSQL
  userPoolId: string;
  userPoolClientId: string;
  enableOcr: boolean;
  ocrEngine?: string;
  sagemakerEndpointName?: string;
  sagemakerInferenceComponentName?: string;
  agentRuntimeArn?: string;
  modelId: string;
  modelRegion: string;
  dsqlEndpoint: string;
  dsqlRegion: string;
  dsqlClusterArn: string;
}

export class Api extends Construct {
  public readonly apiEndpoint: string;
  public readonly documentBucket: Bucket;
  public readonly syncBucket: Bucket;
  public readonly handler: DockerImageFunction;

  constructor(scope: Construct, id: string, props: ApiProps) {
    super(scope, id);

    const { imagesTable, jobsTable } = props;

    const { modelId, modelRegion } = props;

    // S3バケット（ドキュメント保存用）
    const documentBucket = new Bucket(this, "DocumentBucket", {
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      encryption: BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
      cors: [
        {
          allowedHeaders: ["*"],
          allowedMethods: [
            HttpMethods.GET,
            HttpMethods.POST,
            HttpMethods.PUT,
            HttpMethods.DELETE,
            HttpMethods.HEAD,
          ],
          allowedOrigins: ["*"],
          exposedHeaders: ["ETag", "x-amz-request-id", "x-amz-id-2"],
          maxAge: 3600,
        },
      ],
    });

    // S3バケット（同期用）
    const syncBucket = new Bucket(this, "SyncBucket", {
      blockPublicAccess: BlockPublicAccess.BLOCK_ALL,
      encryption: BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      removalPolicy: RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // Lambda実行ロール
    const lambdaRole = new Role(this, "LambdaExecutionRole", {
      assumedBy: new ServicePrincipal("lambda.amazonaws.com"),
      managedPolicies: [
        ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AWSLambdaBasicExecutionRole"
        ),
      ],
    });

    // S3へのアクセス権限（オブジェクト読み書きのみ。バケット削除やポリシー変更は付与しない）
    documentBucket.grantReadWrite(lambdaRole);
    syncBucket.grantReadWrite(lambdaRole);

    // SageMakerへのアクセス権限（OCRが有効な場合のみ）
    if (props.enableOcr && props.sagemakerEndpointName) {
      lambdaRole.addToPolicy(
        new PolicyStatement({
          actions: [
            "sagemaker:InvokeEndpoint",
            "sagemaker:DescribeInferenceComponent",
            "sagemaker:DescribeEndpoint",
          ],
          resources: ["*"],
        })
      );
    }

    // Bedrockへのアクセス権限
    lambdaRole.addToPolicy(
      new PolicyStatement({
        actions: [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ],
        resources: ["*"],
      })
    );

    // DynamoDBへのアクセス権限
    lambdaRole.addToPolicy(
      new PolicyStatement({
        actions: [
          "dynamodb:PutItem",
          "dynamodb:GetItem",
          "dynamodb:UpdateItem",
          "dynamodb:DeleteItem",
          "dynamodb:Query",
          "dynamodb:Scan",
        ],
        resources: [
          imagesTable.tableArn,
          jobsTable.tableArn,
          props.schemasTable.tableArn,
          props.userPreferencesTable.tableArn,
          `${imagesTable.tableArn}/index/*`, // GSIへのアクセス権限も追加
          `${jobsTable.tableArn}/index/*`, // JobsTable GSI (ImageIdIndex)
        ],
      })
    );

    // DSQL 接続権限
    lambdaRole.addToPolicy(
      new PolicyStatement({
        actions: ["dsql:DbConnectAdmin"],
        resources: [props.dsqlClusterArn],
      })
    );

    // Lambda関数の作成
    const lambdaFunction = new DockerImageFunction(this, "ApiFunction", {
      code: DockerImageCode.fromImageAsset("lambda/api", {
        platform: Platform.LINUX_AMD64,
      }),
      timeout: Duration.minutes(15),
      memorySize: 4096,
      environment: {
        BUCKET_NAME: documentBucket.bucketName,
        SYNC_BUCKET_NAME: syncBucket.bucketName,
        IMAGES_TABLE_NAME: imagesTable.tableName,
        JOBS_TABLE_NAME: jobsTable.tableName,
        SCHEMAS_TABLE_NAME: props.schemasTable.tableName,
        ENABLE_OCR: props.enableOcr.toString(),
        SAGEMAKER_ENDPOINT_NAME: props.sagemakerEndpointName || "",
        SAGEMAKER_INFERENCE_COMPONENT_NAME:
          props.sagemakerInferenceComponentName || "",
        OCR_ENGINE: props.ocrEngine || "paddle",
        MODEL_ID: modelId,
        MODEL_REGION: modelRegion,
        AGENT_RUNTIME_ARN: props.agentRuntimeArn || "",
        DSQL_ENDPOINT: props.dsqlEndpoint,
        DSQL_REGION: props.dsqlRegion,
        USER_PREFERENCES_TABLE_NAME: props.userPreferencesTable.tableName,
        PORT: "8080",
        // Lambda Web Adapter関連の環境変数
        AWS_LWA_PORT: "8080",
        AWS_LWA_READINESS_CHECK_PATH: "/health",
      },
      role: lambdaRole,
    });

    // プロパティに保存
    this.handler = lambdaFunction;
    this.documentBucket = documentBucket;

    // SchemaGenerate Worker Lambda
    // スキーマ自動生成の Bedrock 呼び出しが 40-50 秒かかり、API Gateway の 29 秒制限を
    // 超えるため非同期化した Worker Lambda。API Lambda が async invoke で起動する。
    const schemaGenerate = new DockerImageFunction(this, "SchemaGenerate", {
      code: DockerImageCode.fromImageAsset("lambda/api", {
        file: "Dockerfile.worker",
        cmd: ["app.workers.schema_generate.schema_generate_handler"],
        platform: Platform.LINUX_AMD64,
      }),
      timeout: Duration.minutes(5),
      memorySize: 2048,
      environment: {
        BUCKET_NAME: documentBucket.bucketName,
        JOBS_TABLE_NAME: jobsTable.tableName,
        MODEL_ID: modelId,
        MODEL_REGION: modelRegion,
      },
    });

    // Worker が JobsTable を更新
    jobsTable.grantReadWriteData(schemaGenerate);

    // S3 からサンプルファイルを取得
    documentBucket.grantRead(schemaGenerate);

    // Bedrock 呼び出し
    schemaGenerate.addToRolePolicy(
      new PolicyStatement({
        actions: [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
        ],
        resources: ["*"],
      })
    );

    // API Lambda から SchemaGenerate を async invoke するための権限
    schemaGenerate.grantInvoke(lambdaFunction);

    // API Lambda の環境変数に function name を注入
    lambdaFunction.addEnvironment(
      "SCHEMA_GENERATE_FUNCTION_NAME",
      schemaGenerate.functionName
    );

    // PdfConvert Worker Lambda
    // PDF→画像変換は数十秒かかりうえに HTTP 応答後の実行環境回収で取りこぼす恐れがあるため、
    // API Lambda 内スレッドではなく独立 Worker に async invoke で委譲する。
    const pdfConvert = new DockerImageFunction(this, "PdfConvert", {
      code: DockerImageCode.fromImageAsset("lambda/api", {
        file: "Dockerfile.worker",
        cmd: ["app.workers.pdf_convert.pdf_convert_handler"],
        platform: Platform.LINUX_AMD64,
      }),
      timeout: Duration.minutes(5),
      memorySize: 4096,
      environment: {
        BUCKET_NAME: documentBucket.bucketName,
        IMAGES_TABLE_NAME: imagesTable.tableName,
        SCHEMAS_TABLE_NAME: props.schemasTable.tableName,
      },
    });

    imagesTable.grantReadWriteData(pdfConvert);
    props.schemasTable.grantReadData(pdfConvert);
    documentBucket.grantReadWrite(pdfConvert);

    // API Lambda から PdfConvert を async invoke するための権限と function name 注入
    pdfConvert.grantInvoke(lambdaFunction);
    lambdaFunction.addEnvironment(
      "PDF_CONVERT_FUNCTION_NAME",
      pdfConvert.functionName
    );

    // S3SyncImport Worker Lambda
    // S3 同期インポートの重い処理をブラウザのループから独立 Worker に移し、
    // 画面を閉じても取りこぼさないようにする。
    const s3SyncImport = new DockerImageFunction(this, "S3SyncImport", {
      code: DockerImageCode.fromImageAsset("lambda/api", {
        file: "Dockerfile.worker",
        cmd: ["app.workers.s3_sync_import.s3_sync_import_handler"],
        platform: Platform.LINUX_AMD64,
      }),
      timeout: Duration.minutes(15),
      memorySize: 4096,
      environment: {
        BUCKET_NAME: documentBucket.bucketName,
        SYNC_BUCKET_NAME: syncBucket.bucketName,
        IMAGES_TABLE_NAME: imagesTable.tableName,
        SCHEMAS_TABLE_NAME: props.schemasTable.tableName,
        PDF_CONVERT_FUNCTION_NAME: pdfConvert.functionName,
      },
    });

    imagesTable.grantReadWriteData(s3SyncImport);
    props.schemasTable.grantReadData(s3SyncImport);
    documentBucket.grantReadWrite(s3SyncImport);
    syncBucket.grantRead(s3SyncImport);
    // Worker が PDF 変換を再委譲するため PdfConvert を invoke できる
    pdfConvert.grantInvoke(s3SyncImport);

    // API Lambda から S3SyncImport を async invoke するための権限と function name 注入
    s3SyncImport.grantInvoke(lambdaFunction);
    lambdaFunction.addEnvironment(
      "S3_SYNC_IMPORT_FUNCTION_NAME",
      s3SyncImport.functionName
    );

    // AgentRuntime呼び出し権限
    if (props.agentRuntimeArn) {
      lambdaFunction.addToRolePolicy(
        new PolicyStatement({
          actions: ["bedrock-agentcore:InvokeAgentRuntime"],
          resources: [props.agentRuntimeArn, props.agentRuntimeArn + "/*"],
        })
      );
    }


    // Cognitoユーザープール参照
    const userPool = UserPool.fromUserPoolId(
      this,
      "ImportedUserPool",
      props.userPoolId
    );

    // Cognitoオーソライザー
    const authorizer = new CognitoUserPoolsAuthorizer(this, "ApiAuthorizer", {
      cognitoUserPools: [userPool],
    });

    // API Gatewayの作成
    const api = new RestApi(this, "OcrApi", {
      defaultCorsPreflightOptions: {
        allowOrigins: Cors.ALL_ORIGINS,
        allowMethods: Cors.ALL_METHODS,
      },
      deployOptions: {
        stageName: "prod",
      },
    });

    // ルートリソースに対応するプロキシ統合
    const proxyResource = api.root.addResource("{proxy+}");

    proxyResource.addMethod(
      "ANY",
      new LambdaIntegration(lambdaFunction, {
        proxy: true,
        // Lambda プロキシ統合のレスポンス設定
        passthroughBehavior: PassthroughBehavior.WHEN_NO_MATCH,
        integrationResponses: [
          {
            statusCode: "200",
            responseParameters: {
              "method.response.header.Access-Control-Allow-Origin": "'*'",
              "method.response.header.Access-Control-Allow-Headers":
                "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With'",
              "method.response.header.Access-Control-Allow-Methods":
                "'GET,POST,PUT,DELETE,OPTIONS'",
            },
          },
          {
            // エラーレスポンスの処理
            selectionPattern: ".*",
            statusCode: "400",
            responseParameters: {
              "method.response.header.Access-Control-Allow-Origin": "'*'",
            },
          },
        ],
      }),
      {
        methodResponses: [
          {
            statusCode: "200",
            responseParameters: {
              "method.response.header.Access-Control-Allow-Origin": true,
              "method.response.header.Access-Control-Allow-Headers": true,
              "method.response.header.Access-Control-Allow-Methods": true,
            },
          },
          {
            statusCode: "400",
            responseParameters: {
              "method.response.header.Access-Control-Allow-Origin": true,
            },
          },
        ],
        // 認証設定
        authorizer,
        authorizationType: AuthorizationType.COGNITO,
      }
    );

    // Gateway Responses に CORS ヘッダー追加（Authorizer エラー等でも CORS が返るように）
    for (const type of [
      apigateway.ResponseType.DEFAULT_4XX,
      apigateway.ResponseType.DEFAULT_5XX,
    ]) {
      api.addGatewayResponse(`GatewayResponse${type.responseType}`, {
        type,
        responseHeaders: {
          "Access-Control-Allow-Origin": "'*'",
          "Access-Control-Allow-Headers": "'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Requested-With'",
        },
      });
    }

    // エンドポイントのCFn出力
    this.apiEndpoint = api.url;
    new CfnOutput(this, "ApiEndpoint", {
      value: api.url,
      description: "API Gateway endpoint URL",
    });

    // DynamoDB テーブル名の出力
    new CfnOutput(this, "ImagesTableName", {
      value: imagesTable.tableName,
      description: "DynamoDB Images Table Name",
    });

    new CfnOutput(this, "JobsTableName", {
      value: jobsTable.tableName,
      description: "DynamoDB Jobs Table Name",
    });

    // S3 バケット名の出力
    new CfnOutput(this, "DocumentBucketName", {
      value: documentBucket.bucketName,
      description: "S3 Document Bucket Name",
    });

    new CfnOutput(this, "SyncBucketName", {
      value: syncBucket.bucketName,
      description: "S3 Sync Bucket Name",
    });

    // プロパティに割り当て
    this.syncBucket = syncBucket;
  }
}

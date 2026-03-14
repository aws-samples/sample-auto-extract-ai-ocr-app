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
  toolsTable?: Table;
  userPoolId: string;
  userPoolClientId: string;
  enableOcr: boolean;
  sagemakerEndpointName?: string;
  sagemakerInferenceComponentName?: string;
  agentRuntimeArn?: string;
  modelId: string;
  modelRegion: string;
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

    // S3へのアクセス権限
    lambdaRole.addToPolicy(
      new PolicyStatement({
        actions: ["s3:*"],
        resources: [
          documentBucket.bucketArn, 
          `${documentBucket.bucketArn}/*`,
          syncBucket.bucketArn,
          `${syncBucket.bucketArn}/*`
        ],
      })
    );

    // SageMakerへのアクセス権限（OCRが有効な場合のみ）
    if (props.enableOcr && props.sagemakerEndpointName) {
      lambdaRole.addToPolicy(
        new PolicyStatement({
          actions: [
            "sagemaker:InvokeEndpoint",
            "sagemaker:DescribeInferenceComponent",
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
          ...(props.toolsTable ? [props.toolsTable.tableArn] : []),
          `${imagesTable.tableArn}/index/*`, // GSIへのアクセス権限も追加
        ],
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
        TOOLS_TABLE_NAME: props.toolsTable?.tableName || "",
        ENABLE_OCR: props.enableOcr.toString(),
        SAGEMAKER_ENDPOINT_NAME: props.sagemakerEndpointName || "",
        SAGEMAKER_INFERENCE_COMPONENT_NAME:
          props.sagemakerInferenceComponentName || "",
        MODEL_ID: modelId,
        MODEL_REGION: modelRegion,
        AGENT_RUNTIME_ARN: props.agentRuntimeArn || "",
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

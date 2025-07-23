import * as cdk from "aws-cdk-lib";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import {
  ManagedPolicy,
  PolicyStatement,
  Role,
  ServicePrincipal,
} from "aws-cdk-lib/aws-iam";
import {
  CfnModel,
  CfnEndpointConfig,
  CfnEndpoint,
  CfnInferenceComponent,
} from "aws-cdk-lib/aws-sagemaker";
import { Construct } from "constructs";
import * as path from "path";

/**
 * OCR機能をSageMakerで実装するためのプロパティ
 */
export interface OcrProps {
  /**
   * SageMakerリソースのベース名
   */
  baseName?: string;

  /**
   * Dockerfileを含むコンテナディレクトリへのパス
   */
  containerPath?: string;

  /**
   * SageMakerエンドポイントのインスタンスタイプ
   * @default ml.g4dn.2xlarge
   */
  instanceType?: string;

  /**
   * コンテナの環境変数
   */
  environment?: Record<string, string>;
}

/**
 * OCR機能をSageMakerで実装するためのコンストラクト
 */
export class Ocr extends Construct {
  /**
   * SageMakerエンドポイント名
   */
  public readonly endpointName: string;

  /**
   * SageMaker推論コンポーネント名
   */
  public readonly inferenceComponentName: string;

  /**
   * SageMaker実行ロールARN
   */
  public readonly sagemakerRoleArn: string;

  constructor(scope: Construct, id: string, props: OcrProps = {}) {
    super(scope, id);

    // デフォルト値の設定
    const baseName = props.baseName || "yomitoku";
    const instanceType = props.instanceType || "ml.g4dn.2xlarge";
    const containerPath =
      props.containerPath || path.join(__dirname, "../../container");
    const modelName = `${baseName}-model`;
    const endpointConfigName = `${baseName}-endpoint-config`;
    this.endpointName = `${baseName}-endpoint`;
    const variantName = "AllTraffic";
    this.inferenceComponentName = `${baseName}-inference-component`;

    // デフォルトの環境変数（Dockerfileの設定に合わせる）
    const defaultEnv = {
      USE_GPU: "true",
      CUDA_VISIBLE_DEVICES: "0",
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512",
    };

    // デフォルトと指定された環境変数をマージ
    const environment = {
      ...defaultEnv,
      ...(props.environment || {}),
    };

    // SageMaker用のIAMロール
    const sagemakerRole = new Role(this, "SageMakerExecutionRole", {
      assumedBy: new ServicePrincipal("sagemaker.amazonaws.com"),
      managedPolicies: [
        ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
        ManagedPolicy.fromAwsManagedPolicyName("AmazonS3ReadOnlyAccess"),
      ],
    });

    // CloudWatch Logsの許可を追加
    sagemakerRole.addToPolicy(
      new PolicyStatement({
        actions: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ],
        resources: ["*"],
      })
    );

    // ECRへの認証許可を追加
    sagemakerRole.addToPolicy(
      new PolicyStatement({
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      })
    );

    // コンテナイメージのビルドとECRへのプッシュ
    const dockerImage = new DockerImageAsset(this, "YomiTokuDockerImage", {
      directory: containerPath,
      buildArgs: {},
      exclude: [".git", "node_modules"],
      platform: Platform.LINUX_AMD64,
    });

    // SageMaker Model
    const model = new CfnModel(this, "YomiTokuModel", {
      executionRoleArn: sagemakerRole.roleArn,
      modelName: modelName,
      primaryContainer: {
        image: dockerImage.imageUri,
        environment: environment,
      },
    });

    // SageMaker Endpoint Config
    const endpointConfig = new CfnEndpointConfig(
      this,
      "YomiTokuEndpointConfig",
      {
        endpointConfigName: endpointConfigName,
        executionRoleArn: sagemakerRole.roleArn,
        productionVariants: [
          {
            variantName: variantName,
            instanceType: instanceType,
            initialInstanceCount: 1,
            // 負荷分散戦略（オプション）
            routingConfig: {
              routingStrategy: "LEAST_OUTSTANDING_REQUESTS",
            },
          },
        ],
      }
    );

    // SageMaker Endpoint
    const endpoint = new CfnEndpoint(this, "YomiTokuEndpoint", {
      endpointName: this.endpointName,
      endpointConfigName: endpointConfig.attrEndpointConfigName,
    });

    endpoint.addDependency(endpointConfig);

    // 推論コンポーネント
    const inferenceComponent = new CfnInferenceComponent(
      this,
      "YomiTokuInferenceComponent",
      {
        inferenceComponentName: this.inferenceComponentName,
        endpointName: endpoint.endpointName!,
        variantName: variantName,
        specification: {
          modelName: model.modelName!,
          computeResourceRequirements: {
            numberOfAcceleratorDevicesRequired: 1, // GPU数
            numberOfCpuCoresRequired: 1, // CPU数
            minMemoryRequiredInMb: 4096, // 最小メモリ (4GB)
          },
        },
        runtimeConfig: {
          copyCount: 1, // 起動時のコピー数
        },
      }
    );

    inferenceComponent.addDependency(endpoint);
    inferenceComponent.addDependency(model);

    // SageMakerロールARNを保存
    this.sagemakerRoleArn = sagemakerRole.roleArn;

    // 出力値の設定
    new cdk.CfnOutput(this, "DockerImageUri", {
      value: dockerImage.imageUri,
      description: "ECRのDockerイメージURI",
    });

    new cdk.CfnOutput(this, "SageMakerEndpointName", {
      value: this.endpointName,
      description: "SageMakerエンドポイント名",
    });

    new cdk.CfnOutput(this, "SageMakerInferenceComponentName", {
      value: this.inferenceComponentName,
      description: "SageMaker推論コンポーネント名",
    });

    new cdk.CfnOutput(this, "SageMakerRoleArn", {
      value: this.sagemakerRoleArn,
      description: "SageMaker実行ロールARN",
    });
  }
}

import * as cdk from "aws-cdk-lib";
import {
  ScalableTarget,
  ServiceNamespace,
  TargetTrackingScalingPolicy,
  StepScalingPolicy,
  PredefinedMetric,
  AdjustmentType,
  MetricAggregationType,
} from "aws-cdk-lib/aws-applicationautoscaling";
import { Metric } from "aws-cdk-lib/aws-cloudwatch";
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

export interface OcrProps {
  baseName?: string;
  ocrEngine?: "paddle" | "deepseek" | "yomitoku-mp";
  instanceType?: string;
  environment?: Record<string, string>;
  enableZeroScale?: boolean;
  scaleInCooldownSeconds?: number;
  marketplaceModelPackageArn?: string;
}

export class Ocr extends Construct {
  public readonly endpointName: string;
  public readonly inferenceComponentName: string;
  public readonly sagemakerRoleArn: string;

  constructor(scope: Construct, id: string, props: OcrProps = {}) {
    super(scope, id);

    // デフォルト値の設定
    const baseName = props.baseName || "ocr";
    const ocrEngine = props.ocrEngine || "paddle";
    const isMarketplace = ocrEngine === "yomitoku-mp";

    if (isMarketplace && props.enableZeroScale) {
      throw new Error("sagemakerZeroScale is not supported with yomitoku-mp (Marketplace models do not support InferenceComponent)");
    }

    const instanceType =
      props.instanceType ||
      (isMarketplace
        ? "ml.g5.xlarge"
        : ocrEngine === "paddle"
        ? "ml.g4dn.2xlarge"
        : "ml.g4dn.4xlarge");

    // OCRエンジンに応じたコンテナパス（Marketplace は不要）
    const containerMap: Record<string, string> = {
      paddle: "paddle-ocr",
      deepseek: "deepseek-ocr",
    };

    const variantName = "AllTraffic";
    this.inferenceComponentName = isMarketplace
      ? ""
      : `${baseName}-inference-component`;

    // SageMaker用のIAMロール
    const sagemakerRole = new Role(this, "SageMakerExecutionRole", {
      assumedBy: new ServicePrincipal("sagemaker.amazonaws.com"),
      managedPolicies: [
        ManagedPolicy.fromAwsManagedPolicyName("AmazonSageMakerFullAccess"),
        ManagedPolicy.fromAwsManagedPolicyName("AmazonS3ReadOnlyAccess"),
      ],
    });

    sagemakerRole.addToPolicy(
      new PolicyStatement({
        actions: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ],
        resources: [
          `arn:aws:logs:${cdk.Stack.of(this).region}:${
            cdk.Stack.of(this).account
          }:log-group:/aws/sagemaker/*`,
        ],
      })
    );

    sagemakerRole.addToPolicy(
      new PolicyStatement({
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      })
    );

    let model: CfnModel;

    if (isMarketplace) {
      // Marketplace モデル（Yomitoku-Pro）: ModelPackage ARN を使用
      if (!props.marketplaceModelPackageArn) {
        throw new Error(
          "marketplaceModelPackageArn is required for yomitoku-mp engine"
        );
      }
      model = new CfnModel(this, "OcrModel", {
        executionRoleArn: sagemakerRole.roleArn,
        enableNetworkIsolation: true,
        primaryContainer: {
          modelPackageName: props.marketplaceModelPackageArn,
        },
      });
    } else {
      // 自前コンテナ（PaddleOCR / DeepSeek）
      const containerPath = path.join(
        __dirname,
        `../../ocr-containers/${containerMap[ocrEngine] || ocrEngine}`
      );

      let defaultEnv: Record<string, string> = {
        USE_GPU: "true",
        CUDA_VISIBLE_DEVICES: "0",
        OCR_ENGINE: ocrEngine,
      };

      if (ocrEngine === "deepseek") {
        defaultEnv = {
          ...defaultEnv,
          CROP_MODE: "true",
          MODEL_PATH: "deepseek-ai/DeepSeek-OCR",
          TORCH_CUDA_ARCH_LIST: "8.6",
          NVIDIA_VISIBLE_DEVICES: "all",
          NVIDIA_DRIVER_CAPABILITIES: "compute,utility",
        };
      }

      const environment = {
        ...defaultEnv,
        ...(props.environment || {}),
      };

      const dockerImage = new DockerImageAsset(this, "OcrDockerImage", {
        directory: containerPath,
        buildArgs: {},
        exclude: [".git", "node_modules"],
        platform: Platform.LINUX_AMD64,
      });

      model = new CfnModel(this, "OcrModel", {
        modelName: containerMap[ocrEngine],
        executionRoleArn: sagemakerRole.roleArn,
        primaryContainer: {
          image: dockerImage.imageUri,
          environment: environment,
        },
      });

      new cdk.CfnOutput(this, "DockerImageUri", {
        value: dockerImage.imageUri,
        description: "ECRのDockerイメージURI",
      });
    }

    const endpointConfig = new CfnEndpointConfig(this, "OcrEndpointConfig", {
      ...(isMarketplace ? {} : { executionRoleArn: sagemakerRole.roleArn }),
      productionVariants: [
        {
          variantName: variantName,
          ...(isMarketplace
            ? { modelName: model.attrModelName }
            : {}),
          instanceType: instanceType,
          initialInstanceCount: 1,
          ...(isMarketplace
            ? {}
            : {
                routingConfig: {
                  routingStrategy: "LEAST_OUTSTANDING_REQUESTS",
                },
              }),
          containerStartupHealthCheckTimeoutInSeconds: 600,
          modelDataDownloadTimeoutInSeconds: 600,
        },
      ],
    });

    const endpoint = new CfnEndpoint(this, "OcrEndpoint", {
      endpointConfigName: endpointConfig.attrEndpointConfigName,
    });

    this.endpointName = endpoint.attrEndpointName;
    endpoint.addDependency(endpointConfig);

    // InferenceComponent + Auto Scaling（Marketplace 以外のみ）
    if (!isMarketplace) {
      let cpuCores = 1;
      let memoryMb = 4096;
      let acceleratorDevices = 1;

      if (ocrEngine === "deepseek") {
        cpuCores = 8;
        memoryMb = 42768;
        acceleratorDevices = 1;
      }

      const inferenceComponent = new CfnInferenceComponent(
        this,
        "OcrInferenceComponent",
        {
          inferenceComponentName: this.inferenceComponentName,
          endpointName: endpoint.attrEndpointName,
          variantName: variantName,
          specification: {
            modelName: model.attrModelName,
            computeResourceRequirements: {
              numberOfAcceleratorDevicesRequired: acceleratorDevices,
              numberOfCpuCoresRequired: cpuCores,
              minMemoryRequiredInMb: memoryMb,
            },
          },
          runtimeConfig: {
            copyCount: 1,
          },
        }
      );

      inferenceComponent.addDependency(endpoint);
      inferenceComponent.addDependency(model);

      if (props.enableZeroScale) {
        const resourceId = `inference-component/${this.inferenceComponentName}`;

        const scalableTarget = new ScalableTarget(this, "ScalableTarget", {
          serviceNamespace: ServiceNamespace.SAGEMAKER,
          scalableDimension: "sagemaker:inference-component:DesiredCopyCount",
          resourceId: resourceId,
          minCapacity: 0,
          maxCapacity: 1,
        });

        scalableTarget.node.addDependency(inferenceComponent);

        new TargetTrackingScalingPolicy(this, "TargetTrackingPolicy", {
          scalingTarget: scalableTarget,
          targetValue: 1,
          predefinedMetric:
            PredefinedMetric.SAGEMAKER_INFERENCE_COMPONENT_INVOCATIONS_PER_COPY,
          scaleInCooldown: cdk.Duration.seconds(
            props.scaleInCooldownSeconds || 3600
          ),
          scaleOutCooldown: cdk.Duration.seconds(60),
        });

        const noCapacityMetric = new Metric({
          namespace: "AWS/SageMaker",
          metricName: "NoCapacityInvocationFailures",
          dimensionsMap: {
            InferenceComponentName: this.inferenceComponentName,
          },
          statistic: "Maximum",
          period: cdk.Duration.seconds(60),
        });

        new StepScalingPolicy(this, "StepScalingPolicy", {
          scalingTarget: scalableTarget,
          adjustmentType: AdjustmentType.CHANGE_IN_CAPACITY,
          metricAggregationType: MetricAggregationType.MAXIMUM,
          cooldown: cdk.Duration.seconds(60),
          scalingSteps: [
            { change: 1, lower: 0 },
            { change: 0, upper: -1 },
          ],
          metric: noCapacityMetric,
        });
      }
    }

    this.sagemakerRoleArn = sagemakerRole.roleArn;

    new cdk.CfnOutput(this, "SageMakerEndpointName", {
      value: this.endpointName,
      description: "SageMakerエンドポイント名",
    });

    if (!isMarketplace) {
      new cdk.CfnOutput(this, "SageMakerInferenceComponentName", {
        value: this.inferenceComponentName,
        description: "SageMaker推論コンポーネント名",
      });
    }

    new cdk.CfnOutput(this, "SageMakerRoleArn", {
      value: this.sagemakerRoleArn,
      description: "SageMaker実行ロールARN",
    });
  }
}

import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { Platform } from 'aws-cdk-lib/aws-ecr-assets';
import {
  StateMachine,
  Map,
  LogLevel,
  DefinitionBody,
} from 'aws-cdk-lib/aws-stepfunctions';
import { LambdaInvoke } from 'aws-cdk-lib/aws-stepfunctions-tasks';
import { DockerImageFunction, DockerImageCode } from 'aws-cdk-lib/aws-lambda';
import { LogGroup, RetentionDays } from 'aws-cdk-lib/aws-logs';
import { PolicyStatement } from 'aws-cdk-lib/aws-iam';

export interface StepFunctionsProps {
  imagesTable: cdk.aws_dynamodb.Table;
  jobsTable: cdk.aws_dynamodb.Table;
  schemasTable: cdk.aws_dynamodb.Table;
  documentBucket: cdk.aws_s3.Bucket;
  enableOcr: boolean;
  ocrEngine?: string;
  sagemakerEndpointName?: string;
  sagemakerInferenceComponentName?: string;
  modelId: string;
  modelRegion: string;
  agentRuntimeArn?: string;
  dsqlEndpoint?: string;
  dsqlRegion?: string;
  dsqlClusterArn?: string;
}

export class StepFunctions extends Construct {
  public readonly stateMachine: StateMachine;
  public readonly agentKickFunction?: DockerImageFunction;

  constructor(scope: Construct, id: string, props: StepFunctionsProps) {
    super(scope, id);

    const { modelId, modelRegion } = props;

    const processImage = new DockerImageFunction(this, 'ProcessImage', {
      code: DockerImageCode.fromImageAsset('lambda/api', {
        file: 'Dockerfile.stepfunctions',
        platform: Platform.LINUX_AMD64,
      }),
      timeout: cdk.Duration.minutes(15),
      memorySize: 4096,
      environment: {
        IMAGES_TABLE_NAME: props.imagesTable.tableName,
        JOBS_TABLE_NAME: props.jobsTable.tableName,
        SCHEMAS_TABLE_NAME: props.schemasTable.tableName,
        BUCKET_NAME: props.documentBucket.bucketName,
        MODEL_ID: modelId,
        MODEL_REGION: modelRegion,
        ENABLE_OCR: props.enableOcr.toString(),
        SAGEMAKER_ENDPOINT_NAME: props.sagemakerEndpointName || '',
        SAGEMAKER_INFERENCE_COMPONENT_NAME: props.sagemakerInferenceComponentName || '',
        OCR_ENGINE: props.ocrEngine || 'paddle',
      },
    });
    
    props.imagesTable.grantReadWriteData(processImage);
    props.jobsTable.grantReadWriteData(processImage);
    props.schemasTable.grantReadData(processImage);
    props.documentBucket.grantReadWrite(processImage);
    
    processImage.addToRolePolicy(new PolicyStatement({
      actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
      resources: ['*'],
    }));
    
    if (props.enableOcr && props.sagemakerEndpointName) {
      processImage.addToRolePolicy(new PolicyStatement({
        actions: ['sagemaker:InvokeEndpoint'],
        resources: ['*'],
      }));
    }

    const processImageTask = new LambdaInvoke(this, 'ProcessImageTask', {
      lambdaFunction: processImage,
      outputPath: '$.Payload',
    });

    // AgentKick Lambda (runs after ProcessImage, checks agent_enabled internally)
    let chainedDefinition: cdk.aws_stepfunctions.IChainable = processImageTask;

    if (props.agentRuntimeArn) {
      const agentKick = new DockerImageFunction(this, 'AgentKick', {
        code: DockerImageCode.fromImageAsset('lambda/api', {
          file: 'Dockerfile.agentkick',
          platform: Platform.LINUX_AMD64,
        }),
        timeout: cdk.Duration.minutes(10),
        memorySize: 512,
        environment: {
          SCHEMAS_TABLE_NAME: props.schemasTable.tableName,
          IMAGES_TABLE_NAME: props.imagesTable.tableName,
          JOBS_TABLE_NAME: props.jobsTable.tableName,
          BUCKET_NAME: props.documentBucket.bucketName,
          AGENT_RUNTIME_ARN: props.agentRuntimeArn,
          DSQL_ENDPOINT: props.dsqlEndpoint || '',
          DSQL_REGION: props.dsqlRegion || '',
          MODEL_ID: props.modelId,
          MODEL_REGION: props.modelRegion,
        },
      });

      props.imagesTable.grantReadWriteData(agentKick);
      props.schemasTable.grantReadData(agentKick);
      props.jobsTable.grantReadWriteData(agentKick);
      props.documentBucket.grantRead(agentKick);

      // AgentCore Runtime invoke
      agentKick.addToRolePolicy(new PolicyStatement({
        actions: ['bedrock-agentcore:InvokeAgentRuntime'],
        resources: [props.agentRuntimeArn, `${props.agentRuntimeArn}/*`],
      }));

      // DSQL access for resolving usecase tools
      if (props.dsqlClusterArn) {
        agentKick.addToRolePolicy(new PolicyStatement({
          actions: ['dsql:DbConnectAdmin'],
          resources: [props.dsqlClusterArn],
        }));
      }

      const agentKickTask = new LambdaInvoke(this, 'AgentKickTask', {
        lambdaFunction: agentKick,
        outputPath: '$.Payload',
      });

      chainedDefinition = processImageTask.next(agentKickTask);
      this.agentKickFunction = agentKick;
    }

    const processImagesMap = new Map(this, 'ProcessImagesMap', {
      maxConcurrency: 5,
      itemsPath: '$.images',
      parameters: {
        'image_id.$': '$$.Map.Item.Value.image_id',
        'skip_ocr.$': '$$.Map.Item.Value.skip_ocr',
        'job_id.$': '$.job_id',
      },
    });
    processImagesMap.itemProcessor(chainedDefinition);

    this.stateMachine = new StateMachine(this, 'StateMachine', {
      definitionBody: DefinitionBody.fromChainable(processImagesMap),
      timeout: cdk.Duration.hours(2),
      logs: {
        destination: new LogGroup(this, 'LogGroup', {
          retention: RetentionDays.ONE_WEEK,
        }),
        level: LogLevel.ALL,
      },
    });
  }
}

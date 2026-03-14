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
  sagemakerEndpointName?: string;
  sagemakerInferenceComponentName?: string;
  modelId: string;
  modelRegion: string;
}

export class StepFunctions extends Construct {
  public readonly stateMachine: StateMachine;

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

    const processImagesMap = new Map(this, 'ProcessImagesMap', {
      maxConcurrency: 5,
      itemsPath: '$.images',
      parameters: {
        'image_id.$': '$$.Map.Item.Value.image_id',
        'job_id.$': '$.job_id',
      },
    });
    processImagesMap.itemProcessor(processImageTask);

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

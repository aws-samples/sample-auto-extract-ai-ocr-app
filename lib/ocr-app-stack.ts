import * as cdk from "aws-cdk-lib";
import { Construct } from "constructs";

import { Auth } from "./constructs/auth";
import { Api } from "./constructs/api";
import { Web } from "./constructs/web";
import { Database } from "./constructs/database";
import { Ocr } from "./constructs/ocr";
import { Agent } from "./constructs/agent";
import { StepFunctions } from "./constructs/step-functions";
import { AppParameters } from "./parameters";

export interface OcrAppStackProps extends cdk.StackProps {
  params: AppParameters;
}

export class OcrAppStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: OcrAppStackProps) {
    super(scope, id, props);

    const p = props.params;

    const auth = new Auth(this, "Auth", {
      selfSignUpEnabled: p.selfSignUpEnabled,
      allowedSignUpEmailDomains: p.allowedSignUpEmailDomains,
    });

    const database = new Database(this, "Database");

    let ocrEndpoint = undefined;
    if (p.enableOcr) {
      const ocr = new Ocr(this, "OcrEndpoint", {
        enableZeroScale: p.sagemakerZeroScale,
        scaleInCooldownSeconds: p.sagemakerScaleInCooldownSeconds,
        ocrEngine: p.ocrEngine,
      });
      ocrEndpoint = ocr;
    }

    let agent = undefined;
    if (p.enableAgent) {
      agent = new Agent(this, "Agent", {
        region: this.region,
        enableDemo: p.enableAgentDemo,
        schemasTable: database.schemasTable,
      });
    }

    const api = new Api(this, "Api", {
      imagesTable: database.imagesTable,
      jobsTable: database.jobsTable,
      schemasTable: database.schemasTable,
      toolsTable: agent?.toolsTable,
      userPoolId: auth.userPool.userPoolId,
      userPoolClientId: auth.client.userPoolClientId,
      enableOcr: p.enableOcr,
      sagemakerEndpointName: ocrEndpoint?.endpointName,
      sagemakerInferenceComponentName: ocrEndpoint?.inferenceComponentName,
      agentRuntimeArn: agent?.runtimeArn,
      modelId: p.modelId,
      modelRegion: p.modelRegion,
    });

    const stepFunctions = new StepFunctions(this, "StepFunctions", {
      imagesTable: database.imagesTable,
      jobsTable: database.jobsTable,
      schemasTable: database.schemasTable,
      documentBucket: api.documentBucket,
      enableOcr: p.enableOcr,
      sagemakerEndpointName: ocrEndpoint?.endpointName,
      sagemakerInferenceComponentName: ocrEndpoint?.inferenceComponentName,
      modelId: p.modelId,
      modelRegion: p.modelRegion,
    });

    stepFunctions.stateMachine.grantStartExecution(api.handler);

    api.handler.addEnvironment(
      "STATE_MACHINE_ARN",
      stepFunctions.stateMachine.stateMachineArn
    );

    new Web(this, "WebConstruct", {
      buildFolder: "/dist",
      userPoolId: auth.userPool.userPoolId,
      userPoolClientId: auth.client.userPoolClientId,
      apiUrl: api.apiEndpoint,
      enableOcr: p.enableOcr,
      enableAgent: p.enableAgent,
      syncBucketName: api.syncBucket.bucketName,
      cloudFrontGeoRestriction: p.cloudFrontGeoRestriction,
    });

    new cdk.CfnOutput(this, "StateMachineArn", {
      value: stepFunctions.stateMachine.stateMachineArn,
      description: "OCR Step Functions State Machine ARN",
    });
  }
}

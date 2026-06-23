import * as cdk from "aws-cdk-lib";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

import { Auth } from "./constructs/auth";
import { Api } from "./constructs/api";
import { Web } from "./constructs/web";
import { Database } from "./constructs/database";
import { Dsql } from "./constructs/dsql";
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

    const database = new Database(this, "Database");

    const auth = new Auth(this, "Auth", {
      selfSignUpEnabled: p.selfSignUpEnabled,
      allowedSignUpEmailDomains: p.allowedSignUpEmailDomains,
    });

    const dsql = new Dsql(this, "Dsql", {
      userPoolId: auth.userPool.userPoolId,
      schemasTable: database.schemasTable,
    });

    // Post Auth Trigger に DSQL 接続情報を注入
    auth.postAuthFunction.addEnvironment("DSQL_ENDPOINT", dsql.clusterEndpoint);
    auth.postAuthFunction.addEnvironment("DSQL_REGION", this.region);
    auth.postAuthFunction.addToRolePolicy(
      new PolicyStatement({
        actions: ["dsql:DbConnectAdmin"],
        resources: [dsql.clusterArn],
      })
    );

    let ocrEndpoint = undefined;
    if (p.enableOcr) {
      const ocr = new Ocr(this, "OcrEndpoint", {
        enableZeroScale: p.sagemakerZeroScale,
        scaleInCooldownSeconds: p.sagemakerScaleInCooldownSeconds,
        ocrEngine: p.ocrEngine,
        marketplaceModelPackageArn: p.marketplaceModelPackageArn,
      });
      ocrEndpoint = ocr;
    }

    let agent = undefined;
    if (p.enableAgent) {
      agent = new Agent(this, "Agent", {
        region: this.region,
        enableDemo: p.enableAgentDemo,
        schemasTable: database.schemasTable,
        dsqlEndpoint: dsql.clusterEndpoint,
        dsqlRegion: this.region,
        dsqlClusterArn: dsql.clusterArn,
      });
    }

    const api = new Api(this, "Api", {
      imagesTable: database.imagesTable,
      jobsTable: database.jobsTable,
      schemasTable: database.schemasTable,
      userPreferencesTable: database.userPreferencesTable,
      // toolsTable removed — tools are now managed via AgentCore Gateway + DSQL
      userPoolId: auth.userPool.userPoolId,
      userPoolClientId: auth.client.userPoolClientId,
      enableOcr: p.enableOcr,
      ocrEngine: p.ocrEngine,
      sagemakerEndpointName: ocrEndpoint?.endpointName,
      sagemakerInferenceComponentName: ocrEndpoint?.inferenceComponentName,
      agentRuntimeArn: agent?.runtimeArn,
      modelId: p.modelId,
      modelRegion: p.modelRegion,
      dsqlEndpoint: dsql.clusterEndpoint,
      dsqlRegion: this.region,
      dsqlClusterArn: dsql.clusterArn,
    });

    const stepFunctions = new StepFunctions(this, "StepFunctions", {
      imagesTable: database.imagesTable,
      jobsTable: database.jobsTable,
      schemasTable: database.schemasTable,
      documentBucket: api.documentBucket,
      enableOcr: p.enableOcr,
      ocrEngine: p.ocrEngine,
      sagemakerEndpointName: ocrEndpoint?.endpointName,
      sagemakerInferenceComponentName: ocrEndpoint?.inferenceComponentName,
      modelId: p.modelId,
      modelRegion: p.modelRegion,
      enableAgent: p.enableAgent,
      agentRuntimeArn: agent?.runtimeArn,
      dsqlEndpoint: dsql.clusterEndpoint,
      dsqlRegion: this.region,
      dsqlClusterArn: dsql.clusterArn,
    });

    stepFunctions.stateMachine.grantStartExecution(api.handler);

    api.handler.addEnvironment(
      "STATE_MACHINE_ARN",
      stepFunctions.stateMachine.stateMachineArn
    );

    // AgentKick Lambda invoke from API
    if (stepFunctions.agentKickFunction) {
      api.handler.addEnvironment(
        "AGENT_KICK_FUNCTION_NAME",
        stepFunctions.agentKickFunction.functionName
      );
      stepFunctions.agentKickFunction.grantInvoke(api.handler);
    }

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

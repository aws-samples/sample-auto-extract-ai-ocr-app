import { Construct } from "constructs";
import { Duration, CfnOutput, CustomResource, RemovalPolicy } from "aws-cdk-lib";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import {
  PolicyStatement,
  Role,
  ServicePrincipal,
} from "aws-cdk-lib/aws-iam";
import { DockerImageCode, DockerImageFunction } from "aws-cdk-lib/aws-lambda";
import {
  Table,
  AttributeType,
  BillingMode,
} from "aws-cdk-lib/aws-dynamodb";
import { Provider } from "aws-cdk-lib/custom-resources";
import { CfnRuntime } from "aws-cdk-lib/aws-bedrockagentcore";
import * as path from "path";

export interface AgentProps {
  region: string;
  enableDemo?: boolean;
  schemasTable: Table;
}

export class Agent extends Construct {
  public readonly runtimeArn: string;
  public readonly role: Role;
  public readonly customersTable?: Table;
  public readonly toolsTable: Table;

  constructor(scope: Construct, id: string, props: AgentProps) {
    super(scope, id);

    // Create ToolsTable
    this.toolsTable = new Table(this, "ToolsTable", {
      partitionKey: { name: "tool_name", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
      pointInTimeRecovery: true,
    });

    // Docker image asset (ECRにpush)
    const dockerImage = new DockerImageAsset(this, "Image", {
      directory: path.join(__dirname, "../../agentcore/runtime"),
      platform: Platform.LINUX_ARM64,
    });

    // IAM Role for AgentCore Runtime
    this.role = new Role(this, "Role", {
      assumedBy: new ServicePrincipal("bedrock-agentcore.amazonaws.com"),
    });

    // Grant Bedrock permissions
    this.role.addToPolicy(
      new PolicyStatement({
        actions: ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        resources: ["*"],
      })
    );

    // Grant ECR permissions (automatic via grantPull)
    dockerImage.repository.grantPull(this.role);

    // Grant CloudWatch Logs permissions
    this.role.addToPolicy(
      new PolicyStatement({
        actions: [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ],
        resources: ["*"],
      })
    );

    // Grant X-Ray permissions for OpenTelemetry
    this.role.addToPolicy(
      new PolicyStatement({
        actions: [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
          "xray:GetSamplingRules",
          "xray:GetSamplingTargets",
        ],
        resources: ["*"],
      })
    );

    // Grant CloudWatch Metrics permissions
    this.role.addToPolicy(
      new PolicyStatement({
        actions: ["cloudwatch:PutMetricData"],
        resources: ["*"],
      })
    );

    // Grant ToolsTable write permissions
    this.toolsTable.grantWriteData(this.role);

    // Grant CustomersTable read permissions if provided
    if (this.customersTable) {
      this.customersTable.grantReadData(this.role);
    }

    // Demo data setup
    if (props.enableDemo) {
      // Create Customers table
      this.customersTable = new Table(this, "CustomersTable", {
        partitionKey: { name: "customer_id", type: AttributeType.STRING },
        billingMode: BillingMode.PAY_PER_REQUEST,
        removalPolicy: RemovalPolicy.DESTROY,
        pointInTimeRecovery: true,
      });

      // Add GSI for customer name search
      this.customersTable.addGlobalSecondaryIndex({
        indexName: "customer_name-index",
        partitionKey: { name: "customer_name", type: AttributeType.STRING },
      });

      // Grant DynamoDB read permissions
      this.customersTable.grantReadData(this.role);

      // Output table name
      new CfnOutput(this, "CustomersTableName", {
        value: this.customersTable.tableName,
        description: "DynamoDB Customers Table Name",
      });

      // Insert demo data
      const handler = new DockerImageFunction(this, "DemoDataHandler", {
        code: DockerImageCode.fromImageAsset("lambda/demo-custom-resource", {
          platform: Platform.LINUX_AMD64,
        }),
        timeout: Duration.seconds(60),
      });

      this.customersTable.grantWriteData(handler);
      props.schemasTable.grantWriteData(handler);

      const provider = new Provider(this, "DemoDataProvider", {
        onEventHandler: handler,
      });

      new CustomResource(this, "DemoData", {
        serviceToken: provider.serviceToken,
        properties: {
          CustomersTableName: this.customersTable.tableName,
          SchemasTableName: props.schemasTable.tableName,
        },
      });
    }

    // Create AgentCore Runtime
    const runtime = new CfnRuntime(this, "Runtime", {
      agentRuntimeName: "ocr_agent_runtime",
      agentRuntimeArtifact: {
        containerConfiguration: {
          containerUri: dockerImage.imageUri,
        },
      },
      roleArn: this.role.roleArn,
      networkConfiguration: {
        networkMode: "PUBLIC",
      },
      environmentVariables: {
        AWS_REGION: props.region,
        MAX_ITERATIONS: "20",
        CUSTOMERS_TABLE: this.customersTable?.tableName || "",
        TOOLS_TABLE: this.toolsTable.tableName,
      },
      description: "OCR Agent Runtime with DynamoDB search tools",
    });

    // Ensure role is created before runtime
    runtime.node.addDependency(this.role);

    this.runtimeArn = runtime.attrAgentRuntimeArn;

    // Output
    new CfnOutput(this, "AgentRuntimeArn", {
      value: this.runtimeArn,
      description: "Agent Runtime ARN",
    });

    new CfnOutput(this, "AgentRuntimeId", {
      value: runtime.attrAgentRuntimeId,
      description: "Agent Runtime ID",
    });
  }
}

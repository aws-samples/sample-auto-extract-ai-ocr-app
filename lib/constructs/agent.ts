import { Construct } from "constructs";
import {
  Aws,
  Duration,
  CfnOutput,
  CustomResource,
  RemovalPolicy,
} from "aws-cdk-lib";
import { DockerImageAsset, Platform } from "aws-cdk-lib/aws-ecr-assets";
import {
  PolicyStatement,
  Role,
  ServicePrincipal,
} from "aws-cdk-lib/aws-iam";
import {
  Architecture,
  DockerImageCode,
  DockerImageFunction,
} from "aws-cdk-lib/aws-lambda";
import {
  Table,
  AttributeType,
  BillingMode,
} from "aws-cdk-lib/aws-dynamodb";
import { Rule } from "aws-cdk-lib/aws-events";
import { LambdaFunction } from "aws-cdk-lib/aws-events-targets";
import { Provider } from "aws-cdk-lib/custom-resources";
import { CfnRuntime } from "aws-cdk-lib/aws-bedrockagentcore";
import * as agentcore from "@aws-cdk/aws-bedrock-agentcore-alpha";
import * as iam from "aws-cdk-lib/aws-iam";
import * as path from "path";
import * as fs from "fs";
import { envSuffix } from "../utils/naming";

export interface AgentProps {
  region: string;
  enableDemo?: boolean;
  schemasTable: Table;
  dsqlEndpoint: string;
  dsqlRegion: string;
  dsqlClusterArn: string;
  /** DSQL DDL custom resource — demo data must run after tables are created */
  dsqlDdlResource?: CustomResource;
  /** DSQL Seed custom resource — demo data must run after the 'all' group is created */
  dsqlSeedResource?: CustomResource;
  /** 環境名（base/dev/stg/prod）。リソース名の env suffix に使う。 */
  envName?: string;
}

export class Agent extends Construct {
  public readonly runtimeArn: string;
  public readonly role: Role;
  public readonly customersTable?: Table;
  public readonly gateway: agentcore.Gateway;
  public readonly gatewayEndpoint: string;

  constructor(scope: Construct, id: string, props: AgentProps) {
    super(scope, id);

    // =========================================================================
    // AgentCore Gateway (L2 Construct)
    // =========================================================================

    this.gateway = new agentcore.Gateway(this, "Gateway", {
      gatewayName: `ocr-tool-gateway${envSuffix(props.envName)}`,
      protocolConfiguration: new agentcore.McpProtocolConfiguration({
        supportedVersions: [agentcore.MCPProtocolVersion.MCP_2025_03_26],
      }),
      authorizerConfiguration: agentcore.GatewayAuthorizer.usingAwsIam(),
    });

    // Gateway Role に追加権限
    const gatewayRole = this.gateway.role as iam.Role;
    gatewayRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["bedrock-agentcore:GetGateway"],
        resources: [
          `arn:aws:bedrock-agentcore:${Aws.REGION}:${Aws.ACCOUNT_ID}:gateway/*`,
        ],
      })
    );

    // Gateway endpoint (手動構築 - L2 の gatewayUrl が /mcp を含むか要確認)
    this.gatewayEndpoint =
      this.gateway.gatewayUrl ||
      `https://${this.gateway.gatewayId}.gateway.bedrock-agentcore.${Aws.REGION}.amazonaws.com/mcp`;

    // =========================================================================
    // Tool Sync Custom Resource (Gateway → DSQL)
    // =========================================================================

    const toolSyncFn = new DockerImageFunction(this, "ToolSyncFunction", {
      code: DockerImageCode.fromImageAsset("lambda/tool-sync", {
        platform: Platform.LINUX_AMD64,
      }),
      timeout: Duration.seconds(120),
      memorySize: 256,
      environment: {
        GATEWAY_ENDPOINT: this.gatewayEndpoint,
        DSQL_ENDPOINT: props.dsqlEndpoint,
        DSQL_REGION: props.dsqlRegion,
      },
      description: "Syncs Gateway tools to DSQL via EventBridge",
    });

    // DSQL access
    toolSyncFn.addToRolePolicy(
      new PolicyStatement({
        actions: ["dsql:DbConnectAdmin"],
        resources: [props.dsqlClusterArn],
      })
    );

    // Gateway invoke (for tools/list)
    toolSyncFn.addToRolePolicy(
      new PolicyStatement({
        actions: [
          "bedrock-agentcore:InvokeGateway",
          "bedrock-agentcore:GetGateway",
        ],
        resources: [this.gateway.gatewayArn],
      })
    );

    // CustomResource: sync on deploy (initial + CDK-managed target changes)
    const toolSyncProvider = new Provider(this, "ToolSyncProvider", {
      onEventHandler: toolSyncFn,
    });

    const toolSyncResource = new CustomResource(this, "ToolSync", {
      serviceToken: toolSyncProvider.serviceToken,
      properties: {
        DeployTimestamp: Date.now().toString(),
      },
    });

    // Tool sync must run after DSQL DDL (tools table must exist)
    if (props.dsqlDdlResource) {
      toolSyncResource.node.addDependency(props.dsqlDdlResource);
    }

    // EventBridge Rule: sync on runtime Gateway Target changes
    new Rule(this, "ToolSyncRule", {
      eventPattern: {
        source: ["aws.bedrock-agentcore"],
        detailType: ["AWS API Call via CloudTrail"],
        detail: {
          eventSource: ["bedrock-agentcore.amazonaws.com"],
          eventName: [
            "CreateGatewayTarget",
            "UpdateGatewayTarget",
            "DeleteGatewayTarget",
          ],
        },
      },
      targets: [new LambdaFunction(toolSyncFn)],
    });

    // =========================================================================
    // Demo Lambda Targets (only deployed when enableDemo is true)
    // =========================================================================

    let customerLookupFn: DockerImageFunction | undefined;

    if (props.enableDemo) {
      // --- customer-lookup ---
      const customerLookupSchemaPath = path.join(
        __dirname,
        "../../lambda/tools/customer-lookup/tool-schema.json"
      );
      const customerLookupSchema = JSON.parse(
        fs.readFileSync(customerLookupSchemaPath, "utf8")
      );
      const customerLookupToolSchema = agentcore.ToolSchema.fromInline(
        customerLookupSchema.tools as any
      );

      customerLookupFn = new DockerImageFunction(
        this,
        "CustomerLookupFunction",
        {
          code: DockerImageCode.fromImageAsset("lambda/tools/customer-lookup", {
            platform: Platform.LINUX_ARM64,
          }),
          architecture: Architecture.ARM_64,
          timeout: Duration.seconds(30),
          memorySize: 256,
          description: "AgentCore Gateway Target: customer-lookup",
        }
      );

      const customerLookupTarget = this.gateway.addLambdaTarget(
        "CustomerLookupTarget",
        {
          gatewayTargetName: "customer-lookup",
          lambdaFunction: customerLookupFn,
          toolSchema: customerLookupToolSchema,
          description: "Lambda target for customer-lookup",
        }
      );
      // L2 calls grantInvoke but doesn't set dependency — target creation
      // fails validation if the role policy hasn't propagated yet
      customerLookupTarget.node.addDependency(gatewayRole);

      // --- calc-verify ---
      const calcVerifySchemaPath = path.join(
        __dirname,
        "../../lambda/tools/calc-verify/tool-schema.json"
      );
      const calcVerifySchema = JSON.parse(
        fs.readFileSync(calcVerifySchemaPath, "utf8")
      );
      const calcVerifyToolSchema = agentcore.ToolSchema.fromInline(
        calcVerifySchema.tools as any
      );

      const calcVerifyFn = new DockerImageFunction(this, "CalcVerifyFunction", {
        code: DockerImageCode.fromImageAsset("lambda/tools/calc-verify", {
          platform: Platform.LINUX_ARM64,
        }),
        architecture: Architecture.ARM_64,
        timeout: Duration.seconds(30),
        memorySize: 256,
        description: "AgentCore Gateway Target: calc-verify",
      });

      const calcVerifyTarget = this.gateway.addLambdaTarget("CalcVerifyTarget", {
        gatewayTargetName: "calc-verify",
        lambdaFunction: calcVerifyFn,
        toolSchema: calcVerifyToolSchema,
        description: "Lambda target for calc-verify",
      });
      calcVerifyTarget.node.addDependency(gatewayRole);

      // Tool sync must wait for demo targets to be created
      toolSyncResource.node.addDependency(customerLookupTarget);
      toolSyncResource.node.addDependency(calcVerifyTarget);
    }

    // =========================================================================
    // Docker image for AgentCore Runtime
    // =========================================================================

    const dockerImage = new DockerImageAsset(this, "Image", {
      directory: path.join(__dirname, "../../agentcore/runtime"),
      platform: Platform.LINUX_ARM64,
    });

    // =========================================================================
    // IAM Role for AgentCore Runtime
    // =========================================================================

    this.role = new Role(this, "Role", {
      assumedBy: new ServicePrincipal("bedrock-agentcore.amazonaws.com"),
    });

    // Bedrock model invocation
    this.role.addToPolicy(
      new PolicyStatement({
        actions: ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
        resources: ["*"],
      })
    );

    // ECR pull
    dockerImage.repository.grantPull(this.role);

    // CloudWatch Logs
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

    // X-Ray (OpenTelemetry)
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

    // CloudWatch Metrics
    this.role.addToPolicy(
      new PolicyStatement({
        actions: ["cloudwatch:PutMetricData"],
        resources: ["*"],
      })
    );

    // Gateway invoke permissions for Runtime
    this.role.addToPolicy(
      new PolicyStatement({
        actions: [
          "bedrock-agentcore:InvokeGateway",
          "bedrock-agentcore:GetGateway",
          "bedrock-agentcore:ListGateways",
        ],
        resources: [this.gateway.gatewayArn],
      })
    );

    // =========================================================================
    // Demo data (CustomersTable)
    // =========================================================================

    if (props.enableDemo) {
      this.customersTable = new Table(this, "CustomersTable", {
        partitionKey: { name: "customer_id", type: AttributeType.STRING },
        billingMode: BillingMode.PAY_PER_REQUEST,
        removalPolicy: RemovalPolicy.DESTROY,
        pointInTimeRecovery: true,
      });

      this.customersTable.addGlobalSecondaryIndex({
        indexName: "CustomerNameIndex",
        partitionKey: { name: "customer_name", type: AttributeType.STRING },
      });

      // Grant customer-lookup Lambda Target access to CustomersTable
      this.customersTable.grantReadData(customerLookupFn!);

      // Pass table name as env var to customer-lookup Lambda
      customerLookupFn!.addEnvironment(
        "CUSTOMERS_TABLE",
        this.customersTable.tableName
      );

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
        environment: {
          DSQL_ENDPOINT: props.dsqlEndpoint,
          DSQL_REGION: props.dsqlRegion,
        },
      });

      this.customersTable.grantWriteData(handler);
      props.schemasTable.grantWriteData(handler);

      // DSQL access for usecase + usecase_tools insertion
      handler.addToRolePolicy(
        new PolicyStatement({
          actions: ["dsql:DbConnectAdmin"],
          resources: [props.dsqlClusterArn],
        })
      );

      const provider = new Provider(this, "DemoDataProvider", {
        onEventHandler: handler,
      });

      const demoDataResource = new CustomResource(this, "DemoData", {
        serviceToken: provider.serviceToken,
        properties: {
          CustomersTableName: this.customersTable.tableName,
          SchemasTableName: props.schemasTable.tableName,
        },
      });

      // Demo data must run after tool-sync (tools must exist in DSQL first)
      demoDataResource.node.addDependency(toolSyncResource);

      // Demo data must run after DSQL DDL (tables must exist)
      if (props.dsqlDdlResource) {
        demoDataResource.node.addDependency(props.dsqlDdlResource);
      }

      // Demo data must run after DSQL Seed (the 'all' group must exist before
      // we grant viewer permission to it in group_usecases)
      if (props.dsqlSeedResource) {
        demoDataResource.node.addDependency(props.dsqlSeedResource);
      }
    }

    // =========================================================================
    // AgentCore Runtime
    // =========================================================================

    const runtime = new CfnRuntime(this, "Runtime", {
      agentRuntimeName: `ocr_agent_runtime${envSuffix(props.envName, "_")}`,
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
        AGENTCORE_GATEWAY_ENDPOINT: this.gatewayEndpoint,
      },
      description: "OCR Agent Runtime with Gateway MCP tools",
    });

    runtime.node.addDependency(this.role);

    this.runtimeArn = runtime.attrAgentRuntimeArn;

    // =========================================================================
    // Outputs
    // =========================================================================

    new CfnOutput(this, "AgentRuntimeArn", {
      value: this.runtimeArn,
      description: "Agent Runtime ARN",
    });

    new CfnOutput(this, "AgentRuntimeId", {
      value: runtime.attrAgentRuntimeId,
      description: "Agent Runtime ID",
    });

    new CfnOutput(this, "GatewayEndpoint", {
      value: this.gatewayEndpoint,
      description: "AgentCore Gateway MCP Endpoint",
    });
  }
}

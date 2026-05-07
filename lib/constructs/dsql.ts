import { CfnOutput, RemovalPolicy, CustomResource, Duration } from "aws-cdk-lib";
import { CfnCluster } from "aws-cdk-lib/aws-dsql";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Provider } from "aws-cdk-lib/custom-resources";
import { Table } from "aws-cdk-lib/aws-dynamodb";
import { Construct } from "constructs";
import * as path from "path";

export interface DsqlProps {
  userPoolId: string;
  schemasTable: Table;
}

export class Dsql extends Construct {
  public readonly clusterEndpoint: string;
  public readonly clusterArn: string;
  public readonly clusterIdentifier: string;

  constructor(scope: Construct, id: string, props: DsqlProps) {
    super(scope, id);

    const cluster = new CfnCluster(this, "Cluster", {
      deletionProtectionEnabled: false,
    });
    cluster.applyRemovalPolicy(RemovalPolicy.DESTROY);

    this.clusterIdentifier = cluster.attrIdentifier;
    this.clusterArn = cluster.attrResourceArn;
    this.clusterEndpoint = `${cluster.attrIdentifier}.dsql.${cluster.stack.region}.on.aws`;

    // DSQL 管理用 Lambda
    const adminFn = new NodejsFunction(this, "AdminFunction", {
      runtime: Runtime.NODEJS_20_X,
      entry: path.join(__dirname, "../../lambda/dsql-admin/index.ts"),
      handler: "handler",
      timeout: Duration.minutes(5),
      bundling: {
        commandHooks: {
          beforeBundling: () => [],
          beforeInstall: () => [],
          afterBundling: (inputDir: string, outputDir: string) => [
            `cp ${inputDir}/lambda/dsql-admin/ddl.sql ${outputDir}/ddl.sql`,
          ],
        },
      },
      environment: {
        DSQL_ENDPOINT: this.clusterEndpoint,
        DSQL_REGION: cluster.stack.region,
        USER_POOL_ID: props.userPoolId,
        SCHEMAS_TABLE_NAME: props.schemasTable.tableName,
      },
    });

    adminFn.addToRolePolicy(
      new PolicyStatement({
        actions: ["dsql:DbConnectAdmin"],
        resources: [this.clusterArn],
      })
    );
    adminFn.addToRolePolicy(
      new PolicyStatement({
        actions: ["cognito-idp:ListUsers"],
        resources: [`arn:aws:cognito-idp:${cluster.stack.region}:${cluster.stack.account}:userpool/${props.userPoolId}`],
      })
    );
    props.schemasTable.grantReadData(adminFn);

    // DDL 実行
    const ddlProvider = new Provider(this, "DdlProvider", {
      onEventHandler: adminFn,
    });

    const ddlResource = new CustomResource(this, "DdlExecution", {
      serviceToken: ddlProvider.serviceToken,
      properties: {
        action: "ddl",
        version: "1",
      },
    });

    // Seed 実行（DDL の後に実行）
    const seedProvider = new Provider(this, "SeedProvider", {
      onEventHandler: adminFn,
    });

    const seedResource = new CustomResource(this, "SeedExecution", {
      serviceToken: seedProvider.serviceToken,
      properties: {
        action: "seed",
        version: "1",
      },
    });
    seedResource.node.addDependency(ddlResource);

    new CfnOutput(this, "ClusterEndpoint", {
      value: this.clusterEndpoint,
      description: "Aurora DSQL Cluster Endpoint",
    });

    new CfnOutput(this, "ClusterArn", {
      value: this.clusterArn,
      description: "Aurora DSQL Cluster ARN",
    });
  }
}

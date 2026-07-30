import { Duration, CfnOutput } from "aws-cdk-lib";
import { Construct } from "constructs";
import { UserPool, UserPoolClient } from "aws-cdk-lib/aws-cognito";
import { Runtime } from "aws-cdk-lib/aws-lambda";
import { NodejsFunction } from "aws-cdk-lib/aws-lambda-nodejs";
import { PolicyStatement } from "aws-cdk-lib/aws-iam";
import { Table } from "aws-cdk-lib/aws-dynamodb";
import * as agw from "aws-cdk-lib/aws-apigatewayv2";
import * as agwi from "aws-cdk-lib/aws-apigatewayv2-integrations";
import * as agwa from "aws-cdk-lib/aws-apigatewayv2-authorizers";
import * as path from "path";

export interface WebSocketProps {
  userPool: UserPool;
  userPoolClient: UserPoolClient;
  connectionsTable: Table;
  imagesTable: Table;
  dsqlEndpoint: string;
  dsqlRegion: string;
  dsqlClusterArn: string;
}

/**
 * プレゼンス機能用の WebSocket API。
 *
 * 参照実装: https://github.com/aws-samples/websocket-api-cognito-auth-sample
 * （2025-12-12 に Archived。実装をコピーする前提の参照）
 *
 * 認証はクエリ文字列 `idToken` で Cognito ID トークンを渡す方式（$connect のみ）。
 * resource_id（image_id）の紐付けは $connect 時ではなく、接続後の最初のメッセージ
 * （{action: "subscribe", imageId: "..."}）で行う。
 */
export class WebSocket extends Construct {
  public readonly api: agw.WebSocketApi;
  private readonly stageName = "prod";

  constructor(scope: Construct, id: string, props: WebSocketProps) {
    super(scope, id);

    // Lambda Authorizer（$connect 専用、Cognito IDトークン検証のみ）
    const authHandler = new NodejsFunction(this, "AuthorizerFunction", {
      runtime: Runtime.NODEJS_20_X,
      entry: path.join(__dirname, "../../lambda/websocket/authorizer/index.ts"),
      handler: "handler",
      timeout: Duration.seconds(10),
      environment: {
        USER_POOL_ID: props.userPool.userPoolId,
        APP_CLIENT_ID: props.userPoolClient.userPoolClientId,
      },
    });

    // $connect / $disconnect / $default 統合Lambda
    const websocketHandler = new NodejsFunction(this, "HandlerFunction", {
      runtime: Runtime.NODEJS_20_X,
      entry: path.join(__dirname, "../../lambda/websocket/handler/index.ts"),
      handler: "handler",
      timeout: Duration.seconds(30),
      environment: {
        CONNECTIONS_TABLE_NAME: props.connectionsTable.tableName,
        IMAGES_TABLE_NAME: props.imagesTable.tableName,
        DSQL_ENDPOINT: props.dsqlEndpoint,
        DSQL_REGION: props.dsqlRegion,
      },
    });

    props.connectionsTable.grantReadWriteData(websocketHandler);
    props.imagesTable.grantReadData(websocketHandler);
    websocketHandler.addToRolePolicy(
      new PolicyStatement({
        actions: ["dsql:DbConnectAdmin"],
        resources: [props.dsqlClusterArn],
      })
    );

    const authorizer = new agwa.WebSocketLambdaAuthorizer("Authorizer", authHandler, {
      identitySource: ["route.request.querystring.idToken"],
    });

    this.api = new agw.WebSocketApi(this, "Api", {
      connectRouteOptions: {
        authorizer,
        integration: new agwi.WebSocketLambdaIntegration("ConnectIntegration", websocketHandler),
      },
      disconnectRouteOptions: {
        integration: new agwi.WebSocketLambdaIntegration("DisconnectIntegration", websocketHandler),
      },
      defaultRouteOptions: {
        integration: new agwi.WebSocketLambdaIntegration("DefaultIntegration", websocketHandler),
      },
    });

    new agw.WebSocketStage(this, "Stage", {
      webSocketApi: this.api,
      stageName: this.stageName,
      autoDeploy: true,
    });

    // websocketHandler がクライアントへ PostToConnection できるように権限付与
    this.api.grantManageConnections(websocketHandler);

    new CfnOutput(this, "Endpoint", {
      value: this.apiEndpoint,
      description: "WebSocket API Endpoint for Presence feature",
    });
  }

  public get apiEndpoint(): string {
    // WebSocketApi.apiEndpoint は wss:// スキームを返す
    return `${this.api.apiEndpoint}/${this.stageName}`;
  }
}

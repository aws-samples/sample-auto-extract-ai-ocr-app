import { Construct } from "constructs";
import { RemovalPolicy, CfnOutput } from "aws-cdk-lib";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";

export class Database extends Construct {
  public readonly imagesTable: Table;
  public readonly jobsTable: Table;
  public readonly schemasTable: Table;
  public readonly userPreferencesTable: Table;
  public readonly connectionsTable: Table;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    // 画像情報を保存するテーブル
    this.imagesTable = new Table(this, "ImagesTable", {
      partitionKey: { name: "id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // 開発環境用。本番環境では RETAIN にすべき
      pointInTimeRecovery: true,
    });

    // GSI を追加（アプリ名でのフィルタリング用）
    this.imagesTable.addGlobalSecondaryIndex({
      indexName: "AppNameIndex",
      partitionKey: { name: "app_name", type: AttributeType.STRING },
      sortKey: { name: "upload_time", type: AttributeType.STRING },
    });

    // GSI: アップロードユーザーでのフィルタリング用
    this.imagesTable.addGlobalSecondaryIndex({
      indexName: "UploadedByIndex",
      partitionKey: { name: "uploaded_by", type: AttributeType.STRING },
      sortKey: { name: "upload_time", type: AttributeType.STRING },
    });

    // ジョブ情報を保存するテーブル
    this.jobsTable = new Table(this, "JobsTable", {
      partitionKey: { name: "id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // 開発環境用
      pointInTimeRecovery: true,
    });

    this.jobsTable.addGlobalSecondaryIndex({
      indexName: "ImageIdIndex",
      partitionKey: { name: "image_id", type: AttributeType.STRING },
      sortKey: { name: "created_at", type: AttributeType.STRING },
    });

    // スキーマ情報を保存するテーブル
    this.schemasTable = new Table(this, "SchemasTable", {
      partitionKey: { name: "schema_type", type: AttributeType.STRING },
      sortKey: { name: "name", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // 開発環境用
      pointInTimeRecovery: true,
    });

    // ユーザー設定テーブル（Star 等）
    this.userPreferencesTable = new Table(this, "UserPreferencesTable", {
      partitionKey: { name: "user_id", type: AttributeType.STRING },
      sortKey: { name: "sk", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
    });

    // WebSocket 接続管理テーブル（プレゼンス機能用）
    // PK: resource_id（例: "image#<image_id>"）, SK: connection_id
    // TTL(removed_at) + Heartbeat による定期更新で disconnect 検知の穴を補う
    this.connectionsTable = new Table(this, "ConnectionsTable", {
      partitionKey: { name: "resource_id", type: AttributeType.STRING },
      sortKey: { name: "connection_id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY,
      timeToLiveAttribute: "removed_at",
    });

    // GSI: connection_id からの逆引き（$disconnect 時に resource_id が分からないため必須）
    this.connectionsTable.addGlobalSecondaryIndex({
      indexName: "ConnectionIdIndex",
      partitionKey: { name: "connection_id", type: AttributeType.STRING },
    });

    // テーブル名を出力
    new CfnOutput(this, "ImagesTableName", {
      value: this.imagesTable.tableName,
      description: "DynamoDB Images Table Name",
    });

    new CfnOutput(this, "JobsTableName", {
      value: this.jobsTable.tableName,
      description: "DynamoDB Jobs Table Name",
    });

    new CfnOutput(this, "SchemasTableName", {
      value: this.schemasTable.tableName,
      description: "DynamoDB Schemas Table Name",
    });

    new CfnOutput(this, "ConnectionsTableName", {
      value: this.connectionsTable.tableName,
      description: "DynamoDB WebSocket Connections Table Name",
    });
  }
}

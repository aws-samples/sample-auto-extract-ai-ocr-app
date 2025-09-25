import { Construct } from "constructs";
import { RemovalPolicy, CfnOutput } from "aws-cdk-lib";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";

export interface DatabaseProps {
  // DynamoDB は VPC 不要なので props は空でも良い
}

export class Database extends Construct {
  public readonly imagesTable: Table;
  public readonly jobsTable: Table;
  public readonly schemasTable: Table;

  constructor(scope: Construct, id: string, props: DatabaseProps = {}) {
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

    // ジョブ情報を保存するテーブル
    this.jobsTable = new Table(this, "JobsTable", {
      partitionKey: { name: "id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // 開発環境用
      pointInTimeRecovery: true,
    });

    // スキーマ情報を保存するテーブル
    this.schemasTable = new Table(this, "SchemasTable", {
      partitionKey: { name: "schema_type", type: AttributeType.STRING },
      sortKey: { name: "name", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
      removalPolicy: RemovalPolicy.DESTROY, // 開発環境用
      pointInTimeRecovery: true,
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
  }
}

import { Construct } from "constructs";
import { RemovalPolicy, CfnOutput, Duration } from "aws-cdk-lib";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import { PythonFunction } from "@aws-cdk/aws-lambda-python-alpha";
import { Runtime, Architecture } from "aws-cdk-lib/aws-lambda";
import { Asset } from "aws-cdk-lib/aws-s3-assets";
import * as path from "path";

export interface DatabaseProps {
  // DynamoDB は VPC 不要なので props は空でも良い
}

export class Database extends Construct {
  public readonly imagesTable: Table;
  public readonly jobsTable: Table;
  public readonly schemasTable: Table;
  public readonly initSchemasLambda: PythonFunction;

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

    // デフォルトスキーマファイルをS3アセットとして準備
    const schemaAsset = new Asset(this, "SchemaAsset", {
      path: path.join(__dirname, "../../default-schemas/apps-schemas.json"),
    });

    // スキーマ初期化用のLambda関数
    this.initSchemasLambda = new PythonFunction(this, "InitSchemasFunction", {
      entry: path.join(__dirname, "../../lambda/init-schemas"),
      runtime: Runtime.PYTHON_3_9,
      architecture: Architecture.X86_64,
      index: "index.py",
      handler: "handler",
      environment: {
        SCHEMAS_TABLE_NAME: this.schemasTable.tableName,
        DEFAULT_S3_BUCKET: schemaAsset.s3BucketName,
        DEFAULT_S3_KEY: schemaAsset.s3ObjectKey,
      },
      timeout: Duration.minutes(5),
    });

    // Lambda関数にスキーマテーブルへの書き込み権限を付与
    this.schemasTable.grantReadWriteData(this.initSchemasLambda);

    // Lambda関数にS3アセットへのアクセス権限を付与
    schemaAsset.grantRead(this.initSchemasLambda);

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

    // スキーマ初期化Lambda関数の名前を出力（手動実行用）
    new CfnOutput(this, "InitSchemasLambdaName", {
      value: this.initSchemasLambda.functionName,
      description: "Lambda function name for initializing schemas",
    });

    // S3アセット情報を出力（デバッグ用）
    new CfnOutput(this, "SchemaAssetBucket", {
      value: schemaAsset.s3BucketName,
      description: "S3 bucket containing schema asset",
    });

    new CfnOutput(this, "SchemaAssetKey", {
      value: schemaAsset.s3ObjectKey,
      description: "S3 key for schema asset",
    });
  }
}

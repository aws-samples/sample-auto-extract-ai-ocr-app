# デプロイ手順

このドキュメントは TypeScript CDK を用いた AWS 環境へのデプロイ手順を解説します。詳細は [AWS CDK TypeScript ガイド](https://docs.aws.amazon.com/ja_jp/cdk/v2/guide/work-with-cdk-typescript.html)もご参照ください。

## 前提条件

- AWS 認証情報の設定
  - `aws configure` で設定
  - Administrator 権限を推奨
  - 詳細は [AWS CLI の設定ガイド](https://docs.aws.amazon.com/ja_jp/cli/latest/userguide/cli-configure-files.html)を参照
- AWS CDK のインストール
  - Node.js 環境が必要
  - `npm install -g aws-cdk` でインストール
- Docker 環境
  - [Docker Desktop](https://www.docker.com/products/docker-desktop/) のインストールが必要

## Bedrock モデルのセットアップ

必要なモデルの有効化手順

1. AWS コンソールにログインし、Bedrock サービスに移動
2. リージョンを `バージニア北部（us-east-1）` に変更
3. 以下のモデルへのアクセスを有効化：

- Anthropic Claude 3.7 Sonnet

## 使用するモデルの変更

`cdk.json` にて、使用する Bedrock モデルの ID とリージョンを指定することができます。モデルの ID は [Amazon Bedrock でサポートされている基盤モデル](https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/models-supported.html) を参照すると良いです。また、使用するモデルを変更する場合は、上記のステップと同様にモデルアクセスを有効化する必要があります。

```
"model_id": "anthropic.claude-3-5-sonnet-20240620-v1:0",
"model_region": "us-east-1",
```

## CDK による AWS リソースのデプロイ

CDK デプロイの際に必要な依存パッケージのインストールします。

```sh
npm i
```

新規の AWS アカウント/リージョンで初めて CDK を使用する場合は、以下のコマンドを実行してください。

```sh
cdk bootstrap
```

AWS リソースのデプロイを行います。リソースの変更を行った際は毎回このコマンドを実行してください。

```sh
cdk deploy
```

デプロイ後に出力される `OcrAppStack.WebConstructCloudFrontURL` の URL にアクセスすることで、Web サイトにアクセスできます。

## Lambda によるスキーマの初期化

Lambda のコンソールから `OcrAppStack-DatabaseInitSchemasFunction` で始まる Lambda 関数を実行すると、スキーマの初期化が行われます。CDK によるデプロイが終わった後にこちらの Lambda を実行すると、`default-schemas` ディレクトリにあるスキーマ情報に基づいて、デフォルトのスキーマが追加されます。

> 現在は、アプリケーションからスキーマを作成する機能が追加されているため、この Lambda 関数は実行する必要はありません。デバッグ用の機能です。

## AWS リソースの削除

削除するとリソースとデータは完全に消去されるので注意してください。

```sh
cdk destroy
```

## 開発方法

### CDK Hotswap 機能の活用

AWS CDK では、開発サイクルを効率化するために「hotswap」機能が提供されています。この機能を使うと、CloudFormation スタック全体の更新をバイパスし、変更されたコンポーネント（Lambda 関数や ECS タスク定義など）のみを迅速に更新できます。

使用する場合は、アプリケーションコードを変更した後、通常の cdk deploy の代わりに以下のコマンドを使用します：

```bash
cdk deploy --hotswap
```

### ローカルでの開発手順

#### 1. 環境変数の設定

`cdk deploy` コマンドの実行後、出力されるリソース情報を利用してアプリケーションの環境変数を設定します。

出力例:

```
Outputs:
OcrAppStack.ApiApiEndpointE2C5D803 = https://XXXXXXXXXXXX.execute-api.us-east-1.amazonaws.com/prod/
OcrAppStack.ApiDocumentBucketName14F33E89 = ocrappstack-apidocumentbucket1e0f08d4-XXXXXXXXXXXX
OcrAppStack.ApiImagesTableName87FC28D3 = OcrAppStack-DatabaseImagesTable3098F792-XXXXXXXXXXXX
OcrAppStack.ApiJobsTableName16618860 = OcrAppStack-DatabaseJobsTable7C20F61C-XXXXXXXXXXXX
OcrAppStack.ApiOcrApiEndpoint94C64180 = https://XXXXXXXXXXXX.execute-api.us-east-1.amazonaws.com/prod/
OcrAppStack.AuthUserPoolClientId8216BF9A = XXXXXXXXXXXX
OcrAppStack.AuthUserPoolIdC0605E59 = us-east-1_XXXXXXXXXXXX
OcrAppStack.DatabaseImagesTableName88591548 = OcrAppStack-DatabaseImagesTable3098F792-XXXXXXXXXXXX
OcrAppStack.DatabaseInitSchemasLambdaNameD08AA3A3 = OcrAppStack-DatabaseInitSchemasFunctionAF9A0DE0-XXXXXXXXXXXX
OcrAppStack.DatabaseJobsTableNameFCF442A2 = OcrAppStack-DatabaseJobsTable7C20F61C-XXXXXXXXXXXX
OcrAppStack.DatabaseSchemaAssetBucketEBD470E0 = cdk-hnb659fds-assets-XXXXXXXXXXXX-us-east-1
OcrAppStack.DatabaseSchemaAssetKeyD848FF37 = XXXXXXXXXXXX.json
OcrAppStack.DatabaseSchemasTableNameCF14F20C = OcrAppStack-DatabaseSchemasTable97CF304A-XXXXXXXXXXXX
OcrAppStack.NetworkVpcIdF213FEE5 = vpc-XXXXXXXXXXXX
OcrAppStack.OcrEndpointDockerImageUriDFE2281D = XXXXXXXXXXXX.dkr.ecr.us-east-1.amazonaws.com/cdk-hnb659fds-container-assets-XXXXXXXXXXXX-us-east-1:XXXXXXXXXXXX
OcrAppStack.OcrEndpointSageMakerEndpointName031E6036 = yomitoku-endpoint
OcrAppStack.OcrEndpointSageMakerInferenceComponentNameAD008265 = yomitoku-inference-component
OcrAppStack.OcrEndpointSageMakerRoleArn4F9772E2 = arn:aws:iam::XXXXXXXXXXXX:role/OcrAppStack-OcrEndpointSageMakerExecutionRoleF2F0DF-XXXXXXXXXXXX
OcrAppStack.WebConstructCloudFrontURL2550F65B = https://XXXXXXXXXXXX.cloudfront.net
OcrAppStack.ApiCPUApiUrl5804A8EA = https://XXXXXXXXXXXX.cloudfront.net
OcrAppStack.ApiGpuGPUApiUrl995B8AA0 = https://XXXXXXXXXXXX.cloudfront.net
OcrAppStack.AuthUserPoolClientId8216BF9A = XXXXXXXXXXXX
OcrAppStack.AuthUserPoolIdC0605E59 = ap-northeast-XXXXXXXXXXXX
OcrAppStack.DatabaseBastionHostId64F20BB1 = i-XXXXXXXXXXXX
OcrAppStack.NetworkVpcIdF213FEE5 = vpc-XXXXXXXXXXXX
OcrAppStack.WebConstructCloudFrontURL2550F65B = https://XXXXXXXXXXXX.cloudfront.net
```

この出力情報を基に、プロジェクトルートの `web` ディレクトリにある `.sample-env` ファイルを参考にして、新規に `.env` ファイルを作成します。

#### 2. 環境変数ファイルの設定例

`.sample-env` ファイルをコピーして `.env` ファイルを作成し、以下のように `cdk deploy` の出力値を使って設定します：

```properties
VITE_APP_USER_POOL_CLIENT_ID=<YOUR_USER_POOL_CLIENT_ID>   # AuthUserPoolClientId の値
VITE_APP_USER_POOL_ID=<YOUR_USER_POOL_ID>                # AuthUserPoolId の値
VITE_APP_REGION=<YOUR_AWS_REGION>                        # リージョン名（デプロイしたリージョンです）
VITE_API_BASE_URL=<YOUR_API_ENDPOINT>                    # ApiOcrApiEndpoint の値
VITE_ENABLE_OCR=false                                    # OCR機能の有効/無効
```

#### 3. ローカル開発サーバーの起動

環境変数の設定が完了したら、以下のコマンドでローカル開発サーバーを起動できます：

```bash
cd web
npm install
npm run dev
```

ブラウザで `http://localhost:3000` を開くと、アプリケーションにアクセスできます。

#### 4. API との連携

ローカル開発環境は、CDK でデプロイされたバックエンドリソース（API Gateway、Lambda、Bedrock など）と連携します。API 呼び出しはすべて `.env` ファイルで設定した `VITE_API_BASE_URL` のエンドポイントに転送されます。

#### 5. デバッグとテスト

開発中のデバッグにはブラウザの開発者ツールを活用してください。Network タブで API リクエストの状況を確認できます。

変更を加えるたびに自動的にアプリケーションが再ビルドされ、ブラウザが更新されます。

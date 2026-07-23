# AutoExtract

AutoExtract は OCR + Bedrock を活用した帳票読み取りの AI-OCR ソリューションです。帳票からの情報抽出を半自動化し、人間によるデータ入力チェックをサポートするツールです。

![demo](docs/imgs/demo.gif)

## 主な機能

- **OCR と情報抽出**：SageMaker 上の OCR エンジン（PaddleOCR / Yomitoku / DeepSeek）で読み取り、Bedrock でスキーマに沿った情報を抽出します。
- **ユースケース管理**：抽出したい項目（スキーマ）を LLM で自動生成・編集できます。複数ページ PDF は全ページ統合とページ別個別を選べます。
- **エージェント検証**：Bedrock AgentCore が抽出結果を自動で検証し、修正候補を提案します。
- **権限管理と共有**：ユースケースをユーザーやグループ単位で共有でき、管理画面から一元管理できます。
- **リアルタイム連携**：WebSocket で、閲覧中のユーザーをお互いに表示します。

## アーキテクチャ

![architecture](docs/imgs/architecture.drawio.png)

React（CloudFront + S3）、FastAPI（Lambda）、AWS CDK による 3 層構成です。OCR は SageMaker、情報抽出は Bedrock、エージェント検証は Bedrock AgentCore を利用し、権限管理を Aurora DSQL、帳票データを DynamoDB で管理します。詳細は [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) を参照してください。

## クイックスタート

デプロイには Node.js と Docker、設定済みの AWS 認証情報が必要です。

```sh
# 1. 依存パッケージのインストール
npm ci

# 2. (初回のみ) 対象アカウント / リージョンで CDK bootstrap
cdk bootstrap

# 3. デプロイ
npm run cdk:deploy
```

`npm run cdk:deploy` は、対話型のデプロイラッパー `scripts/cdk.sh` を起動します。引数なしで実行すると、コマンド、環境（base / dev / stg / prod）、リージョンを対話的に選択できます。引数で直接指定することもできます（npm 経由の場合は `--` が必須です）。

```sh
npm run cdk:deploy -- dev --region us-east-1
```

デプロイ後に出力される `WebConstructCloudFrontURL` の URL にアクセスすると Web サイトを開けます。管理画面を使うには最初のユーザーを管理者へ昇格させる必要があります（[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) の「初期管理者の設定」を参照）。

## ドキュメント

- [デプロイと設定ガイド](docs/DEPLOYMENT.md)：デプロイ方法、パラメータ設定、OCR エンジンの変更、初期管理者の設定、ローカル開発。
- [実装説明](docs/ARCHITECTURE.md)：全体構成、処理の流れ、データストアの使い分け、権限設計、エージェント検証。

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

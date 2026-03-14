---
inclusion: always
---

# Technology Stack

## インフラ
- AWS CDK (TypeScript) によるIaCデプロイ
- AWS Lambda (Python / Docker) - API バックエンド
- Amazon API Gateway - REST API
- Amazon DynamoDB - データストア (Jobs, Images, Schemas テーブル)
- Amazon S3 - ドキュメント・画像保存
- Amazon SageMaker - OCR 推論エンドポイント
- Amazon Bedrock (Claude) - LLM による情報抽出・スキーマ生成
- Amazon Bedrock AgentCore - AI Agent 検証機能
- Amazon CloudFront + S3 - フロントエンドホスティング
- Amazon Cognito - 認証
- AWS Step Functions - バックグラウンド処理

## バックエンド
- Python 3 (FastAPI)
- Pydantic - スキーマバリデーション
- boto3 - AWS SDK

## フロントエンド
- React 18 + TypeScript
- Vite - ビルドツール
- Tailwind CSS - スタイリング
- React Router - ルーティング
- AWS Amplify - Cognito 認証連携

## コーディング規約
- バックエンド: Python の型ヒントを使用、ドメイン駆動設計に準拠
- フロントエンド: 関数コンポーネント + Hooks パターン
- CDK: Construct パターンで機能単位に分割

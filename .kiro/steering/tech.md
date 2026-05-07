---
inclusion: always
---

# Technology Stack

## インフラ
- AWS CDK (TypeScript) によるIaCデプロイ
- AWS Lambda (Python / Docker) - API バックエンド
- Amazon API Gateway - REST API
- Amazon DynamoDB - データストア (Jobs, Images, Schemas, UserPreferences テーブル + Agent 有効時: Tools, Customers)
- Amazon Aurora DSQL - RBAC 権限管理 (PostgreSQL 互換、サーバーレス)
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

## フロントエンド カラーテーマ方針
- 色の定義は `web/src/index.css` の CSS 変数（`:root { --color-xxx }`)で一元管理
- `tailwind.config.js` で CSS 変数を参照し、Tailwind カスタムカラーとして登録（例: `primary: 'var(--color-primary)'`）
- コンポーネントでは Tailwind クラス（`bg-primary`, `border-default`, `text-muted` 等）のみ使用する
- `var(--color-xxx)` をコンポーネントの style や className に直接書かない
- 新しい色が必要な場合: index.css に CSS 変数追加 → tailwind.config.js に登録 → Tailwind クラスで使用

## コーディング規約
- バックエンド: Python の型ヒントを使用、ドメイン駆動設計に準拠
- フロントエンド: 関数コンポーネント + Hooks パターン
- CDK: Construct パターンで機能単位に分割

## 静的解析
- `lambda/api/app/` 配下の Python ファイルを変更した場合、必ず pyflakes で構文エラー・未使用 import を確認する
```bash
cd lambda/api/app && python -m pyflakes routers/ services/ domains/ schemas/ utils/ repositories/ clients/ workers/ dependencies/
```

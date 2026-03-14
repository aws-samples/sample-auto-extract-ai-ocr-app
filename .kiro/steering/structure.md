---
inclusion: always
---

# Project Structure

```
├── bin/                        # CDK アプリケーションエントリポイント
│   └── ocr-app.ts
├── lib/                        # CDK スタック・コンストラクト定義
│   ├── parameters.ts           # アプリパラメータ管理（環境切り替え）
│   ├── ocr-app-stack.ts        # メインスタック
│   └── constructs/             # 機能別 Construct
│       ├── api.ts              # API Gateway + Lambda
│       ├── auth.ts             # Cognito 認証 + Pre Sign-up Lambda
│       ├── database.ts         # DynamoDB テーブル
│       ├── dsql.ts             # Aurora DSQL (RBAC マスタ)
│       ├── ocr.ts              # SageMaker OCR エンドポイント
│       ├── web.ts              # CloudFront + S3 フロントエンド
│       ├── agent.ts            # Bedrock AgentCore
│       └── step-functions.ts   # Step Functions ワークフロー
├── lambda/                     # Lambda 関数
│   ├── api/                    # メイン API (FastAPI)
│   │   ├── app/
│   │   │   ├── domains/        # ビジネスロジック
│   │   │   ├── services/       # サービス層
│   │   │   ├── repositories/   # データアクセス層
│   │   │   ├── routers/        # API エンドポイント定義
│   │   │   ├── schemas/        # Pydantic スキーマ
│   │   │   └── utils/          # ユーティリティ
│   │   └── Dockerfile
│   ├── pre-signup/             # Cognito Pre Sign-up トリガー (TypeScript)
│   └── demo-custom-resource/   # デモデータ投入用 Custom Resource
├── web/                        # フロントエンド (React + Vite)
│   └── src/
│       ├── components/         # 共通コンポーネント
│       ├── pages/              # ページコンポーネント
│       ├── types/              # TypeScript 型定義
│       ├── utils/              # ユーティリティ
│       └── styles/             # スタイル
├── agentcore/                  # Bedrock AgentCore ランタイム
├── ocr-containers/             # OCR コンテナイメージ
│   ├── paddle-ocr/
│   └── deepseek-ocr/
├── cdk.json                    # CDK feature flags のみ
└── docs/                       # ドキュメント・画像
```

## 命名規約
- CDK Construct: PascalCase (例: `OcrEndpoint`)
- Lambda ファイル: snake_case (例: `extraction_service.py`)
- React コンポーネント: PascalCase (例: `ExtractedInfoDisplay.tsx`)
- 型定義ファイル: kebab-case (例: `app-schema.ts`)

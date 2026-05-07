---
inclusion: always
---

# Project Structure

```
├── bin/                        # CDK エントリポイント
│   └── ocr-app.ts
├── lib/                        # CDK スタック・コンストラクト
│   ├── parameters.ts           # アプリパラメータ（環境切り替え）
│   ├── ocr-app-stack.ts
│   └── constructs/             # 機能別 Construct
│       ├── api.ts              # API Gateway + Lambda
│       ├── auth.ts             # Cognito 認証
│       ├── database.ts         # DynamoDB テーブル
│       ├── dsql.ts             # Aurora DSQL
│       ├── ocr.ts              # SageMaker OCR エンドポイント
│       ├── web.ts              # CloudFront + S3
│       ├── agent.ts            # Bedrock AgentCore
│       └── step-functions.ts   # Step Functions ワークフロー
├── lambda/
│   ├── api/                    # メイン API (FastAPI / Docker)
│   │   ├── app/
│   │   │   ├── main.py         # FastAPI 初期化・サービス DI（app.state）
│   │   │   ├── config.py       # アプリケーション設定（Settings クラス）
│   │   │   ├── background.py   # バックグラウンドタスク拡張
│   │   │   ├── clients/        # AWS SDK クライアント一元管理
│   │   │   │   ├── aws.py      # S3, DynamoDB, SageMaker, SFn 等 + エンドポイント操作
│   │   │   │   ├── bedrock.py  # Bedrock API 呼び出し + リトライ
│   │   │   │   ├── dsql.py     # DSQL 接続・OCC リトライ
│   │   │   │   └── agent.py    # AgentCore Runtime クライアント
│   │   │   ├── dependencies/   # FastAPI Depends 注入
│   │   │   │   ├── services.py # サービス DI（app.state → Service 取得）
│   │   │   │   └── auth.py     # 認証・認可 Depends ヘルパー
│   │   │   ├── workers/        # Step Functions Lambda ハンドラー
│   │   │   │   └── step_functions.py
│   │   │   ├── routers/        # API エンドポイント定義
│   │   │   ├── services/       # サービス層（ユースケース単位）
│   │   │   ├── domains/        # ビジネスロジック（純粋関数）
│   │   │   ├── repositories/   # データアクセス層
│   │   │   ├── schemas/        # Pydantic スキーマ
│   │   │   └── utils/          # 純粋関数ユーティリティ
│   │   ├── Dockerfile
│   │   └── Dockerfile.stepfunctions
│   ├── pre-signup/             # Cognito Pre Sign-up (TypeScript)
│   ├── post-auth/              # Cognito Post Auth → DSQL 同期
│   ├── dsql-admin/             # DSQL DDL/SQL 管理
│   └── demo-custom-resource/   # デモデータ投入
├── web/                        # フロントエンド (React + Vite)
│   └── src/
│       ├── components/
│       │   ├── ui/             # 汎用 UI 部品
│       │   │   ├── Button, Input, Select, Modal, Table ...
│       │   │   ├── Badge, Alert, Toast, Tooltip
│       │   │   ├── CardTable, Pagination, Tabs
│       │   │   └── index.ts    # 一括 re-export
│       │   ├── layout/         # レイアウト
│       │   │   ├── AppLayout, Sidebar, UserMenu
│       │   │   └── AuthWrapper
│       │   └── shared/         # 複数ページ共有の業務コンポーネント
│       │       ├── StatusBadge, ConfirmModal
│       │       ├── CustomPromptModal, PermissionModal
│       ├── pages/
│       │   ├── Home, Stars, History, NotFound
│       │   ├── Admin.tsx       # 管理者ページ（タブ切り替え）
│       │   ├── admin/          # Admin 専用コンポーネント
│       │   │   ├── UsersTab, GroupsTab, UsecasesTab
│       │   │   ├── ToolsTab, ImagesTab
│       │   │   ├── UserDetailModal, StatsCards, AdminToolbar
│       │   ├── upload/         # アップロード + ファイル一覧
│       │   │   ├── Upload, FileList, OcrActionBar
│       │   │   ├── S3SyncModal, SharingModal, LoadingToast
│       │   ├── ocr-result/     # OCR 結果表示・編集・抽出
│       │   │   ├── OCRResult, ImagePreview, OcrResultEditor
│       │   │   ├── ExtractedInfoDisplay, ExtractionStatusDisplay
│       │   │   ├── AgentValidationPanel, ReExtractModal
│       │   └── schema/         # スキーマ定義
│       │       ├── SchemaGenerator, SchemaPreview
│       ├── services/           # API 呼び出し層
│       │   ├── api.ts          # axios インスタンス + インターセプター
│       │   ├── adminApi.ts, ocrApi.ts, imageApi.ts
│       ├── hooks/              # カスタムフック
│       │   └── usePolling.ts
│       ├── contexts/           # React Context
│       │   └── AppContext.tsx
│       ├── types/              # TypeScript 型定義
│       │   ├── user, group, usecase, tool
│       │   ├── ocr, extraction, agent, app-schema
│       └── utils/
│           └── dateUtils.ts
├── agentcore/                  # Bedrock AgentCore ランタイム
├── ocr-containers/             # OCR コンテナイメージ
└── docs/                       # ドキュメント
```

## 命名規約

- CDK Construct: PascalCase（例: `OcrEndpoint`）
- Lambda ファイル: snake_case（例: `extraction_service.py`）
- React コンポーネント: PascalCase（例: `ExtractedInfoDisplay.tsx`）

## web/src/ フロントエンド構成方針

- `components/ui/` — テーマ非依存の汎用 UI 部品
- `components/shared/` — 複数ページで使う業務コンポーネント
- `pages/<feature>/` — ページ専用コンポーネントを同ディレクトリに配置
- `services/` — API 呼び出し層（機能別分割）
- `utils/` — 純粋関数のみ（API 呼び出しは services/ に置く）

コンポーネント配置: 1 ページ専用 → `pages/<feature>/`、2 ページ以上で共有 → `components/shared/`、汎用 → `components/ui/`

## lambda/api/app/ レイヤー設計

依存方向:
```
routers → dependencies/ → services → domains
                                    → repositories → clients/（DB系: DynamoDB, DSQL）
                                    → clients/（外部API系: Bedrock, SageMaker, SFn, S3, AgentCore）
workers/ → services（FastAPI DI 不使用、自前インスタンス化）
```

外部 API 系の clients/ について: Clean Architecture では Gateway/Adapter 層を介した抽象化が推奨されるが、現時点ではプロジェクト規模に対してオーバーエンジニアリングとなるため、services/ から clients/ の薄いラッパーを直接呼び出す設計としている。

各レイヤーの責務詳細は `docs/api-structure.md` を参照。

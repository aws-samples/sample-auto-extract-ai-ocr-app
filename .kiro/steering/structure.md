---
inclusion: always
---

# Project Structure

```
├── .kiro/
│   ├── specs/                      # Kiro Spec ファイル（機能単位）
│   │   ├── access-control/         # ユーザー/グループ/権限管理 + 共有
│   │   ├── authentication/         # Cognito 認証・DSQL ユーザー同期
│   │   ├── image-lifecycle/        # Image の全ライフサイクル
│   │   ├── ocr-processing/        # OCR 処理エンジン
│   │   ├── data-extraction/       # Bedrock 構造化データ抽出
│   │   ├── schema-management/     # スキーマ（ユースケース）定義・管理
│   │   ├── s3-sync/               # S3 バッチ同期
│   │   ├── user-preferences/      # ユーザー個人設定（スター）
│   │   └── agent-validation/      # AI Agent 検証（Experimental）
│   ├── steering/                   # プロジェクトコンテキスト
│   ├── plans/                      # 実装プラン
│   └── settings/                   # Kiro 設定
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
│   │   │   ├── main.py         # FastAPI アプリ初期化・サービス DI（app.state）
│   │   │   ├── clients/         # AWS SDK クライアント一元管理
│   │   │   │   ├── aws.py      # S3, DynamoDB, SageMaker 等グローバルインスタンス
│   │   │   │   ├── bedrock.py  # Bedrock API 呼び出し + リトライ
│   │   │   │   ├── dsql.py     # DSQL 接続管理・OCC リトライ・query ヘルパー
│   │   │   │   └── agent.py    # AgentCore Runtime クライアント
│   │   │   ├── dependencies/   # FastAPI Depends 注入
│   │   │   │   ├── services.py # サービス DI（app.state → Service 取得）
│   │   │   │   └── auth.py     # 認証・認可 Depends ヘルパー
│   │   │   ├── workers/        # Step Functions Lambda エントリポイント
│   │   │   │   └── step_functions.py
│   │   │   ├── routers/        # API エンドポイント定義
│   │   │   ├── services/       # サービス層
│   │   │   ├── domains/        # ビジネスロジック（純粋関数のみ）
│   │   │   ├── repositories/   # データアクセス層
│   │   │   ├── schemas/        # Pydantic スキーマ
│   │   │   └── utils/          # 純粋関数ユーティリティ
│   │   ├── Dockerfile
│   │   └── Dockerfile.stepfunctions
│   ├── pre-signup/             # Cognito Pre Sign-up トリガー (TypeScript)
│   ├── post-auth/              # Cognito Post Auth トリガー (Python) → DSQL 同期
│   ├── dsql-admin/             # DSQL 管理用 Lambda (TypeScript) DDL/SQL 実行
│   └── demo-custom-resource/   # デモデータ投入用 Custom Resource
├── web/                        # フロントエンド (React + Vite)
│   └── src/
│       ├── components/
│       │   ├── ui/             # デザインシステム（汎用 UI 部品）
│       │   ├── layout/         # レイアウトコンポーネント
│       │   └── shared/         # 複数ページで使う業務コンポーネント
│       │       ├── StatusBadge.tsx
│       │       ├── ConfirmModal.tsx
│       │       ├── CustomPromptModal.tsx
│       │       └── PermissionModal.tsx
│       ├── contexts/           # React Context
│       │   └── AppContext.tsx
│       ├── hooks/              # 共通カスタムフック
│       │   └── usePolling.ts
│       ├── services/           # API 呼び出し層（機能別分割）
│       │   ├── api.ts          # axios インスタンス + インターセプター
│       │   ├── adminApi.ts     # 管理者 API
│       │   ├── ocrApi.ts       # OCR / Agent / 抽出関連
│       │   └── imageApi.ts     # 画像 CRUD
│       ├── pages/              # ページコンポーネント（機能単位）
│       │   ├── Home.tsx
│       │   ├── Stars.tsx
│       │   ├── History.tsx
│       │   ├── Admin.tsx
│       │   ├── admin/          # 管理者ページ（タブ分割）
│       │   │   ├── AdminToolbar.tsx
│       │   │   ├── StatsCards.tsx
│       │   │   ├── UsersTab.tsx
│       │   │   ├── UserDetailModal.tsx
│       │   │   ├── GroupsTab.tsx
│       │   │   ├── UsecasesTab.tsx
│       │   │   ├── ToolsTab.tsx
│       │   │   └── ImagesTab.tsx
│       │   ├── upload/         # Upload ページ + 専用コンポーネント
│       │   │   ├── Upload.tsx
│       │   │   ├── FileList.tsx
│       │   │   ├── OcrActionBar.tsx
│       │   │   ├── S3SyncModal.tsx
│       │   │   ├── SharingModal.tsx
│       │   │   └── LoadingToast.tsx
│       │   ├── ocr-result/     # OCRResult ページ + 専用コンポーネント
│       │   │   ├── OCRResult.tsx
│       │   │   ├── ImagePreview.tsx
│       │   │   ├── OcrResultEditor.tsx
│       │   │   ├── ExtractionStatusDisplay.tsx
│       │   │   ├── ExtractedInfoDisplay.tsx
│       │   │   ├── AgentValidationPanel.tsx
│       │   │   └── ReExtractModal.tsx
│       │   └── schema/         # SchemaGenerator + 専用コンポーネント
│       │       ├── SchemaGenerator.tsx
│       │       └── SchemaPreview.tsx
│       ├── types/              # TypeScript 型定義
│       └── utils/              # 純粋関数ユーティリティ
│           └── dateUtils.ts
├── agentcore/                  # Bedrock AgentCore ランタイム
├── ocr-containers/             # OCR コンテナイメージ
├── cdk.json                    # CDK feature flags のみ
└── docs/                       # ドキュメント・画像
```

## 命名規約
- CDK Construct: PascalCase (例: `OcrEndpoint`)
- Lambda ファイル: snake_case (例: `extraction_service.py`)
- React コンポーネント: PascalCase (例: `ExtractedInfoDisplay.tsx`)
- 型定義ファイル: kebab-case (例: `app-schema.ts`)

## web/src/ フロントエンド構成方針

### ディレクトリの責務

- `components/ui/` — テーマ非依存の汎用 UI 部品。業務ロジックを含まない
- `components/layout/` — ヘッダー、サイドバー等のレイアウト
- `components/shared/` — 複数ページで使う業務コンポーネント（StatusBadge, ConfirmModal 等）
- `pages/` — ルーティング単位のページ。機能単位でサブディレクトリに分割
- `pages/<feature>/` — ページ専用コンポーネントを同ディレクトリに配置（upload/, ocr-result/, schema/）
- `services/` — API 呼び出し層。機能別に分割（api.ts, adminApi.ts, ocrApi.ts, imageApi.ts）
- `hooks/` — 共通カスタムフック（usePolling 等）
- `contexts/` — React Context（グローバル状態）
- `types/` — TypeScript 型定義
- `utils/` — 純粋関数ユーティリティのみ（副作用のある API 呼び出しは services/ に置く）

### コンポーネント配置ルール

- 1 つのページでしか使わないコンポーネント → そのページの `pages/<feature>/` に配置
- 2 つ以上のページで使うコンポーネント → `components/shared/` に配置
- テーマ・業務非依存の汎用部品 → `components/ui/` に配置

## lambda/api/app/ レイヤー設計

依存方向:
```
routers → dependencies/ → services → domains
                                    → repositories → clients/（DB系: DynamoDB, DSQL）
                                    → clients/（外部API系: Bedrock, SageMaker, SFn, S3, AgentCore）
workers/ → services（FastAPI DI 不使用、自前インスタンス化）
```

外部 API 系の clients/ について: Clean Architecture では Gateway/Adapter 層を介した抽象化が推奨されるが、現時点ではプロジェクト規模に対してオーバーエンジニアリングとなるため、services/ から clients/ の薄いラッパーを直接呼び出す設計としている。

### clients/ — AWS SDK クライアント一元管理

AWS SDK クライアントの生成・グローバルインスタンス管理・外部 API ラッパー。
- `aws.py`: S3, DynamoDB, SageMaker 等のグローバルインスタンス
- `bedrock.py`: Bedrock API 呼び出し + リトライ
- `dsql.py`: DSQL 接続 + OCC リトライ
- `agent.py`: AgentCore Runtime クライアント

### dependencies/ — FastAPI Depends 注入

FastAPI の `Depends` で Router に注入されるもの。
- `services.py`: `app.state` からサービスインスタンスを取得
- `auth.py`: 認証・認可チェック（`require_auth`, `RequirePermission` 等）

### workers/ — Step Functions Lambda エントリポイント

FastAPI Lambda とは別の Lambda のエントリポイント。
`services/` を利用するが、FastAPI の DI（`app.state`）は使わず自前でインスタンス化する。

### routers/ — プレゼンテーション層

HTTP リクエスト/レスポンスの入出口。責務は以下に限定する:

- リクエストのパース・バリデーション（Pydantic スキーマ利用）
- 認証・認可チェック（`Depends(require_auth)`, `Depends(RequirePermission(...))` 等）
- Service の呼び出し（`Depends(get_xxx_service)` で DI）
- HTTPException への変換・レスポンス返却

ルーターに書いてはいけないもの:

- ビジネスルール（「auto グループは削除不可」等）→ サービス層に置く
- データレベルのフィルタリング（「admin なら全件、一般ユーザーなら権限のある分だけ」等）→ サービス層に user_id/role を渡して中で分岐する
- Repository の直接呼び出し → 必ずサービス層を経由する
- cognito_sub → email 等のデータ加工 → サービス層または utils/ で行う

ルーターに書いてよいもの:

- `Depends` ベースの認証・認可（「このエンドポイントにアクセスできるか」）
- Pydantic `Literal` 等による入力値の形式バリデーション
- cognito_sub の取得（`get_cognito_sub(req)`）とサービスへの受け渡し
- サービスの戻り値をそのまま返す、または HTTPException に変換する

### 入力バリデーションの方針

- 列挙値（role, permission 等）は Pydantic の `Literal` 型で定義し、ルーターでの手動チェックを避ける
- 形式バリデーション（空チェック、型チェック等）は Pydantic スキーマまたはルーターで行う
- ビジネスルールに基づくバリデーション（「最後の owner は削除不可」等）はサービス層で行う

### 認証認可の責務分離

認証認可は2つのレベルに分かれ、それぞれ異なるレイヤーが担当する:

- エンドポイントレベル（「この API にアクセスできるか」）→ ルーター層の `Depends` で制御
- データレベル（「このデータをこのユーザーに返していいか」）→ サービス層で user_id/role を受け取って制御

### 認証認可ヘルパー（utils/auth.py）

FastAPI `Depends` ベースで統一。ルーターから利用する:

- `Depends(require_auth)` — ログイン必須
- `Depends(RequireRole(min_role))` — 指定ロール以上を要求（admin, author, reader）
- `Depends(RequirePermission(min_level))` — ユースケース単位の権限チェック（owner/editor/viewer）。パスパラメータ `app_name` を自動取得し、admin は全権限スキップ

### サービス DI パターン（dependencies.py）

全サービスは `main.py` で `app.state` に生成し、`dependencies/services.py` の `Depends` 関数経由で Router に注入:

- `Depends(get_upload_service)`, `Depends(get_schema_service)`, `Depends(get_admin_service)`, `Depends(get_user_service)`, `Depends(get_sharing_service)` 等
- `workers/step_functions.py`（Step Functions Lambda）は FastAPI コンテキスト外のため `Depends` 対象外。自前でインスタンス生成する。

### services/ — アプリケーション層（ユースケース）

ユースケース単位の処理フローを調整する。責務:

- 複数の Repository / Domain を組み合わせたオーケストレーション
- トランザクション境界の管理
- 外部サービス呼び出しの調整（S3, Step Functions 等）
- ユーザーの role/権限に応じたデータフィルタリング（「admin なら全件」等の分岐）
- ビジネスルールの適用（「auto グループは削除不可」「所有者チェック」等）

サービスに書いてはいけないもの:

- HTTP 固有の処理（Request オブジェクトの参照、HTTPException の直接生成は許容するが、HTTP ヘッダーの解析等は避ける）
- 複数サービスで重複するデータ加工ロジック → utils/ に切り出す

### domains/ — ドメイン層（ビジネスロジック）

外部依存を持たない純粋なビジネスロジック。DB や AWS SDK に直接依存しない。

### repositories/ — データアクセス層

データストア（DynamoDB, DSQL, S3）への読み書きを抽象化する。ビジネスロジックを含めない。

### schemas/ — リクエスト/レスポンス定義

Pydantic モデルによる API の入出力スキーマ定義。複数の router/service から参照されるモデルはここに置く。
router 内でしか使わないリクエストモデル（例: admin の GroupCreate 等）は router 内で定義してよい。

### utils/ — 横断的ユーティリティ

認証ヘルパー、画像処理、Bedrock 呼び出しなど、特定レイヤーに属さない共通機能。

- 複数サービスで使うデータ加工ロジックはここに置く（例: `enrich_image_emails` — 画像リストへの email 付与）
- 特定の 1 サービスでしか使わないロジックはそのサービス内に置く

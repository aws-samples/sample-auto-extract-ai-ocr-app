# データモデル設計

## データストアの使い分け

| データストア | 用途 | 選定理由 |
|---|---|---|
| Aurora DSQL | RBAC 権限管理（Users, Groups, Usecases, Tools + 中間テーブル） | VPC 不要のサーバーレス構成。多対多リレーションを中間テーブル + JOIN で表現。OCC によるトランザクション整合性 |
| DynamoDB | 帳票データ（Images, Jobs, Schemas, UserPreferences） | スキーマレスで構造不定のデータ（OCR 結果等）を Map 型で格納。GSI による柔軟なアクセスパターン。Streams で分析基盤への連携が可能 |

前方互換性の観点から、業務データのマスタは DynamoDB に集約されている。DSQL は RBAC 権限管理に特化している。

---

## ER 図

> DSQL は FK 制約非サポート。参照関係はアプリケーション層で管理。
> `app_name` はユースケースの識別子（英数字+アンダースコア）。DSQL・DynamoDB 横断の結合キー。表示名は `display_name` で別管理。

### テーブル役割の補足

- `SchemasTable`（DynamoDB）= ユースケース定義のマスタ。フィールド定義・入力方法・カスタムプロンプト等の設定を保持。元々はこのテーブルだけでユースケースを管理していた
- `usecases`（DSQL）= RBAC 権限管理用に後から追加。SchemasTable の `name`（= `app_name`）で紐付け。ユースケース作成時は SchemasTable → DSQL usecases の順で伝播する
- `ImagesTable`（DynamoDB）= 個々の帳票画像レコード。ユースケースではない。`app_name` でどのユースケースに属するかを示す
- `JobsTable`（DynamoDB）= Agent 検証ジョブ専用。OCR・抽出の処理管理には使われていない（それらは ImagesTable のステータス遷移 + Step Functions で管理）

```mermaid
classDiagram
    direction TB

    namespace Cognito {
        class AmazonCognito {
            string sub PK
            string email
        }
    }

    namespace DSQL_Master {
        class users {
            UUID id PK
            VARCHAR cognito_sub UNIQUE
            VARCHAR email UNIQUE
            VARCHAR display_name
            VARCHAR department
            VARCHAR role
            BOOLEAN is_active
            TIMESTAMP created_at
            TIMESTAMP updated_at
        }
        class groups {
            UUID id PK
            VARCHAR name UNIQUE
            VARCHAR description
            VARCHAR source
            TIMESTAMP created_at
        }
        class usecases {
            UUID id PK
            VARCHAR app_name UNIQUE
            UUID created_by FK
            TIMESTAMP created_at
            TIMESTAMP updated_at
        }
        class tools {
            UUID id PK
            VARCHAR name UNIQUE
            VARCHAR tool_name UNIQUE
            VARCHAR description
            BOOLEAN is_active
        }
    }

    namespace DSQL_Junction {
        class user_groups {
            UUID user_id FK
            UUID group_id FK
            VARCHAR source
            TIMESTAMP synced_at
        }
        class user_usecases {
            UUID user_id FK
            UUID usecase_id FK
            VARCHAR permission
        }
        class group_usecases {
            UUID group_id FK
            UUID usecase_id FK
            VARCHAR permission
        }
        class user_tools {
            UUID user_id FK
            UUID tool_id FK
        }
        class group_tools {
            UUID group_id FK
            UUID tool_id FK
        }
        class usecase_tools {
            UUID usecase_id FK
            UUID tool_id FK
        }
    }

    namespace DynamoDB {
        class ImagesTable {
            String id PK
            String filename
            String s3_key
            String converted_s3_key
            String upload_time
            String status
            String app_name GSI
            String uploaded_by GSI
            Map ocr_result
            Map extracted_info
            Map extraction_mapping
            String extraction_status
            Boolean verification_completed
            String verified_by
            String verification_completed_at
            String parent_document_id
            String page_processing_mode
            Number total_pages
            Number page_number
            String sync_source_path
            String job_id
        }
        class JobsTable {
            <<Agent Jobs Only>>
            String id PK
            String image_id
            String job_type
            String status
            List suggestions
            String error
            String created_at
            String updated_at
            String completed_at
        }
        class SchemasTable {
            <<Usecase Master>>
            String schema_type PK
            String name PK
            String display_name
            String description
            List fields
            Map input_methods
            String custom_prompt
            String created_at
            String updated_at
        }
        class UserPreferencesTable {
            String user_id PK
            String sk PK "star#app_name"
        }
        class AgentToolsTable {
            String tool_name PK
            String description
        }
        class CustomersTable {
            <<Agent Demo Only>>
            String customer_id PK
            String customer_name GSI
        }
    }

    AmazonCognito --> users : sub = cognito_sub
    users --> usecases : id = created_by
    users --> user_groups : id = user_id
    groups --> user_groups : id = group_id
    users --> user_usecases : id = user_id
    usecases --> user_usecases : id = usecase_id
    groups --> group_usecases : id = group_id
    usecases --> group_usecases : id = usecase_id
    users --> user_tools : id = user_id
    tools --> user_tools : id = tool_id
    groups --> group_tools : id = group_id
    tools --> group_tools : id = tool_id
    usecases --> usecase_tools : id = usecase_id
    tools --> usecase_tools : id = tool_id
    users ..> ImagesTable : cognito_sub = uploaded_by
    users ..> UserPreferencesTable : cognito_sub = user_id
    SchemasTable ..> usecases : name = app_name
    SchemasTable ..> ImagesTable : name = app_name
    ImagesTable ..> JobsTable : id = image_id

    style AmazonCognito fill:#FADBD8,stroke:#E74C3C,color:#2C3E50
    style users fill:#D6EAF8,stroke:#2E86C1,color:#1B2631
    style groups fill:#D5F5E3,stroke:#27AE60,color:#1B2631
    style usecases fill:#E8DAEF,stroke:#8E44AD,color:#1B2631
    style tools fill:#FDEBD0,stroke:#E67E22,color:#1B2631
    style user_groups fill:#F2F3F4,stroke:#AEB6BF,color:#2C3E50
    style user_usecases fill:#F2F3F4,stroke:#AEB6BF,color:#2C3E50
    style group_usecases fill:#F2F3F4,stroke:#AEB6BF,color:#2C3E50
    style user_tools fill:#F2F3F4,stroke:#AEB6BF,color:#2C3E50
    style group_tools fill:#F2F3F4,stroke:#AEB6BF,color:#2C3E50
    style usecase_tools fill:#F2F3F4,stroke:#AEB6BF,color:#2C3E50
    style ImagesTable fill:#FEF9E7,stroke:#D4AC0D,color:#1B2631
    style JobsTable fill:#FEF9E7,stroke:#D4AC0D,color:#1B2631
    style SchemasTable fill:#FEF9E7,stroke:#D4AC0D,color:#1B2631
    style UserPreferencesTable fill:#FEF9E7,stroke:#D4AC0D,color:#1B2631
    style AgentToolsTable fill:#F2F3F4,stroke:#AEB6BF,color:#7F8C8D
```

### 権限解決の流れ

```
1. users.role を確認（admin なら全権限、author なら作成可能、reader なら共有のみ）
2. user_usecases で直接権限を確認
3. user_groups → group_usecases で間接権限を確認
4. 2 と 3 の最大権限を採用（owner > editor > viewer）
```

### Aurora DSQL の制約と設計上の対応

DSQL の制約により、本プロジェクトで明示的に設計を変えている箇所:

| 制約 | 設計への影響 |
|---|---|
| PK は連番（SERIAL）非推奨。UUID 推奨。PK は後から変更不可 | DSQL 側は全テーブルで `gen_random_uuid()` を PK に採用。DynamoDB 側は既存の String UUID をそのまま維持し前方互換性を保持 |
| FK 制約なし（CASCADE DELETE も不可） | 参照整合性はアプリケーション層で管理。DSQL 内の削除は中間テーブル → usecases の順で明示的に DELETE（`delete_group`, `delete_usecase_by_app_name` 参照） |
| OCC（楽観的同時実行制御） | `clients/dsql.py` の `with_retry()` で SerializationFailure を最大3回リトライ（指数バックオフ）。書き込み系の全操作で使用 |
| DDL は 1文/トランザクション | `ddl.sql` で各 CREATE TABLE / CREATE INDEX ASYNC を個別トランザクションで実行。コメントで明記 |

> 参照: [Migrating from PostgreSQL to Aurora DSQL](https://docs.aws.amazon.com/aurora-dsql/latest/userguide/working-with-postgresql-compatibility-unsupported-features.html)

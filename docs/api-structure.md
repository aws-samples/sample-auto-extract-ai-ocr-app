# API Structure

FastAPI ベースの REST API。Lambda (Docker) + API Gateway でホスティング。

## レイヤー責務

### routers/ — プレゼンテーション層

HTTP リクエスト/レスポンスの入出口。以下に限定する:

- `Depends` ベースの認証・認可
- Pydantic による入力バリデーション（router 内完結のリクエストモデルは router 内で定義可）
- `get_cognito_sub(req)` の取得とサービスへの受け渡し
- サービス呼び出し（`Depends(get_xxx_service)` で DI）
- サービス層のカスタム例外（NotFoundError, LastOwnerError 等）を型で分岐して HTTPException に変換

書いてはいけないもの: ビジネスルール、データレベルのフィルタリング、Repository の直接呼び出し、cognito_sub → email 等のデータ加工

### services/ — アプリケーション層

ユースケース単位のオーケストレーション:

- 複数の Repository / Domain の組み合わせ
- ユーザーの role/権限に応じたデータフィルタリング
- ビジネスルールの適用（「auto グループは削除不可」等）

書いてはいけないもの: HTTP 固有の処理（HTTPException, StreamingResponse 等）。ドメイン例外（ValueError, PermissionError 等）を raise し、router 層で HTTPException に変換する。複数サービスで重複するロジックは utils/ に切り出す

### domains/ — ドメイン層

外部依存を持たない純粋なビジネスロジック。DB や AWS SDK に直接依存しない

### repositories/ — データアクセス層

データストア（DynamoDB, DSQL, S3）への読み書きを抽象化する。ビジネスロジックを含めない

### clients/ — AWS SDK クライアント

クライアント生成・グローバルインスタンス・外部 API ラッパー

### utils/ — 横断的ユーティリティ

画像処理、Bedrock レスポンスパース等の純粋関数。複数サービスで使うものをここに置き、1 サービス専用のものはそのサービス内に置く

### サービス DI パターン

全サービスは `main.py` で `app.state` に生成し、`dependencies/services.py` の `Depends` 関数経由で Router に注入。`workers/step_functions.py` は FastAPI コンテキスト外のため自前でインスタンス生成する

## 認証・認可

| デコレータ | 説明 |
|---|---|
| `Depends(require_auth)` | ログイン必須 |
| `Depends(RequireRole(min_role))` | 指定ロール以上を要求（admin, author, reader） |
| `Depends(RequirePermission(min_level))` | ユースケース単位の権限チェック（owner/editor/viewer）。パスパラメータ `app_name` を自動取得し、user_usecases + group_usecases から最大権限を判定。admin は全スキップ |

エンドポイントレベルの認可は router 層の `Depends` で制御。データレベルの認可（「このデータをこのユーザーに返していいか」）はサービス層で user_id/role を受け取って制御する

---

## サービスごとの構成図


### UploadService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/upload.py"]
            U1["POST /generate-presigned-url 📋viewer"]
            U5["GET /images 🔓"]
            U6["DELETE /images/:id 🔓"]
            U_etc["... 他 3 エンドポイント"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py"]
            svc_di["services.py"]
        end

        subgraph Service["services/upload_service.py"]
            US1["generate_presigned_url"]
            US2["handle_upload_complete"]
            US3["get_image_stream"]
            US4["generate_download_url"]
            US5["get_images_list"]
            US6["get_images_for_user"]
            US7["get_images_for_permitted_apps"]
            US8["delete_image"]
        end

        subgraph Repos["repositories/"]
            image_repo["image_repository"]
            schema_repo["schema_repository"]
            usecase_repo["usecase_repository"]
            user_repo["user_repository"]
        end

        subgraph Utils["utils/"]
            helpers["helpers.py（resize_image, decimal_to_float）"]
        end

        subgraph OtherSvc["他サービス"]
            pdf_svc["pdf_conversion_service（トップレベル import）"]
        end

        subgraph Clients["clients/"]
            s3_cl["aws.py（s3_client）"]
        end
    end

    subgraph External["外部サービス"]
        s3["S3"]
    end

    Router --> Deps
    Deps --> Service
    US1 --> image_repo
    US1 --> schema_repo
    US1 --> s3_cl
    s3_cl --> s3
    US2 --> image_repo
    US2 --> s3_cl
    US2 --> pdf_svc
    US2 --> helpers
    US6 --> image_repo
    US6 --> usecase_repo
    US6 -.->|"_enrich_uploaded_by_email"| user_repo
    US6 --> helpers

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
    style Utils fill:#f3f4f6,stroke:#6b7280
    style OtherSvc fill:#d1fae5,stroke:#10b981
    style Clients fill:#f3f4f6,stroke:#6b7280
    style External fill:#ede9fe,stroke:#8b5cf6
```

### OcrService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/ocr.py"]
            O1["POST /ocr/start"]
            O_etc["... 他 4 エンドポイント"]
        end

        subgraph Deps["dependencies/"]
            svc_di["services.py"]
        end

        subgraph Service["services/ocr_service.py"]
            OS1["start_step_functions_job"]
            OS2["start_step_functions_for_image"]
            OS3["get_ocr_result"]
            OS4["update_ocr_result"]
            OS5["get_endpoint_status"]
            OS6["process_image_ocr"]
        end

        subgraph Repos["repositories/"]
            image_repo["image_repository"]
        end

        subgraph Domain["domains/ocr_engine.py"]
            parse_ocr["parse_ocr_response（純粋関数）"]
        end

        subgraph Clients["clients/"]
            sfn_cl["aws.py（sfn_client）"]
            smr_cl["aws.py（sagemaker_runtime_client）"]
            s3_cl["aws.py（s3_client）"]
            sm_status["aws.py（get_inference_component_status）"]
            sm_wakeup["aws.py（trigger_endpoint_wakeup）"]
        end

        subgraph OtherSvc["他サービス"]
            parent_sync["pdf_conversion_service.sync_parent_status"]
        end
    end

    subgraph SF_Lambda["Step Functions Lambda"]
        sf_handler["workers/step_functions.py"]
    end

    subgraph External["外部サービス"]
        SF["Step Functions"]
        sagemaker["SageMaker"]
        s3["S3"]
    end

    O1 --> OS1
    Router --> Deps
    Deps --> Service
    OS1 --> sfn_cl
    sfn_cl -->|"sfn.start_execution"| SF
    SF --> sf_handler
    sf_handler --> OS6
    OS6 --> smr_cl
    smr_cl --> sagemaker
    OS6 --> parse_ocr
    OS6 --> image_repo
    OS6 --> s3_cl
    s3_cl --> s3
    OS6 --> parent_sync

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
    style Domain fill:#ffedd5,stroke:#f97316
    style Clients fill:#f3f4f6,stroke:#6b7280
    style OtherSvc fill:#d1fae5,stroke:#10b981
    style SF_Lambda fill:#fef3c7,stroke:#f59e0b
    style External fill:#ede9fe,stroke:#8b5cf6
```

### ExtractionService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/extraction.py"]
            E1["POST /ocr/extract/:id"]
            E5["POST /ocr/extract/verification/:id 🔓"]
            E_etc["... 他 3 エンドポイント"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py（verification のみ require_auth）"]
            svc_di["services.py"]
        end

        subgraph Service["services/extraction_service.py"]
            ES_extract["extract_information"]
            ES_result["get_extraction_result"]
            ES_start["start_extraction"]
            ES_status["get_extraction_status"]
            ES_update["update_extraction_result"]
            ES_verify["update_verification_status"]
        end

        subgraph Domain["domains/"]
            prompts["extraction_engine.py（プロンプト構築 + パース）"]
            fields["schema_fields.py"]
        end

        subgraph Repos["repositories/"]
            image_repo["image_repository"]
            schema_repo["schema_repository"]
        end

        subgraph Clients["clients/"]
            bedrock_cl["bedrock.py（call_bedrock / call_bedrock_with_retry）"]
            s3_cl["aws.py（s3_client）"]
        end

        subgraph Utils["utils/"]
            bedrock_util["bedrock.py（parse_converse_response, extract_json_from_response）"]
            helpers_util["helpers.py（decimal_to_float）"]
        end

        subgraph OtherSvc["他サービス"]
            parent_sync["pdf_conversion_service.sync_parent_status"]
        end
    end

    subgraph External["外部サービス"]
        bedrock["Bedrock Claude"]
        s3["S3"]
    end

    Router --> Deps
    Deps --> Service
    ES_extract --> prompts
    ES_extract --> bedrock_cl
    bedrock_cl --> bedrock
    ES_extract --> bedrock_util
    ES_extract --> image_repo
    ES_extract --> schema_repo
    ES_extract --> s3_cl
    s3_cl --> s3
    ES_extract --> parent_sync

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Domain fill:#ffedd5,stroke:#f97316
    style Repos fill:#fed7aa,stroke:#f97316
    style Clients fill:#f3f4f6,stroke:#6b7280
    style Utils fill:#f3f4f6,stroke:#6b7280
    style OtherSvc fill:#d1fae5,stroke:#10b981
    style External fill:#ede9fe,stroke:#8b5cf6
```

### SchemaService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/schema.py"]
            S1["POST /apps 👤author"]
            S_etc["... 他 9 エンドポイント"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py（require_auth / RequireRole / RequirePermission）"]
            svc_di["services.py"]
        end

        subgraph Service["services/schema_service.py"]
            SS1["get_apps_list"]
            SS2["get_app_details"]
            SS3["get_app_fields"]
            SS4["get_custom_prompt"]
            SS5["update_custom_prompt"]
            SS6["save_schema"]
            SS7["update_schema"]
            SS8["delete_app"]
            SS9["generate_schema_presigned_url"]
            SS10["generate_schema"]
        end

        subgraph Domain["domains/"]
            schema_gen["schema_generator.py（プロンプト構築 + パース）"]
            schema_fields["schema_fields.py"]
        end

        subgraph Repos["repositories/"]
            schema_repo["schema_repository"]
            usecase_repo["usecase_repository"]
            image_repo["image_repository"]
        end

        subgraph Clients["clients/"]
            bedrock_cl["bedrock.py（call_bedrock）"]
            s3_cl["aws.py（s3_client）"]
        end

        subgraph Utils["utils/"]
            bedrock_util["bedrock.py（parse_converse_response）"]
            pdf_util["pdf.py（pdf_page_to_jpeg）"]
        end
    end

    subgraph External["外部サービス"]
        s3["S3"]
        bedrock["Bedrock Claude"]
    end

    Router --> Deps
    Deps --> Service
    SS1 --> schema_repo
    SS1 --> usecase_repo
    SS6 --> schema_repo
    SS6 --> usecase_repo
    SS6 --> image_repo
    SS8 --> usecase_repo
    SS8 --> image_repo
    SS8 --> schema_repo
    SS10 --> pdf_util
    SS10 --> bedrock_cl
    SS10 --> bedrock_util
    SS10 --> schema_gen
    SS10 --> s3_cl
    bedrock_cl --> bedrock
    s3_cl --> s3

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Domain fill:#ffedd5,stroke:#f97316
    style Repos fill:#fed7aa,stroke:#f97316
    style Clients fill:#f3f4f6,stroke:#6b7280
    style Utils fill:#f3f4f6,stroke:#6b7280
    style External fill:#ede9fe,stroke:#8b5cf6
```

### SharingService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/sharing.py"]
            SH1["GET /apps/:name/sharing 📋viewer"]
            SH_etc["... 他 5 エンドポイント"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py（RequirePermission）"]
            svc_di["services.py"]
        end

        subgraph Service["services/sharing_service.py"]
            SHS1["get_sharing"]
            SHS2["add_user_sharing / remove_user_sharing"]
            SHS3["add_group_sharing / remove_group_sharing"]
            SHS6["share_with_all"]
        end

        subgraph Repos["repositories/"]
            usecase_repo["usecase_repository"]
            group_repo["group_repository"]
        end
    end

    Router --> Deps
    Deps --> Service
    SHS1 --> usecase_repo
    SHS2 --> usecase_repo
    SHS3 --> usecase_repo
    SHS6 --> usecase_repo
    SHS6 --> group_repo

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
```

### S3SyncService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/s3_sync.py"]
            SS1["POST /s3-sync/:name 📋viewer"]
            SS2["POST /s3-sync/:name/import 📋editor"]
            SS3["GET /s3-sync/:name/list 📋viewer"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py（RequirePermission / get_cognito_sub）"]
            svc_di["services.py"]
        end

        subgraph Service["services/s3_sync_service.py"]
            SSS1["sync_s3_files"]
            SSS2["import_s3_file"]
            SSS3["get_files_with_duplicate_check"]
            SSS4["check_existing_files"]
        end

        subgraph Repos["repositories/"]
            schema_repo["schema_repository"]
            image_repo["image_repository"]
        end

        subgraph OtherSvc["他サービス"]
            upload_svc["UploadService（コンストラクタ注入）"]
        end

        subgraph Clients["clients/"]
            s3_cl["aws.py（s3_client）"]
        end
    end

    subgraph External["外部サービス"]
        s3["S3"]
    end

    Router --> Deps
    Deps --> Service
    SSS1 --> schema_repo
    SSS1 --> s3_cl
    s3_cl --> s3
    SSS2 --> schema_repo
    SSS2 --> image_repo
    SSS2 --> s3_cl
    SSS2 --> upload_svc
    SSS3 --> SSS1
    SSS3 --> SSS4
    SSS4 --> image_repo

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
    style OtherSvc fill:#d1fae5,stroke:#10b981
    style Clients fill:#f3f4f6,stroke:#6b7280
    style External fill:#ede9fe,stroke:#8b5cf6
```

### AdminService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/admin.py 🛡️全て RequireRole admin"]
            A1["GET /admin/users, groups, usecases, tools, images"]
            A_etc["... 他 14 エンドポイント（CRUD + permissions）"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py（RequireRole）"]
            svc_di["services.py（get_admin_service）"]
        end

        subgraph Service["services/admin_service.py"]
            AS1["list_users"]
            AS2["update_user_role"]
            AS3["groups CRUD + members"]
            AS4["list_usecases / get_usecase_permissions"]
            AS5["tools CRUD + permissions"]
        end

        subgraph Repos["repositories/"]
            user_repo["user_repository"]
            group_repo["group_repository"]
            usecase_repo["usecase_repository"]
            tool_repo["tool_repository"]
        end

        subgraph OtherSvc["他サービス"]
            upload_svc["UploadService（router が直接 DI）"]
        end
    end

    Router --> Deps
    Deps --> Service
    A_images["GET /admin/images"] -->|"Depends（get_upload_service）"| upload_svc
    AS1 --> user_repo
    AS1 --> group_repo
    AS2 --> user_repo
    AS3 --> group_repo
    AS4 --> usecase_repo
    AS5 --> tool_repo

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
    style OtherSvc fill:#d1fae5,stroke:#10b981
```

### UserService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/user.py 🔑全て require_auth"]
            US1["GET /user/me"]
            US_etc["... 他 5 エンドポイント"]
        end

        subgraph Deps["dependencies/"]
            auth["auth.py（require_auth）"]
            svc_di["services.py"]
        end

        subgraph Service["services/user_service.py"]
            USS1["get_me"]
            USS2["update_display_name"]
            USS3["stars CRUD（get_stars, add_star, remove_star）"]
            USS6["search"]
        end

        subgraph Repos["repositories/"]
            user_repo["user_repository"]
            prefs_repo["user_preferences_repository"]
            group_repo["group_repository"]
        end
    end

    subgraph External["外部サービス"]
        ddb["DynamoDB"]
        dsql["DSQL"]
    end

    Router --> Deps
    Deps --> Service
    USS1 --> user_repo
    USS2 --> user_repo
    USS3 --> prefs_repo
    USS6 --> user_repo
    USS6 --> group_repo
    user_repo --> dsql
    prefs_repo --> ddb

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
    style External fill:#ede9fe,stroke:#8b5cf6
```

### AgentService

```mermaid
flowchart LR
    subgraph FastAPI["FastAPI Lambda"]
        subgraph Router["routers/agent.py（認証なし）"]
            AG1["GET /ocr/agent/tools"]
            AG2["POST /ocr/agent/:id"]
            AG3["GET /ocr/agent/status/:id"]
        end

        subgraph Deps["dependencies/"]
            svc_di["services.py"]
        end

        subgraph Service["services/agent_service.py"]
            AGS1["get_available_tools"]
            AGS2["start_agent_correction"]
            AGS3["get_agent_job_status"]
        end

        subgraph Repos["repositories/"]
            agent_tools_repo["agent_tools_repository"]
            image_repo["image_repository"]
            job_repo["job_repository"]
        end

        subgraph Clients["clients/"]
            agent_client["agent.py（AgentClient）"]
        end
    end

    subgraph External["外部サービス"]
        agentcore["AgentCore Runtime"]
        ddb["DynamoDB"]
    end

    Router --> Deps
    Deps --> Service
    AGS1 --> agent_tools_repo
    agent_tools_repo --> ddb
    AGS2 --> image_repo
    AGS2 --> job_repo
    AGS2 --> agent_client
    AGS3 --> job_repo
    agent_client --> agentcore

    style Router fill:#dbeafe,stroke:#3b82f6
    style Deps fill:#f3f4f6,stroke:#6b7280
    style Service fill:#d1fae5,stroke:#10b981
    style Repos fill:#fed7aa,stroke:#f97316
    style Clients fill:#f3f4f6,stroke:#6b7280
    style External fill:#ede9fe,stroke:#8b5cf6
```

### Step Functions Lambda

FastAPI Lambda とは別の Lambda。`workers/step_functions.py` がハンドラー実体。
Dockerfile.stepfunctions の CMD が直接 `app.workers.step_functions.process_image_handler` を参照。

```mermaid
flowchart TD
    subgraph SF_Lambda["Step Functions Lambda（Dockerfile.stepfunctions）"]
        subgraph Handler["workers/step_functions.py"]
            handler["process_image_handler"]
        end

        subgraph Services["services/（自前インスタンス化）"]
            OcrSvc["OcrService.process_image_ocr"]
            ExtSvc["ExtractionService.extract_information"]
        end

        subgraph Domains["domains/"]
            ocr_parse["ocr_engine.parse_ocr_response"]
            ext_parse["extraction_engine（プロンプト構築+パース）"]
        end

        subgraph Repos["repositories/"]
            image_repo["image_repository"]
            schema_repo["schema_repository"]
        end

        subgraph Clients["clients/"]
            smr_cl["aws.py（sagemaker_runtime_client）"]
            s3_cl["aws.py（s3_client）"]
            bedrock_cl["bedrock.py（call_bedrock / call_bedrock_with_retry）"]
            sm_status["aws.py（get_inference_component_status）"]
        end
    end

    subgraph External["外部サービス"]
        SF["Step Functions Map 最大並列5"]
        sagemaker["SageMaker OCR"]
        bedrock["Bedrock Claude"]
        s3["S3"]
        ddb["DynamoDB"]
    end

    SF -->|"画像ごと"| handler
    handler -->|"1. skip_ocrでなくENABLE_OCRが有効なら"| OcrSvc
    handler -->|"2. 常に実行"| ExtSvc
    OcrSvc --> smr_cl
    smr_cl --> sagemaker
    OcrSvc --> ocr_parse
    OcrSvc --> image_repo
    OcrSvc --> s3_cl
    s3_cl --> s3
    ExtSvc --> bedrock_cl
    bedrock_cl --> bedrock
    ExtSvc --> ext_parse
    ExtSvc --> image_repo
    ExtSvc --> schema_repo
    image_repo --> ddb

    style SF_Lambda fill:#fef3c7,stroke:#f59e0b
    style Handler fill:#fef3c7,stroke:#f59e0b
    style Services fill:#d1fae5,stroke:#10b981
    style Domains fill:#ffedd5,stroke:#f97316
    style Repos fill:#fed7aa,stroke:#f97316
    style Clients fill:#f3f4f6,stroke:#6b7280
    style External fill:#ede9fe,stroke:#8b5cf6
```

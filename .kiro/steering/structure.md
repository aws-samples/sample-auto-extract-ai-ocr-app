---
inclusion: always
---

# Project Structure

ディレクトリ構造そのものは `ls` や IDE で確認できるためここには記載しない。以下は**コードから自明でない配置ルール・命名規約・設計思想**のみを記載する。

## 命名規約

- CDK Construct: PascalCase(例: `OcrEndpoint`)
- Lambda ファイル: snake_case(例: `extraction_service.py`)
- React コンポーネント: PascalCase(例: `ExtractedInfoDisplay.tsx`)

### AWS リソース物理名

- 原則ケバブケース(例: `ocr-tool-gateway`)。ただし **AgentCore Runtime 名はスネークケース**
  (API 制約でハイフン不可・アンダースコアのみ。例: `ocr_agent_runtime`)、**SageMaker/S3 はハイフン**
  (アンダースコア不可)。リソース種別ごとの許容文字に従う。
- SageMaker Model 名は物理名を指定せず CDK 自動採番に委ねる(イメージ差分ビルド時の Replacement で
  AlreadyExists を避けるため)。
- **マルチ環境の衝突回避**: 固定物理名には env suffix を付ける。base/未指定は suffix 無し(既存名を
  維持)、dev/stg/prod のみ `-{env}`(Runtime はアンダースコアで `_{env}`)。共通ヘルパー
  `lib/utils/naming.ts` の `envSuffix()` を使う。物理名を指定せず CFN 自動採番に任せるリソース
  (DynamoDB/Lambda/S3/StepFunctions 等)は元々衝突しないので suffix 不要。
- DynamoDB GSI 名は PascalCase(例: `AppNameIndex`)。
- 既存の CDK construct id は互換性維持のため据え置く(変更は論理ID変更→リソース置換を招くため)。

## web/src/ フロントエンド構成方針

- `components/ui/` — テーマ非依存の汎用 UI 部品
- `components/shared/` — 複数ページで使う業務コンポーネント
- `pages/<feature>/` — ページ専用コンポーネントを同ディレクトリに配置
- `services/` — API 呼び出し層(機能別分割)
- `utils/` — 純粋関数のみ(API 呼び出しは services/ に置く)

コンポーネント配置ルール: 1 ページ専用 → `pages/<feature>/`、2 ページ以上で共有 → `components/shared/`、汎用 → `components/ui/`

## lambda/api/app/ レイヤー設計

依存方向:

```
routers → dependencies/ → services → domains
                                    → repositories → clients/(DB系: DynamoDB, DSQL)
                                    → clients/(外部API系: Bedrock, SageMaker, SFn, S3, AgentCore)
workers/ → services(FastAPI DI 不使用、自前インスタンス化)
```

外部 API 系の clients/ について: Clean Architecture では Gateway/Adapter 層を介した抽象化が推奨されるが、現時点ではプロジェクト規模に対してオーバーエンジニアリングとなるため、services/ から clients/ の薄いラッパーを直接呼び出す設計としている。

---
inclusion: always
---

# Project Structure

ディレクトリ構造そのものは `ls` や IDE で確認できるためここには記載しない。以下は**コードから自明でない配置ルール・命名規約・設計思想**のみを記載する。

## 命名規約

- Lambda ファイル: snake_case(例: `extraction_service.py`)
- React コンポーネント: PascalCase(例: `ExtractedInfoDisplay.tsx`)

### CDK の construct id（論理 ID）

- construct id は PascalCase。原則クラス名に揃える。

### AWS リソースの物理名

- **原則として物理名は指定しない**。CloudFormation の自動採番
  (`{stackName}-{論理ID}-{hash}`)に任せる。スタック名が環境ごとに分かれるため複数環境でも衝突しない。
- 物理名を明示するのは、自動採番が使えないリソースに限る(名前で一意参照が要る、
  アカウント/リージョンでグローバル一意になる等)。命名は各サービスの API 制約に従う
  (横断的なケバブ/スネーク統一ルールは設けない。サービスごとの許容文字が優先)。
- 物理名を明示するリソースには、複数環境の衝突を避けるため env suffix を付ける。
  共通ヘルパー `lib/utils/naming.ts` の `envSuffix()` を使い、base は suffix 無しで既存名を維持、
  dev/stg/prod のみ付与する。
- DynamoDB GSI 名は PascalCase(例: `AppNameIndex`、`CustomerNameIndex`)。

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

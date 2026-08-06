---
inclusion: always
---

# Project Structure

ディレクトリ構造そのものは `ls` や IDE で確認できるためここには記載しない。以下は**コードから自明でない配置ルール・命名規約・設計思想**のみを記載する。

## 命名規約

- Lambda ファイル: snake_case(例: `extraction_service.py`)
- React コンポーネント: PascalCase(例: `ExtractedInfoDisplay.tsx`)

### CDK

- construct id は PascalCase。原則クラス名に揃える。
- リソースの物理名は原則指定せず、CloudFormation の自動採番に任せる
  (`{stackName}-{論理ID}-{hash}`。スタック名が環境ごとに分かれるため複数環境でも衝突しない)。
- 物理名を明示するのは、自動採番が使えないリソースに限る(名前で一意参照が要る、
  アカウント/リージョンでグローバル一意になる等)。命名は各サービスの API 制約に従い、
  複数環境の衝突を避けるため env suffix を付ける
  (共通ヘルパー `lib/utils/naming.ts` の `envSuffix()`。base は付けず、dev/stg/prod のみ)。

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

## テストの配置

- API 側は `lambda/api/app/tests/<対象と同じレイヤー名>/`、CDK とフロントエンドはリポジトリルートの `test/`。
- ファイル名は `test_<対象関数名>.py` / `<対象>.test.ts`。
- conftest.py は置かない。import パス解決は各ファイル冒頭の `sys.path.insert` 4 行ヘッダで行う（既存ファイルと同じ形にする）。

# デプロイと設定ガイド

AutoExtract のデプロイ方法と、設定できる項目をまとめます。

## デプロイ手順

デプロイの際は、事前に Node.js と Docker のインストール、および AWS 認証情報の設定（`aws sso login` や `aws configure` など）が必要です。

まず、依存パッケージをインストールします。

```sh
npm ci
```

新規の AWS アカウントやリージョンで初めて CDK を使う場合は、CDK bootstrap を実行します。

```sh
cdk bootstrap
```

続いて、デプロイを実行します。

```sh
npm run cdk:deploy
```

デプロイ後に出力される `WebConstructCloudFrontURL` の URL にアクセスすると、Web サイトを開けます。管理画面を使うには、少なくとも 1 人の管理者ユーザーが必要です（後述の「初期ユーザーの投入」を参照してください）。

### デプロイの環境とリージョンの指定

`npm run cdk:deploy` は、対話型のデプロイラッパー `scripts/cdk.sh` を起動します。引数を付けずに実行すると、環境（base / dev / stg / prod）とリージョンを対話的に選択できます。

引数で直接指定することもできます。npm 経由で実行する場合は、引数の前に `--` を挟んでください。挟まないと `--region` などを npm 自身が横取りしてしまいます。

```sh
# 環境 dev、リージョン us-east-1 にデプロイ（-y で確認プロンプトを省略）
npm run cdk:deploy -- dev --region us-east-1 -y
```

`prod` 環境へのデプロイや、後述の削除の際は、確認プロンプトが表示されます。

### その他のコマンド

デプロイ以外にも、以下のコマンドが利用できます。

```sh
npm run cdk:synth   # CloudFormation テンプレートの生成
npm run cdk:diff    # デプロイ済みリソースとの差分確認
npm run cdk:destroy # リソースの削除
```

### 環境の切り替え

環境ごとにスタック名が分かれるため、複数の環境を同じ AWS アカウントに併存できます。環境を指定しない場合は `base` が使われます。

| 環境 | スタック名 |
|---|---|
| `base`（未指定） | `OcrAppStack` |
| `dev` | `OcrAppStack-dev` |
| `stg` | `OcrAppStack-stg` |
| `prod` | `OcrAppStack-prod` |

## パラメータ設定

アプリケーションのパラメータは `lib/parameters.ts` で管理しています。パラメータは 2 つの部分に分かれます。全環境共通のデフォルト値を `defaultParameters` に、環境ごとの差分を `envOverrides` に記述します。`envOverrides` に書いた値だけが `defaultParameters` を上書きします。

```typescript
// 全環境共通のデフォルト値
const defaultParameters: AppParameters = {
  modelId: "us.anthropic.claude-sonnet-4-6",
  modelRegion: "us-east-1",
  ocrEngine: "paddle",
  // ...
};

// 環境ごとの差分のみ記述
const envOverrides: Record<string, Partial<AppParameters>> = {
  prod: {
    sagemakerZeroScale: false,
    selfSignUpEnabled: false,
    waf: { enabled: true, allowedCountryCodes: ["JP"] },
  },
};
```

### パラメータ一覧

| パラメータ | 型 | デフォルト | 説明 |
|---|---|---|---|
| `modelId` | string | `us.anthropic.claude-sonnet-4-6` | 情報抽出とスキーマ生成に使う Bedrock モデル ID |
| `modelRegion` | string | `us-east-1` | Bedrock を呼び出すリージョン |
| `enableOcr` | boolean | `true` | OCR を有効化するか |
| `ocrEngine` | `"paddle"` \| `"deepseek"` \| `"yomitoku-mp"` | `paddle` | 使用する OCR エンジン（後述） |
| `sagemakerZeroScale` | boolean | `true` | アクセスがないとき OCR インスタンスを 0 台まで縮小するか |
| `sagemakerScaleInCooldownSeconds` | number | `3600` | 縮小するまでの待機時間（秒） |
| `marketplaceModelPackageArn` | string? | （未設定） | `yomitoku-mp` を使うときの Marketplace モデルパッケージ ARN |
| `enableAgentDemo` | boolean | `true` | エージェント検証のデモ用ユースケースとツールを自動登録するか（[ARCHITECTURE.md](ARCHITECTURE.md) を参照） |
| `selfSignUpEnabled` | boolean | `true` | Cognito のセルフサインアップを許可するか |
| `allowedSignUpEmailDomains` | string[] | `[]`（制限なし） | サインアップを許可するメールドメイン |
| `waf` | WafOptions | `{ enabled: false }` | CloudFront への WAF 設定（後述） |

`waf` は以下のフィールドを持ちます。

| フィールド | 型 | 説明 |
|---|---|---|
| `enabled` | boolean | `true` で AWS Managed Rules（Common Rule Set）を適用 |
| `allowedIpV4AddressRanges` | string[]? | 許可する IPv4 レンジ（指定時のみ制限） |
| `allowedIpV6AddressRanges` | string[]? | 許可する IPv6 レンジ（指定時のみ制限） |
| `allowedCountryCodes` | string[]? | 許可する国コード（指定時のみ地理制限） |

### 使用するモデルの変更

`modelId` と `modelRegion` を変更します。モデル ID は [Amazon Bedrock でサポートされている基盤モデル](https://docs.aws.amazon.com/ja_jp/bedrock/latest/userguide/models-supported.html) を参照してください。使用するモデルは、事前にモデルアクセスを有効化しておく必要があります。

### OCR エンジンの変更

`ocrEngine` で、使用する OCR エンジンを選択します。

- `paddle`: PaddleOCR（デフォルト）。自前のコンテナで動作し、ゼロスケーリングに対応します。
- `deepseek`: DeepSeek OCR。自前のコンテナで動作し、ゼロスケーリングに対応します。
- `yomitoku-mp`: Yomitoku Pro（AWS Marketplace 版）。高精度な日本語 OCR です。

#### Yomitoku Pro（AWS Marketplace 版）

[AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-64qkuwrqi4lhi) でサブスクライブした後、`lib/parameters.ts` に以下を設定します。

```typescript
ocrEngine: "yomitoku-mp",
marketplaceModelPackageArn: "arn:aws:sagemaker:<region>:<account>:model-package/yomitoku-pro-document-analyzer-...",
sagemakerZeroScale: false, // Marketplace 版はゼロスケーリングに対応しません
```

Marketplace 版はゼロスケーリングに対応しないため、エンドポイントは常時起動となります。利用中は Marketplace のソフトウェア使用料と SageMaker のインスタンス料金が継続的にかかります（料金は [AWS Marketplace の製品ページ](https://aws.amazon.com/marketplace/pp/prodview-64qkuwrqi4lhi) を参照してください）。

### OCR インスタンスのゼロスケーリング

OCR に使う GPU インスタンスのコストを抑えるため、一定時間アクセスがないと、インスタンス数を自動的に 0 まで縮小します。再び OCR が必要になった際は、インスタンスの起動に 10 分ほどかかる点にご注意ください。

- `sagemakerZeroScale`: ゼロスケーリングの有効と無効（デフォルトは有効）
- `sagemakerScaleInCooldownSeconds`: 縮小するまでの待機時間（秒。デフォルトは `3600`、つまり 1 時間）

## 初期ユーザーの投入

デプロイ直後は Cognito にユーザーが 0 人の状態です。管理画面を使うには、少なくとも 1 人の `admin` ロールを持ったユーザーが必要です。初期ユーザー（admin / author / reader）をまとめて登録するには、`scripts/seed-users.ts` を使います。

### CSV の準備

`users.example.csv` をコピーして `users.csv` を作成し、投入したいユーザーを記述します（`users.csv` は仮パスワードを含むため `.gitignore` 済み）。

```csv
email,tempPassword,role,groups,displayName
admin@example.com,TempPass123!,admin,admin,Admin User
alice@example.com,TempPass123!,author,team-a,Alice
bob@example.com,TempPass123!,author,team-a|team-b,Bob
carol@example.com,TempPass123!,reader,,Carol
```

| カラム | 説明 |
|---|---|
| `email` | ユーザーの email（必須） |
| `tempPassword` | 仮パスワード（必須、Cognito の PasswordPolicy を満たすこと） |
| `role` | `admin` / `author` / `reader` のいずれか |
| `groups` | 所属グループ名（パイプ `\|` 区切り、空可）。CSV に登場するグループは自動作成される |
| `displayName` | 表示名（空可。空なら email を利用） |

### 実行方法

**方法 1: deploy コマンドと同時に投入する（推奨）**

`cdk.sh` の `--seed-users` フラグに CSV パスを渡すと、deploy 完了後に自動で seed-users が実行されます。UserPoolId / DSQL エンドポイントは CFN Outputs から自動取得されます。

```sh
npm run cdk:deploy -- base --region us-east-1 --seed-users users.csv
```

**方法 2: deploy 後に単独で実行する**

既にデプロイ済みの環境にユーザーを追加・更新したい場合は、`seed-users` スクリプトを直接呼び出します。**引数を省略すると対話モード**で走り、アカウント確認・リージョン選択・CloudFormation スタック選択・CSV プレビューを経て実行確認に進むので、投入対象を目視で確認できます。

```sh
# 対話モード (推奨): 何も引数を渡さない
npm run settings:users
```

対話モードでは以下の順で選択・確認が行われます。

1. `aws sts get-caller-identity` で現在のアカウント / ARN を表示
2. 主要リージョン (`ap-northeast-1` / `us-east-1`) を横断して `OcrAppStack` で始まるスタックを検索
    - 見つからなければリージョン選択メニューにフォールバック
    - 複数見つかれば環境 (base / dev / stg / prod) とリージョンを表示して選択、単一なら自動選択
3. スタックの Outputs から `UserPoolId` / DSQL `ClusterEndpoint` を自動取得
4. CSV ファイルを指定 (カレントに `users.csv` があれば Enter で採用)
5. CSV の先頭 2 行と検出グループをプレビュー表示、最終確認 `[y/N]` で実行

引数を明示指定して非対話モードで叩くこともできます (自動化やスクリプト経由のとき)。

```sh
npm run settings:users -- \
  --csv users.csv \
  --user-pool-id <AuthUserPoolId の値> \
  --dsql-endpoint <DsqlClusterEndpoint の値> \
  --region <リージョン>
```

`AuthUserPoolId` と `DsqlClusterEndpoint` は `npm run cdk:deploy` の出力または AWS コンソールで確認できます。一部の引数だけ指定した場合は、不足分だけが対話で聞かれます。

**ドライラン**

実際の書き込みを行わず、何が作成・更新されるかだけを確認するには `--dry-run` を付けます。

```sh
npm run settings:users -- --csv users.csv --user-pool-id ... --dsql-endpoint ... --region ... --dry-run
```

### 冪等性と既存ユーザーの扱い

seed-users は同じ CSV を何度実行しても安全に動作します。

- 新規ユーザー: Cognito に仮 PW 付きで作成し、DSQL に `role` と `groups` を反映します
- 既存ユーザー（Cognito に登録済み）: **Cognito 側の PW は上書きしません**。DSQL の `role` と `groups` のみ更新します
- 未ログインユーザー（Cognito にはいるが DSQL にはまだ登録されていない）: Cognito から `sub` を取得して DSQL に事前登録します。ユーザーが初回ログインしても `role` は保持されます

グループは CSV に登場する分だけ自動で作成されます。CSV から削除しても既存の紐付けは剥がれません（誤操作防止）。所属を外したい場合は管理画面から明示的に操作してください。

### 招待メールの送信

既定では Cognito の招待メールは送信されません（`MessageAction: SUPPRESS`）。仮 PW は CSV 経由で運用者から利用者に伝える運用を想定しています。招待メールを送りたい場合は `--send-invitation` を付けてください。

## ローカル開発

フロントエンドをローカルの開発サーバーで動かす場合は、デプロイ時の出力値を使って環境変数を設定します。

まず、`web/.env.sample` をコピーして `web/.env` を作成し、デプロイ時の出力値を各変数に設定します。

```properties
VITE_APP_USER_POOL_CLIENT_ID=      # AuthUserPoolClientId の値
VITE_APP_USER_POOL_ID=             # AuthUserPoolId の値
VITE_APP_REGION=                   # デプロイしたリージョン
VITE_API_BASE_URL=                 # ApiOcrApiEndpoint の値
VITE_ENABLE_OCR=true               # OCR 機能の有効化
VITE_SYNC_BUCKET_NAME=             # S3 同期バケット名
VITE_WEBSOCKET_URL=                # WebSocketEndpoint の値
```

続いて、開発サーバーを起動します。

```sh
npm run web:dev
```

ブラウザで `http://localhost:3000` を開くと、アプリケーションにアクセスできます。

## リソースの削除

以下のコマンドでリソースを削除できます。削除するとリソースとデータは完全に消去されるため、ご注意ください。

```sh
npm run cdk:destroy
```

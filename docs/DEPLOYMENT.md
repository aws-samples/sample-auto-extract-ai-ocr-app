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

デプロイ後に出力される `WebConstructCloudFrontURL` の URL にアクセスすると、Web サイトを開けます。管理画面を使うには、最初のユーザーを管理者に昇格させる必要があります（後述の「初期管理者の設定」を参照してください）。

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

Marketplace 版はゼロスケーリングに対応しないため、エンドポイントは常時起動（ml.g5.xlarge で約 $1.7/h）となります。

> [!Note]
> OSS 版の Yomitoku を利用した実装例は[こちら](https://github.com/gteu/sample-auto-extract-ai-ocr-app)を参照してください。ただし OSS 版は CC BY-NC-SA 4.0 ライセンスが適用され、商用利用が制限される点にご注意ください（[詳細](https://github.com/kotaro-kinoshita/yomitoku?tab=readme-ov-file#license)）。

### OCR インスタンスのゼロスケーリング

OCR に使う GPU インスタンスのコストを抑えるため、一定時間アクセスがないと、インスタンス数を自動的に 0 まで縮小します。再び OCR が必要になった際は、インスタンスの起動に 10 分ほどかかる点にご注意ください。

- `sagemakerZeroScale`: ゼロスケーリングの有効と無効（デフォルトは有効）
- `sagemakerScaleInCooldownSeconds`: 縮小するまでの待機時間（秒。デフォルトは `3600`、つまり 1 時間）

## 初期管理者の設定

デプロイ直後は、すべてのユーザーが一般権限（`author`）です。管理画面にアクセスするには、最初のユーザーを `admin` に昇格させる必要があります。権限の仕組みは [ARCHITECTURE.md](ARCHITECTURE.md) を参照してください。

1. CloudFront URL にアクセスして、サインアップとログインをします
2. CDK 出力の `DsqlClusterEndpoint` を確認します
3. 以下のコマンドで admin 権限を付与します

```sh
DSQL_ENDPOINT=<DsqlClusterEndpoint の値> DSQL_REGION=<リージョン> \
  npm run init-admin -- --email <サインアップしたメールアドレス>
```

実行例:

```sh
DSQL_ENDPOINT=xxxxx.dsql.us-east-1.on.aws DSQL_REGION=us-east-1 \
  npm run init-admin -- --email user@example.com
```

成功すると `Admin granted: {"id":"...","email":"...","role":"admin"}` と表示されます。ブラウザをリロードすると、管理画面にアクセスできるようになります。

## ローカル開発

フロントエンドをローカルの開発サーバーで動かす場合は、デプロイ時の出力値を使って環境変数を設定します。

まず、`web/.env.sample` をコピーして `web/.env` を作成し、デプロイ時の出力値を各変数に設定します。

```properties
VITE_APP_USER_POOL_CLIENT_ID=      # AuthUserPoolClientId の値
VITE_APP_USER_POOL_ID=             # AuthUserPoolId の値
VITE_APP_REGION=                   # デプロイしたリージョン
VITE_API_BASE_URL=                 # ApiOcrApiEndpoint の値
VITE_ENABLE_OCR=true               # OCR 機能の有効化
VITE_ENABLE_AGENT=true             # エージェント検証機能の有効化
VITE_SYNC_BUCKET_NAME=             # S3 同期バケット名
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

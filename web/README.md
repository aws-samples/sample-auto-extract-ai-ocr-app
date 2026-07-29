# AutoExtract - React フロントエンド

このディレクトリには、帳票 OCR・情報抽出システムの React フロントエンドアプリケーションが含まれています。

## 技術スタック

- React 18
- TypeScript
- Tailwind CSS
- AWS Amplify (認証)
- React Router
- Vite (ビルドツール)

## 開発環境のセットアップ

1. 依存パッケージのインストール:

```bash
npm install
```

2. 環境変数の設定:

`.env.sample` ファイルを `.env` にコピーし、必要な環境変数を設定します。

```bash
cp .env.sample .env
```

3. 開発サーバーの起動:

```bash
npm run dev
```

## ビルド

本番用ビルドを作成するには:

```bash
npm run build
```

ビルド成果物は `dist` ディレクトリに生成されます。

## プロジェクト構造

```
web/
├── public/           # 静的アセット
├── src/              # ソースコード
│   ├── assets/       # 画像などのアセット
│   ├── components/   # 再利用可能なコンポーネント
│   ├── pages/        # ページコンポーネント
│   ├── router/       # ルーティング設定
│   ├── types/        # TypeScript 型定義
│   ├── utils/        # ユーティリティ関数
│   ├── App.tsx       # メインアプリケーションコンポーネント
│   ├── main.tsx      # エントリーポイント
│   └── index.css     # グローバルスタイル
├── index.html        # HTMLテンプレート
├── package.json      # 依存関係とスクリプト
├── tsconfig.json     # TypeScript設定
└── vite.config.ts    # Vite設定
```

## 主要機能

- AWS Amplify を使用したユーザー認証
- 帳票ファイルのアップロード
- OCR 処理結果の表示（テキスト抽出、構造化データ、要約）
- ユースケース（処理タイプ）ごとの抽出スキーマ管理

# AutoExtract

輸送伝票・請求書などの PDF や画像をアップロードし、OCR 処理によってテキスト抽出、情報の構造化を行うシステムです。
AWS CDK を使用したフルスタックアプリケーションで、REST API バックエンド、React フロントエンド、および各種 AWS サービスを統合しています。

## ディレクトリ構造

```
.
├── README.md
├── bin
│   └── ocr-app.ts               // CDK のエントリーポイント
├── cdk.context.json
├── cdk.json
├── cdk.out
├── container                    // SageMaker エンドポイント用のコンテナ定義
│   ├── Dockerfile
│   ├── inference.py
│   └── requirements.txt
├── default-schemas              // デフォルトのアプリケーションスキーマ
│   └── apps-schemas.json
├── docs                         // ドキュメント
│   ├── DEPLOYMENT.md
│   ├── IMPLEMENTATION.md
│   └── OPERATIONS.md
├── lambda
│   ├── api                      // API Lambda 関数
│   │   ├── app                  // FastAPI アプリケーション
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── init-schemas             // DB 初期化用 Lambda
├── lib                          // CDK スタック定義
│   ├── constructs
│   └── ocr-app-stack.ts
├── package.json
└── web                          // React フロントエンド
    ├── dist
    ├── src
    │   ├── components
    │   ├── pages
    │   ├── types
    │   └── utils
    └── vite.config.ts
```

## 目次

- [デプロイ手順](docs/DEPLOYMENT.md)
- [アーキテクチャ・実装解説](docs/IMPLEMENTATION.md)

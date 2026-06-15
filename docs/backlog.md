# Backlog

## API パスのリファクタ

**優先度**: 中  
**きっかけ**: `/ocr/start/{app_name}` と `/ocr/start/{image_id}` が同パスパターンで衝突し、skip_ocr が効かないバグが発生。暫定で `/start/image/{image_id}` に分離して対処済み。

**現状の問題**:
- `/ocr` prefix に OCR・抽出・Agent が全部混在
- path parameter の型が曖昧（app_name vs image_id が同じ位置）
- エンドポイント命名が動詞ベースと名詞ベースが混在

**リファクタ案**:
```
# リソース中心の設計
GET/POST /images/{id}/ocr          ← OCR 結果取得・再実行
GET/POST /images/{id}/extraction   ← 抽出結果・再抽出
GET/POST /images/{id}/agent        ← Agent 検証結果・実行
PATCH    /images/{id}              ← verification_completed 等の更新

# バッチ系
POST /apps/{name}/jobs             ← 全 pending 画像のバッチ処理開始
GET  /jobs/{id}                    ← ジョブステータス

# ユーティリティ
GET /ocr/endpoint-status           ← SageMaker エンドポイント状態
```

**影響範囲**:
- フロントエンド全 API 呼び出し箇所
- Step Functions 内 Lambda (agent_kick 等) の API 呼び出し（あれば）
- ocrApi.ts / api.ts のサービス層

**注意**: フロントとバックエンドを同時に切り替える必要あり。段階的に移行するなら旧パスを deprecated alias として残す手もある。

---

## Cognito PostAuth Lambda コールドスタート対策

**優先度**: 中  
**症状**: 初回ログイン時に `CodeArtifactUserPendingException` エラー（Cognito の 5s ハードリミットを超過）  
**原因**: PostAuth Lambda が DockerImage 関数で Init 2.3s + Duration 4.0s = 6.3s かかる

**対策案**:
- Lambda を zip パッケージ化（Docker → zip で Init 時間削減）
- SnapStart 対応（Python は未対応なので Node.js に書き換え）
- Provisioned Concurrency で常時 warm 維持
- PostAuth 処理を非同期化（SQS 経由で別 Lambda に委譲）

---
inclusion: always
---

# Coding Rules

## バックエンド (Python)

- **型ヒントを使用**: 全ての関数引数・返り値に型注釈を付ける。
- **ドメイン駆動設計に準拠**: domains / services / repositories / routers のレイヤー構造で実装する。
- **レイヤー方向を守る**: 上位層 → 下位層のみ呼び出し可。逆方向禁止。
- **DB 操作は Repository 経由のみ**: services/workers/routers から DynamoDB Table を直接操作しない。
- **Router は薄く**: ルーティング・バリデーション・レスポンス変換のみ。ビジネスロジックは services/ へ。
- **Domains は純粋関数**: 外部 I/O (DB, S3, API) を行わない。
- **スキーマは schemas/ に集約**: Pydantic モデルを router 内にインライン定義しない。
- **import はモジュール先頭**: 循環参照を避ける場合を除き、関数内での遅延 import は禁止。
- **変更後は pyflakes 実行**: 変更したファイルに対して `python -m pyflakes` を実行し、未使用 import や構文エラーがないことを確認する。
- **enum 的な値は `StrEnum` にする**: status / mode / type など取りうる値が決まった文字列は、リテラルを散在させず `StrEnum` に集約し、write 前に検証する（`XxxStatus(value)` が無効値で ValueError）。
- **エラーは domain 例外で表す**: services / repositories は `exceptions.py` の `AppError` 系（`NotFoundError` 等）を raise する。`HTTPException` は routers / dependencies のみ。HTTP 変換とレスポンス整形（`{detail, code}`）は `errors.py` の例外ハンドラに一元化し、各所で try/except して 500 に包み直さない。

## フロントエンド (React TypeScript)

- **関数コンポーネント + Hooks パターンを使用**: class component は使わない。
- **型は camelCase 統一**: API レスポンスの snake_case はサービス層で変換。
- **巨大コンポーネントは分割**: 複雑なステート管理・ポーリングはカスタムフックに切り出す。
- **enum 的な値は union 型で締める**: status / type 等は `string` でなく取りうる値の union にする。
- **API エラー表示は `err.userMessage` を使う**: 各所で `detail` / `err.message` を手整形しない。表示用メッセージは axios interceptor が付与する。

## CDK (TypeScript)

- **Construct パターンで機能単位に分割**: 認証・DB・API・OCR エンドポイント等を個別 Construct として切り出す。

## コメント（共通）

- **コメントは WHY を書く**: 何をするか（HOW/WHAT）の逐語説明でなく、なぜそうするかを書く。変更履歴・before/after は書かない（公開リポジトリ）。

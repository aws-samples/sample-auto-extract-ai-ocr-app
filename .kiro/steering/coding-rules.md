---
inclusion: always
---

# Coding Rules

## バックエンド (Python)

- **レイヤー方向を守る**: 上位層 → 下位層のみ呼び出し可。逆方向禁止。
- **DB 操作は Repository 経由のみ**: services/workers/routers から DynamoDB Table を直接操作しない。
- **Router は薄く**: ルーティング・バリデーション・レスポンス変換のみ。ビジネスロジックは services/ へ。
- **Domains は純粋関数**: 外部 I/O (DB, S3, API) を行わない。
- **スキーマは schemas/ に集約**: Pydantic モデルを router 内にインライン定義しない。
- **import はモジュール先頭**: 循環参照を避ける場合を除き、関数内での遅延 import は禁止。
- **変更後は pyflakes 実行**: 変更したファイルに対して `python -m pyflakes` を実行し、未使用 import や構文エラーがないことを確認する。

## フロントエンド (React TypeScript)

- **型は camelCase 統一**: API レスポンスの snake_case はサービス層で変換。
- **巨大コンポーネントは分割**: 複雑なステート管理・ポーリングはカスタムフックに切り出す。

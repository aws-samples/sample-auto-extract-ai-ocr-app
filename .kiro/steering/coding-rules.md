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
- **エラーは domain 例外で表す**: `exceptions.py` の `AppError` 系（`NotFoundError` 等）を raise する。例外クラスは各レイヤーに定義せず `exceptions.py` に集約する。`HTTPException` は routers / dependencies のみ。HTTP 変換とレスポンス整形（`{detail, code}`）は `errors.py` の例外ハンドラに一元化し、各所で try/except して 500 に包み直さない。
- **調査用の情報を含む 5xx は `code` を空にする**: `errors.py` は `code` を持たない 5xx のメッセージを内部情報として隠す。利用者に見せられない詳細（応答の生の形など）を持つ例外は `code` を定義しない。

## フロントエンド (React TypeScript)

- **関数コンポーネント + Hooks パターンを使用**: class component は使わない。
- **型は camelCase 統一**: API レスポンスの snake_case はサービス層で変換。
- **巨大コンポーネントは分割**: 複雑なステート管理・ポーリングはカスタムフックに切り出す。
- **enum 的な値は union 型で締める**: status / type 等は `string` でなく取りうる値の union にする。
- **API エラー表示は `err.userMessage` を使う**: 各所で `detail` / `err.message` を手整形しない。表示用メッセージは axios interceptor が付与する。
- **変更後は Web ビルドを実行**: フロントエンドを変更した場合、リポジトリルートで `npm run web:build` を実行して型チェックと Vite ビルドを確認する。ルートの `npm run build` は CDK 側の TypeScript しか検証しないため、フロントエンド検証の代用にしない。

## CDK (TypeScript)

- **Construct パターンで機能単位に分割**: 認証・DB・API・OCR エンドポイント等を個別 Construct として切り出す。

## テスト（共通）

- **挙動を変えたらテストを書く**: バグ修正・仕様変更では、その挙動を固定するテストを追加または更新する。
- **変更後は `npm test` を実行**: jest（CDK / フロント）と pytest（API）の両方が通ることを確認する。テスト実行方法は `docs/DEPLOYMENT.md` の「テストの実行」を参照。
- **想定挙動を docstring に書く**: 各テストファイルに「何を正しい挙動として想定しているか」を書く。実装が守っていない範囲までは書かない。

## コメント（共通）

- **コメントは WHY を書く**: 何をするか（HOW/WHAT）の逐語説明でなく、なぜそうするかを書く。変更履歴・before/after は書かない（公開リポジトリ）。

/**
 * 環境名（base / dev / stg / prod）をリソース物理名の suffix に変換する。
 *
 * base と未指定は suffix 無し（＝既存環境のリソース名を維持する）。
 * dev / stg / prod のみ `-dev` のような suffix を付け、同一アカウント・同一リージョンに
 * 複数環境を同時デプロイしても物理名が衝突しないようにする。
 *
 * separator でハイフン / アンダースコアを切り替える。AgentCore Runtime 名のように
 * ハイフンを許容せずアンダースコアのみのリソースでは `"_"` を渡す。
 */
export function envSuffix(envName?: string, separator = "-"): string {
  return envName && envName !== "base" ? `${separator}${envName}` : "";
}

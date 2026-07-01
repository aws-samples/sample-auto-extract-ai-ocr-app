#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────
#  CDK ラッパースクリプト
#
#  使い方:
#    ./scripts/cdk.sh <command> [<env>] [--env <env>] [--region <region>] [-y]
#    ./scripts/cdk.sh                                            # 完全対話型
#    npm run cdk:deploy                                          # 同上 (deploy 固定)
#    npm run cdk:deploy -- dev --region us-east-1 -y             # 引数渡し
#
#  command: deploy | destroy | synth | diff (位置引数)
#  env:     base | dev | stg | prod (位置引数 or --env フラグ)
#  --region <region>: AWS リージョン (例: ap-northeast-1, us-east-1)
#  -y / --yes:        ラッパー独自の確認プロンプトをスキップ
#                     (CDK 本体の承認プロンプトは別途残る)
#
#  注: --env=dev / --region=us-east-1 のイコール形式も可
#  注: npm 経由の場合は必ず "--" を入れないと npm に引数を奪われる
#      OK: npm run cdk:deploy -- dev --region us-east-1
#      NG: npm run cdk:deploy --region us-east-1   (npm が --region を奪う)
# ────────────────────────────────────────────────────────────────────
set -euo pipefail

readonly VALID_COMMANDS=("deploy" "destroy" "synth" "diff")
readonly VALID_ENVS=("base" "dev" "stg" "prod")
readonly COMMON_REGIONS=("ap-northeast-1" "us-east-1" "us-west-2" "ap-northeast-3" "eu-west-1")

# ── 色定義 ─────────────────────────────────────────────────
if [[ -t 1 ]]; then
  readonly CYAN=$'\033[0;36m'
  readonly GREEN=$'\033[0;32m'
  readonly YELLOW=$'\033[0;33m'
  readonly RED=$'\033[0;31m'
  readonly BOLD=$'\033[1m'
  readonly RESET=$'\033[0m'
else
  readonly CYAN="" GREEN="" YELLOW="" RED="" BOLD="" RESET=""
fi

die() {
  echo "${RED}ERROR:${RESET} $*" >&2
  exit 1
}

contains() {
  local target="$1"; shift
  local item
  for item in "$@"; do
    [[ "$item" == "$target" ]] && return 0
  done
  return 1
}

show_help() {
  cat <<'USAGE'
CDK ラッパースクリプト

使い方:
  ./scripts/cdk.sh <command> [<env>] [--env <env>] [--region <region>] [-y]
  ./scripts/cdk.sh                                            # 完全対話型
  npm run cdk:deploy                                          # 同上 (deploy 固定)
  npm run cdk:deploy -- dev --region us-east-1 -y             # 引数渡し

引数:
  command           deploy | destroy | synth | diff (位置引数)
  env               base | dev | stg | prod (位置引数 or --env フラグ)
  --env <env>       env を flag で指定する場合
  --region <region> AWS リージョン (例: ap-northeast-1, us-east-1)
  -y, --yes         ラッパー独自の確認プロンプトをスキップ
  -h, --help        このヘルプを表示

例:
  ./scripts/cdk.sh deploy dev --region us-east-1
  ./scripts/cdk.sh deploy --env=base --region=ap-northeast-1 -y
  ./scripts/cdk.sh destroy prod --region us-east-1

npm 経由の注意:
  必ず "--" を入れないと npm に引数を奪われる:
    OK: npm run cdk:deploy -- dev --region us-east-1
    NG: npm run cdk:deploy --region us-east-1   (npm が --region を奪う)
USAGE
}

# ── 引数パース (フラグ + 位置引数) ────────────────────────
SKIP_CONFIRM=false
CDK_COMMAND=""
ENV_NAME=""
REGION=""

require_value() {
  # フラグに続く値があるか確認
  [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || die "$1 には値が必要です"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes)        SKIP_CONFIRM=true; shift ;;
    -h|--help)       show_help; exit 0 ;;
    --env)           require_value "$@"; ENV_NAME="$2"; shift 2 ;;
    --env=*)         ENV_NAME="${1#--env=}"; shift ;;
    --region)        require_value "$@"; REGION="$2"; shift 2 ;;
    --region=*)      REGION="${1#--region=}"; shift ;;
    --*)             die "未知のフラグ: $1" ;;
    *)
      # 位置引数を内容で判定:
      # - 1番目 → command
      # - VALID_ENVS に該当 → env
      # - region 形式 (xx-xxxx-N) に該当 → region
      if [[ -z "$CDK_COMMAND" ]]; then
        CDK_COMMAND="$1"
      elif [[ -z "$ENV_NAME" ]] && contains "$1" "${VALID_ENVS[@]}"; then
        ENV_NAME="$1"
      elif [[ -z "$REGION" ]] && [[ "$1" =~ ^[a-z]{2}-[a-z]+-[0-9]+$ ]]; then
        REGION="$1"
      else
        die "未知の引数: $1 (env / region として認識できません)"
      fi
      shift
      ;;
  esac
done

# ── npm run 経由で渡された --flag を補完 ─────────────────
# npm は "--" を挟まずに渡された --foo bar を npm 自身のオプションとして
# 解釈し、npm_config_foo=bar という環境変数に格納する。
# script に直接届かないが、ここで拾えば npm run cdk:deploy --region X も動く。
[[ -z "$ENV_NAME"  && -n "${npm_config_env:-}"    ]] && ENV_NAME="$npm_config_env"
[[ -z "$REGION"    && -n "${npm_config_region:-}" ]] && REGION="$npm_config_region"
[[ "$SKIP_CONFIRM" == "false" && -n "${npm_config_yes:-}" ]] && SKIP_CONFIRM=true

# ── 対話型: メニュー選択 ─────────────────────────────────
select_from_menu() {
  local prompt="$1"; shift
  local options=("$@")
  local i=1
  echo "${BOLD}${prompt}${RESET}" >&2
  for opt in "${options[@]}"; do
    echo "  $i) $opt" >&2
    ((i++))
  done
  local choice
  read -r -p "番号を選択: " choice
  if ! [[ "$choice" =~ ^[1-9][0-9]*$ ]] || (( choice < 1 || choice > ${#options[@]} )); then
    die "無効な選択: $choice"
  fi
  echo "${options[$((choice - 1))]}"
}

select_region() {
  local options=("${COMMON_REGIONS[@]}" "その他 (手入力)")
  local choice
  choice=$(select_from_menu "リージョンを選択" "${options[@]}")
  if [[ "$choice" == "その他 (手入力)" ]]; then
    read -r -p "リージョン名を入力 (例: us-east-1): " choice
    [[ -z "$choice" ]] && die "リージョンが空です"
  fi
  echo "$choice"
}

# ── 不足分を対話で補完 ──────────────────────────────────
if [[ -z "$CDK_COMMAND" ]]; then
  CDK_COMMAND=$(select_from_menu "実行するコマンドを選択" "${VALID_COMMANDS[@]}")
fi
if [[ -z "$ENV_NAME" ]]; then
  ENV_NAME=$(select_from_menu "対象環境を選択" "${VALID_ENVS[@]}")
fi
if [[ -z "$REGION" ]]; then
  REGION=$(select_region)
fi

# ── バリデーション ───────────────────────────────────────
contains "$CDK_COMMAND" "${VALID_COMMANDS[@]}" \
  || die "コマンドは ${VALID_COMMANDS[*]} のいずれかを指定してください (指定: ${CDK_COMMAND})"
contains "$ENV_NAME" "${VALID_ENVS[@]}" \
  || die "環境は ${VALID_ENVS[*]} のいずれかを指定してください (指定: ${ENV_NAME})"
[[ "$REGION" =~ ^[a-z]{2}-[a-z]+-[0-9]+$ ]] \
  || die "リージョン形式が不正です (指定: ${REGION})"

# ── AWS 認証情報の確認 ───────────────────────────────────
echo "${CYAN}[1/3]${RESET} AWS 認証情報を確認中..."
CALLER_JSON=$(aws sts get-caller-identity --output json 2>/dev/null) \
  || die "AWS credentials が未設定または無効です。aws sso login / aws configure などを実行してください。"

ACCOUNT_ID=$(echo "$CALLER_JSON" | sed -n 's/.*"Account": *"\([^"]*\)".*/\1/p')
ARN=$(echo "$CALLER_JSON" | sed -n 's/.*"Arn": *"\([^"]*\)".*/\1/p')

# ── スタック名解決 ───────────────────────────────────────
echo "${CYAN}[2/3]${RESET} スタック名を解決中..."
if [[ "$ENV_NAME" == "base" ]]; then
  STACK_NAME="OcrAppStack"
else
  STACK_NAME="OcrAppStack-${ENV_NAME}"
fi

# ── サマリー表示 ─────────────────────────────────────────
echo "${CYAN}[3/3]${RESET} 設定サマリー"
echo "  ${BOLD}Command:${RESET} $CDK_COMMAND"
echo "  ${BOLD}Env:    ${RESET} ${GREEN}$ENV_NAME${RESET}"
echo "  ${BOLD}Region: ${RESET} ${GREEN}$REGION${RESET}"
echo "  ${BOLD}Account:${RESET} $ACCOUNT_ID"
echo "  ${BOLD}Role:   ${RESET} $ARN"
echo "  ${BOLD}Stack:  ${RESET} $STACK_NAME"
echo

# ── 確認プロンプト ──────────────────────────────────────
if [[ "$SKIP_CONFIRM" == "false" ]]; then
  if [[ "$ENV_NAME" == "prod" ]]; then
    echo "${RED}${BOLD}⚠ 本番環境への操作です。十分に注意してください。${RESET}"
  fi
  if [[ "$CDK_COMMAND" == "destroy" ]]; then
    echo "${RED}${BOLD}⚠ destroy はリソースとデータを完全に削除します。${RESET}"
  fi
  case "$ENV_NAME" in
    prod) marker="${RED}${BOLD}" ;;
    stg)  marker="${YELLOW}${BOLD}" ;;
    *)    marker="${GREEN}" ;;
  esac
  echo -n "${marker}[$ENV_NAME @ $REGION]${RESET} に対して ${BOLD}$CDK_COMMAND${RESET} を実行します。続行しますか？ [y/N]: "
  read -r answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] || die "キャンセルしました"
fi

# ── CDK コマンド実行 ────────────────────────────────────
# bin/ocr-app.ts は以下の環境変数を参照する:
#   - ENV: 環境名 (base/dev/stg/prod) → parameters.ts でルックアップ
#   - CDK_DEFAULT_REGION: デプロイ先リージョン
#   - CDK_DEFAULT_ACCOUNT: デプロイ先 AWS アカウント
# 加えて、CDK の bundling / asset publishing や、Docker 内で走る AWS SDK
# (boto3 等) が region を必要とするため、AWS_REGION / AWS_DEFAULT_REGION
# も同じ値で export する。片方だけだと SDK が ~/.aws/config の
# default region にフォールバックしてしまう場合がある。
export ENV="$ENV_NAME"
export CDK_DEFAULT_ACCOUNT="$ACCOUNT_ID"
export CDK_DEFAULT_REGION="$REGION"
export AWS_REGION="$REGION"
export AWS_DEFAULT_REGION="$REGION"

echo "${CYAN}▶${RESET} ENV=$ENV_NAME CDK_DEFAULT_REGION=$REGION AWS_REGION=$REGION npx cdk ${CDK_COMMAND} --all"
exec npx cdk "$CDK_COMMAND" --all

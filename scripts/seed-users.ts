#!/usr/bin/env ts-node
/**
 * CSV から Cognito ユーザーを一括作成し、DSQL に role / groups を反映する。
 *
 * CSV 形式:
 *   email,tempPassword,role,groups,displayName
 *   alice@example.com,,admin,admin|team-a,Alice
 *
 * tempPassword は空可。空なら Cognito が自動生成し、既定では招待メールで
 * 利用者へ届く。値を書くと従来通りその値を仮 PW としてセット。
 *
 * 冪等性:
 *   - Cognito に存在するユーザーは PW 系操作をスキップ (本 PW を壊さない)
 *   - DSQL の role / groups は CSV 値で上書き (追加専用、CSV に無いグループは剥がさない)
 *
 * Usage:
 *   非対話モード(引数を全指定、cdk.sh --seed-users から呼ばれるケース):
 *     npx ts-node scripts/seed-users.ts \
 *       --csv users.csv \
 *       --user-pool-id us-east-1_XXXX \
 *       --dsql-endpoint xxxxx.dsql.us-east-1.on.aws \
 *       --region us-east-1 \
 *       [--suppress-invitation] \
 *       [--dry-run]
 *
 *   対話モード(引数を省略、npm run settings:users で叩くケース):
 *     引数のうち --csv / --region / (--user-pool-id + --dsql-endpoint) が
 *     欠けていれば、その項目のみ対話式で選択する。
 *     AWS 認証情報からアカウント表示 → region 選択 → CloudFormation
 *     スタック選択 → Outputs から UserPoolId / DsqlEndpoint 自動取得 →
 *     CSV 選択 → プレビュー → 実行確認、というフローで進む。
 *
 * 必要な IAM 権限:
 *   - sts:GetCallerIdentity (対話モードのアカウント表示)
 *   - cloudformation:ListStacks / DescribeStacks (対話モードのスタック選択)
 *   - cognito-idp:AdminGetUser / AdminCreateUser
 *   - dsql:DbConnectAdmin
 */
import * as fs from "fs";
import * as path from "path";
import * as readline from "readline";
import { execFileSync } from "child_process";
import {
  CognitoIdentityProviderClient,
  AdminGetUserCommand,
  AdminCreateUserCommand,
  UserNotFoundException,
  MessageActionType,
  DeliveryMediumType,
} from "@aws-sdk/client-cognito-identity-provider";
import { DsqlSigner } from "@aws-sdk/dsql-signer";
import * as pg from "pg";

const { Client } = pg;

// ---------- 型 ----------

type Role = "admin" | "author" | "reader";
const VALID_ROLES: Role[] = ["admin", "author", "reader"];

interface UserRow {
  email: string;
  // 空文字列可。空なら Cognito 側で自動生成させる。
  tempPassword: string;
  role: Role;
  groups: string[];
  displayName: string;
  lineNumber: number;
}

interface RawOptions {
  csvPath?: string;
  userPoolId?: string;
  dsqlEndpoint?: string;
  region?: string;
  suppressInvitation: boolean;
  dryRun: boolean;
}

interface Options {
  csvPath: string;
  userPoolId: string;
  dsqlEndpoint: string;
  region: string;
  suppressInvitation: boolean;
  dryRun: boolean;
}

interface Summary {
  cognitoCreated: number;
  cognitoSkipped: number;
  cognitoFailed: number;
  dsqlUpserted: number;
  dsqlFailed: number;
  groupsCreated: number;
  userGroupsCreated: number;
}

// ---------- CLI ----------

function parseArgs(): RawOptions {
  const args = process.argv.slice(2);
  const opts: RawOptions = { suppressInvitation: false, dryRun: false };

  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    switch (a) {
      case "--csv":
        opts.csvPath = args[++i];
        break;
      case "--user-pool-id":
        opts.userPoolId = args[++i];
        break;
      case "--dsql-endpoint":
        opts.dsqlEndpoint = args[++i];
        break;
      case "--region":
        opts.region = args[++i];
        break;
      case "--suppress-invitation":
        opts.suppressInvitation = true;
        break;
      case "--dry-run":
        opts.dryRun = true;
        break;
      case "-h":
      case "--help":
        printHelp();
        process.exit(0);
      default:
        console.error(`Unknown argument: ${a}`);
        printHelp();
        process.exit(1);
    }
  }
  return opts;
}

function printHelp(): void {
  console.log(`Usage: seed-users.ts [--csv <path>] [--user-pool-id <id>] [--dsql-endpoint <endpoint>] [--region <region>] [--suppress-invitation] [--dry-run]

Modes:
  引数を全指定すると非対話モード (cdk.sh --seed-users から呼ばれるケース)。
  --csv / --region / --user-pool-id / --dsql-endpoint のいずれかが欠けていれば
  対話モードに入り、アカウント確認・region 選択・CloudFormation スタック選択・
  Outputs 自動取得・CSV プレビューを経て実行確認する。

Options:
  --csv <path>            CSV file with columns: email,tempPassword,role,groups,displayName
                          (tempPassword は空可。空なら Cognito が仮 PW を自動生成する)
  --user-pool-id <id>     Cognito UserPool ID (e.g. us-east-1_XXXX)
  --dsql-endpoint <ep>    DSQL cluster endpoint hostname
  --region <region>       AWS region (e.g. us-east-1)
  --suppress-invitation   Cognito の招待メール送信を抑止する (default: 送信する)
  --dry-run               Do not write to Cognito or DSQL; print planned actions only
  -h, --help              Show this help
`);
}

// ---------- インタラクティブ ヘルパー ----------

const CYAN = process.stdout.isTTY ? "\x1b[36m" : "";
const GREEN = process.stdout.isTTY ? "\x1b[32m" : "";
const YELLOW = process.stdout.isTTY ? "\x1b[33m" : "";
const BOLD = process.stdout.isTTY ? "\x1b[1m" : "";
const RESET = process.stdout.isTTY ? "\x1b[0m" : "";

const COMMON_REGIONS = ["ap-northeast-1", "us-east-1"];

/** readline で 1 行入力を受ける。 */
function ask(question: string): Promise<string> {
  return new Promise((resolve) => {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question(question, (answer) => {
      rl.close();
      resolve(answer.trim());
    });
  });
}

/** 番号選択メニュー。1..options.length を受け付ける。 */
async function selectFromMenu(title: string, options: string[]): Promise<number> {
  console.log(`${BOLD}${title}${RESET}`);
  options.forEach((o, i) => console.log(`  ${i + 1}) ${o}`));
  while (true) {
    const answer = await ask("番号を選択: ");
    const n = parseInt(answer, 10);
    if (Number.isInteger(n) && n >= 1 && n <= options.length) return n - 1;
    console.log(`${YELLOW}無効な選択です${RESET}`);
  }
}

/** aws CLI を呼び出して JSON をパースして返す。失敗時は例外。 */
function awsCli(args: string[]): any {
  const out = execFileSync("aws", [...args, "--output", "json"], {
    encoding: "utf-8",
    stdio: ["ignore", "pipe", "pipe"],
  });
  return out.trim().length > 0 ? JSON.parse(out) : null;
}

interface StackInfo {
  name: string;
  status: string;
  env: string; // base | dev | stg | prod | (unknown)
}

/**
 * OcrAppStack で始まるスタックを列挙する。
 * base/dev/stg/prod の判別はスタック名末尾から行う。
 */
function listOcrStacks(region: string): StackInfo[] {
  const res = awsCli([
    "cloudformation",
    "list-stacks",
    "--region",
    region,
    "--stack-status-filter",
    "CREATE_COMPLETE",
    "UPDATE_COMPLETE",
    "UPDATE_ROLLBACK_COMPLETE",
    "IMPORT_COMPLETE",
    "IMPORT_ROLLBACK_COMPLETE",
  ]);
  const summaries: any[] = res?.StackSummaries ?? [];
  return summaries
    .filter(
      (s) =>
        typeof s.StackName === "string" &&
        s.StackName.startsWith("OcrAppStack") &&
        // WAF Stack (${stackName}-Waf) は seed 対象外。CloudFront 用 WAF のみを持ち、
        // Cognito / DSQL は含まないため選択肢から除く。
        !s.StackName.endsWith("-Waf")
    )
    .map((s): StackInfo => {
      const name: string = s.StackName;
      const rest = name.slice("OcrAppStack".length);
      let env = "base";
      if (rest.startsWith("-")) env = rest.slice(1);
      return { name, status: s.StackStatus, env };
    });
}

/**
 * 複数リージョンを横断して OcrAppStack を検索する。
 * リージョン単位で失敗した場合は警告を出してスキップし、他リージョンの結果は返す
 * (Opt-in 必要な region などで stack list API が失敗するケースを許容する)。
 */
function findOcrStacks(regions: string[]): Array<StackInfo & { region: string }> {
  const all: Array<StackInfo & { region: string }> = [];
  for (const r of regions) {
    try {
      const list = listOcrStacks(r);
      for (const s of list) all.push({ ...s, region: r });
    } catch (err) {
      console.log(
        `  ${YELLOW}(region=${r} でスタック一覧取得に失敗、スキップ: ${(err as Error).message.split("\n")[0]})${RESET}`
      );
    }
  }
  return all;
}

interface StackOutputs {
  userPoolId?: string;
  dsqlEndpoint?: string;
  all: Array<{ key: string; value: string }>;
}

function getStackOutputs(stackName: string, region: string): StackOutputs {
  const res = awsCli([
    "cloudformation",
    "describe-stacks",
    "--stack-name",
    stackName,
    "--region",
    region,
  ]);
  const outputs: any[] = res?.Stacks?.[0]?.Outputs ?? [];
  const all = outputs.map((o) => ({ key: o.OutputKey as string, value: o.OutputValue as string }));
  const userPoolId = all.find((o) => /UserPoolId/.test(o.key) && !/Client/.test(o.key))?.value;
  const dsqlEndpoint = all.find((o) => /ClusterEndpoint/.test(o.key))?.value;
  return { userPoolId, dsqlEndpoint, all };
}

/**
 * 引数が欠けている項目のみ対話で埋める。
 * cdk.sh から全引数指定で呼ばれるケース (raw が完全) では対話には入らない。
 */
async function resolveOptions(raw: RawOptions): Promise<Options> {
  const needInteractive =
    !raw.csvPath || !raw.region || !raw.userPoolId || !raw.dsqlEndpoint;

  // 非対話モード: 全部揃っていればそのまま返す。
  if (!needInteractive) {
    return raw as Options;
  }

  if (!process.stdin.isTTY) {
    console.error("Missing required arguments in non-interactive environment.");
    console.error("Required: --csv, --user-pool-id, --dsql-endpoint, --region");
    process.exit(1);
  }

  console.log(`${BOLD}== 初期ユーザー投入 対話モード ==${RESET}`);

  // ステップ 1: AWS 認証確認
  console.log(`\n${CYAN}▶${RESET} AWS 認証情報を確認中...`);
  let identity: { Account?: string; Arn?: string; UserId?: string };
  try {
    identity = awsCli(["sts", "get-caller-identity"]) ?? {};
  } catch (err) {
    console.error(`AWS 認証情報の取得に失敗しました: ${(err as Error).message}`);
    console.error("aws sso login / aws configure などで認証情報を設定してください。");
    process.exit(1);
  }
  console.log(`  ${BOLD}Account:${RESET} ${GREEN}${identity.Account ?? "?"}${RESET}`);
  console.log(`  ${BOLD}ARN:    ${RESET} ${identity.Arn ?? "?"}`);

  // ステップ 2+3: スタック検索 (region 指定があれば単一、なければ主要リージョンをスキャン)
  //             + Outputs 取得
  let region = raw.region;
  let userPoolId = raw.userPoolId;
  let dsqlEndpoint = raw.dsqlEndpoint;
  let stackName = "(引数指定)";

  if (!userPoolId || !dsqlEndpoint) {
    // 引数で region 指定があれば単一検索、なければ COMMON_REGIONS を横断検索
    let searchRegions = region ? [region] : [...COMMON_REGIONS];
    console.log(
      `\n${CYAN}▶${RESET} CloudFormation スタックを検索中 (regions: ${searchRegions.join(", ")})...`
    );
    let stacks = findOcrStacks(searchRegions);

    // 主要リージョンで見つからなければ手動リージョン選択にフォールバック
    if (stacks.length === 0 && !region) {
      console.log(
        `  ${YELLOW}主要リージョンで OcrAppStack が見つかりません。リージョンを手動選択してください。${RESET}`
      );
      const idx = await selectFromMenu(
        "リージョン選択",
        [...COMMON_REGIONS, "その他 (手入力)"]
      );
      let chosenRegion: string;
      if (idx < COMMON_REGIONS.length) {
        chosenRegion = COMMON_REGIONS[idx];
      } else {
        while (true) {
          const input = await ask("リージョン名を入力 (例: us-east-1): ");
          if (/^[a-z]{2}-[a-z]+-[0-9]+$/.test(input)) {
            chosenRegion = input;
            break;
          }
          console.log(`${YELLOW}リージョン形式が不正です${RESET}`);
        }
      }
      console.log(`\n${CYAN}▶${RESET} 再検索中 (region=${chosenRegion})...`);
      stacks = findOcrStacks([chosenRegion]);
    }

    if (stacks.length === 0) {
      console.error(
        `OcrAppStack で始まるスタックが見つかりません。まず deploy を実行してください。`
      );
      process.exit(1);
    }

    // スタックが 1 件なら自動選択、複数なら選択メニュー
    let chosen: StackInfo & { region: string };
    if (stacks.length === 1) {
      chosen = stacks[0];
      console.log(
        `  ${GREEN}${chosen.name}${RESET} [env=${chosen.env}] region=${chosen.region} (${chosen.status})`
      );
    } else {
      const labels = stacks.map(
        (s) => `${s.name} [env=${s.env}] region=${s.region} (${s.status})`
      );
      const idx = await selectFromMenu("対象スタックを選択", labels);
      chosen = stacks[idx];
    }
    stackName = chosen.name;
    region = chosen.region;

    console.log(`\n${CYAN}▶${RESET} スタックの Outputs から UserPoolId / DsqlEndpoint を取得...`);
    let outputs: StackOutputs;
    try {
      outputs = getStackOutputs(stackName, region);
    } catch (err) {
      console.error(`Outputs 取得に失敗しました: ${(err as Error).message}`);
      process.exit(1);
    }
    userPoolId = raw.userPoolId ?? outputs.userPoolId;
    dsqlEndpoint = raw.dsqlEndpoint ?? outputs.dsqlEndpoint;
    if (!userPoolId) {
      console.error("Stack Outputs に UserPoolId が見つかりませんでした。");
      process.exit(1);
    }
    if (!dsqlEndpoint) {
      console.error("Stack Outputs に ClusterEndpoint (DSQL) が見つかりませんでした。");
      process.exit(1);
    }
    console.log(`  ${BOLD}UserPoolId:  ${RESET} ${GREEN}${userPoolId}${RESET}`);
    console.log(`  ${BOLD}DsqlEndpoint:${RESET} ${GREEN}${dsqlEndpoint}${RESET}`);
  } else {
    // UserPoolId / DsqlEndpoint が引数指定されていれば、region も引数必須 (すでに raw.region があるはず)
    if (!region) {
      console.error("--user-pool-id / --dsql-endpoint を指定する場合は --region も必要です。");
      process.exit(1);
    }
    console.log(
      `\n${CYAN}▶${RESET} UserPoolId / DsqlEndpoint は引数指定 (スタック選択スキップ, region=${region})`
    );
  }

  // CSV 選択
  let csvPath = raw.csvPath;
  if (!csvPath) {
    console.log(`\n${CYAN}▶${RESET} CSV ファイルを選択`);
    const defaultCandidate = fs.existsSync("users.csv") ? "users.csv" : null;
    const promptLabel = defaultCandidate
      ? `CSV パス [Enter で ${defaultCandidate}]: `
      : "CSV パス: ";
    while (true) {
      const input = await ask(promptLabel);
      const candidate = input.length === 0 ? defaultCandidate : input;
      if (!candidate) {
        console.log(`${YELLOW}CSV パスを入力してください${RESET}`);
        continue;
      }
      if (!fs.existsSync(candidate)) {
        console.log(`${YELLOW}ファイルが見つかりません: ${candidate}${RESET}`);
        continue;
      }
      csvPath = candidate;
      break;
    }
  } else {
    console.log(`\n${CYAN}▶${RESET} CSV: ${GREEN}${csvPath}${RESET} (引数指定)`);
    if (!fs.existsSync(csvPath)) {
      console.error(`CSV ファイルが見つかりません: ${csvPath}`);
      process.exit(1);
    }
  }

  // CSV プレビュー + 最終確認
  console.log(`\n${CYAN}▶${RESET} CSV プレビュー`);
  const content = fs.readFileSync(csvPath, "utf-8");

  // サンプル CSV そのままの実行は事故のもと。users.example.csv と bit-identical
  // であれば警告し、明示的な y 確認を追加する。
  if (fs.existsSync("users.example.csv")) {
    try {
      const exampleContent = fs.readFileSync("users.example.csv", "utf-8");
      if (content === exampleContent) {
        console.log(
          `  ${YELLOW}⚠  警告: 選ばれた CSV は users.example.csv と同一内容です。${RESET}`
        );
        console.log(
          `  ${YELLOW}   サンプルのままである可能性があります。実データに書き換えてから再実行してください。${RESET}`
        );
        const ok = await ask("それでも続行しますか？ [y/N]: ");
        if (!/^y(es)?$/i.test(ok)) {
          console.log("キャンセルしました");
          process.exit(0);
        }
      }
    } catch {
      // 比較に失敗しても致命的ではないので無視
    }
  }

  const rows = parseCsv(content);
  const groups = Array.from(new Set(rows.flatMap((r) => r.groups))).sort();
  console.log(`  ${BOLD}Path:  ${RESET} ${path.resolve(csvPath)}`);
  console.log(`  ${BOLD}Users: ${RESET} ${rows.length}`);
  console.log(`  ${BOLD}Groups:${RESET} ${groups.length === 0 ? "(none)" : groups.join(", ")}`);
  console.log(`  ${BOLD}最初の ${Math.min(2, rows.length)} 行:${RESET}`);
  const previewCount = Math.min(2, rows.length);
  for (let i = 0; i < previewCount; i++) {
    const r = rows[i];
    console.log(
      `    ${r.email} | role=${r.role} | groups=${r.groups.join("|") || "-"} | displayName=${r.displayName}`
    );
  }
  if (rows.length > previewCount) {
    console.log(`    (ほか ${rows.length - previewCount} 行 略)`);
  }

  console.log(`\n${BOLD}== 実行内容 ==${RESET}`);
  console.log(`  ${BOLD}Account:      ${RESET} ${identity.Account ?? "?"}`);
  console.log(`  ${BOLD}Region:       ${RESET} ${region}`);
  console.log(`  ${BOLD}Stack:        ${RESET} ${stackName}`);
  console.log(`  ${BOLD}UserPoolId:   ${RESET} ${userPoolId}`);
  console.log(`  ${BOLD}DsqlEndpoint: ${RESET} ${dsqlEndpoint}`);
  console.log(`  ${BOLD}CSV:          ${RESET} ${csvPath} (${rows.length} users, ${groups.length} groups)`);
  console.log(`  ${BOLD}Send invite:  ${RESET} ${raw.suppressInvitation ? "no (SUPPRESS)" : "yes"}`);
  console.log(`  ${BOLD}Dry run:      ${RESET} ${raw.dryRun ? "yes" : "no"}`);

  const answer = await ask("\nこのまま実行しますか？ [y/N]: ");
  if (!/^y(es)?$/i.test(answer)) {
    console.log("キャンセルしました");
    process.exit(0);
  }

  return {
    csvPath: csvPath!,
    userPoolId: userPoolId!,
    dsqlEndpoint: dsqlEndpoint!,
    region: region!,
    suppressInvitation: raw.suppressInvitation,
    dryRun: raw.dryRun,
  };
}

// ---------- CSV パーサ ----------

/**
 * ダブルクォート囲みとカンマエスケープ("" で " をエスケープ) に対応した
 * 1 行分の CSV パーサ。
 */
function parseCsvLine(line: string): string[] {
  const fields: string[] = [];
  let cur = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += c;
      }
    } else {
      if (c === ",") {
        fields.push(cur);
        cur = "";
      } else if (c === '"' && cur.length === 0) {
        inQuotes = true;
      } else {
        cur += c;
      }
    }
  }
  fields.push(cur);
  return fields;
}

function parseCsv(content: string): UserRow[] {
  const lines = content.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  if (lines.length === 0) {
    throw new Error("CSV is empty");
  }

  const header = parseCsvLine(lines[0]).map((h) => h.trim());
  const expected = ["email", "tempPassword", "role", "groups", "displayName"];
  const missing = expected.filter((c) => !header.includes(c));
  if (missing.length > 0) {
    throw new Error(`CSV header missing columns: ${missing.join(", ")}. Expected: ${expected.join(",")}`);
  }
  const idx = (name: string) => header.indexOf(name);

  const rows: UserRow[] = [];
  for (let i = 1; i < lines.length; i++) {
    const lineNumber = i + 1;
    const fields = parseCsvLine(lines[i]);
    const email = (fields[idx("email")] ?? "").trim();
    const tempPassword = fields[idx("tempPassword")] ?? "";
    const roleStr = (fields[idx("role")] ?? "").trim();
    const groupsStr = (fields[idx("groups")] ?? "").trim();
    const displayName = (fields[idx("displayName")] ?? "").trim();

    if (!email) throw new Error(`Line ${lineNumber}: email is required`);
    if (!VALID_ROLES.includes(roleStr as Role)) {
      throw new Error(`Line ${lineNumber}: invalid role "${roleStr}". Must be one of ${VALID_ROLES.join(", ")}`);
    }
    const groups = groupsStr.length === 0 ? [] : groupsStr.split("|").map((g) => g.trim()).filter((g) => g.length > 0);

    rows.push({
      email,
      tempPassword,
      role: roleStr as Role,
      groups,
      displayName: displayName || email,
      lineNumber,
    });
  }
  return rows;
}

// ---------- Cognito ----------

/**
 * 指定 email のユーザーを Cognito から取得。未存在なら作成する。
 * 戻り値: { sub, created } - created=true は今回このスクリプトが作成したことを表す。
 *
 * suppressInvitation=false (既定) のとき、Cognito は招待メールを送信する。
 * row.tempPassword が空なら Cognito が仮 PW を自動生成し、その値をメール本文に埋める。
 * suppressInvitation=true なら MessageAction=SUPPRESS で作成しメールは飛ばない。
 */
async function ensureCognitoUser(
  client: CognitoIdentityProviderClient,
  userPoolId: string,
  row: UserRow,
  suppressInvitation: boolean,
  dryRun: boolean
): Promise<{ sub: string | null; created: boolean; skipped: boolean }> {
  try {
    const res = await client.send(
      new AdminGetUserCommand({ UserPoolId: userPoolId, Username: row.email })
    );
    const sub = res.UserAttributes?.find((a) => a.Name === "sub")?.Value ?? null;
    return { sub, created: false, skipped: true };
  } catch (err) {
    if (!(err instanceof UserNotFoundException)) throw err;
  }

  if (dryRun) {
    const pwLabel = row.tempPassword ? "(CSV 指定)" : "(Cognito 自動生成)";
    const mailLabel = suppressInvitation ? "(SUPPRESS)" : "(送信)";
    console.log(`  [DRY RUN] Would create Cognito user: ${row.email} PW=${pwLabel} メール=${mailLabel}`);
    return { sub: null, created: true, skipped: false };
  }

  const createRes = await client.send(
    new AdminCreateUserCommand({
      UserPoolId: userPoolId,
      Username: row.email,
      // tempPassword 空なら Cognito 側で自動生成させる (キー自体を含めない)
      ...(row.tempPassword ? { TemporaryPassword: row.tempPassword } : {}),
      // suppress 指定時のみ MessageAction=SUPPRESS。未指定なら Cognito が招待メールを送る
      MessageAction: suppressInvitation ? MessageActionType.SUPPRESS : undefined,
      // メール送信モード時のみ配信メディアを明示。SUPPRESS 時に付けると API 側で拒否される
      DesiredDeliveryMediums: suppressInvitation ? undefined : [DeliveryMediumType.EMAIL],
      UserAttributes: [
        { Name: "email", Value: row.email },
        { Name: "email_verified", Value: "true" },
        { Name: "name", Value: row.displayName },
      ],
    })
  );
  const sub = createRes.User?.Attributes?.find((a) => a.Name === "sub")?.Value ?? null;
  return { sub, created: true, skipped: false };
}

// ---------- DSQL ----------

async function connectDsql(endpoint: string, region: string): Promise<pg.Client> {
  const signer = new DsqlSigner({ hostname: endpoint, region });
  const token = await signer.getDbConnectAdminAuthToken();
  const client = new Client({
    host: endpoint,
    port: 5432,
    user: "admin",
    password: token,
    database: "postgres",
    ssl: true,
  });
  await client.connect();
  return client;
}

/**
 * groups テーブルに冪等 upsert。既存 (source='idp' 等含む) はそのまま。
 * 戻り値: name → group_id のマップ。
 */
async function upsertGroups(
  pgClient: pg.Client,
  groupNames: string[],
  dryRun: boolean
): Promise<{ map: Map<string, string>; created: number }> {
  const map = new Map<string, string>();
  let created = 0;
  for (const name of groupNames) {
    if (dryRun) {
      console.log(`  [DRY RUN] Would ensure group exists: ${name}`);
      map.set(name, "<dry-run>");
      continue;
    }
    const ins = await pgClient.query(
      `INSERT INTO groups (name, source) VALUES ($1, 'manual')
       ON CONFLICT (name) DO NOTHING
       RETURNING id`,
      [name]
    );
    if (ins.rows.length > 0) {
      map.set(name, ins.rows[0].id);
      created++;
    } else {
      const sel = await pgClient.query(`SELECT id FROM groups WHERE name = $1`, [name]);
      if (sel.rows.length === 0) throw new Error(`Group vanished after upsert: ${name}`);
      map.set(name, sel.rows[0].id);
    }
  }
  return { map, created };
}

/**
 * users テーブルに upsert。cognito_sub をキーに INSERT、既存なら role / display_name を更新。
 * post-auth Lambda の SQL は role を保護するので競合しない。
 */
async function upsertUser(
  pgClient: pg.Client,
  sub: string,
  row: UserRow,
  dryRun: boolean
): Promise<string> {
  if (dryRun) {
    console.log(`  [DRY RUN] Would upsert DSQL user: ${row.email} role=${row.role}`);
    return "<dry-run>";
  }
  const res = await pgClient.query(
    `INSERT INTO users (cognito_sub, email, display_name, role)
     VALUES ($1, $2, $3, $4)
     ON CONFLICT (cognito_sub) DO UPDATE SET
       role = EXCLUDED.role,
       display_name = EXCLUDED.display_name,
       updated_at = now()
     RETURNING id`,
    [sub, row.email, row.displayName, row.role]
  );
  return res.rows[0].id;
}

/**
 * user_groups に紐付け。source='manual' で INSERT、既存も source='manual' に更新して
 * post-auth Lambda の IdP 同期削除ロジック (source='idp' のみ対象) から保護する。
 * 戻り値: 追加された件数 (既存を更新した場合も 1 としてカウント)。
 */
async function linkUserGroup(
  pgClient: pg.Client,
  userId: string,
  groupId: string,
  dryRun: boolean
): Promise<void> {
  if (dryRun) return;
  await pgClient.query(
    `INSERT INTO user_groups (user_id, group_id, source, synced_at)
     VALUES ($1, $2, 'manual', now())
     ON CONFLICT (user_id, group_id) DO UPDATE SET
       source = 'manual',
       synced_at = now()`,
    [userId, groupId]
  );
}

// ---------- main ----------

async function main() {
  const raw = parseArgs();
  const opts = await resolveOptions(raw);
  console.log(`\nReading CSV: ${opts.csvPath}`);
  const content = fs.readFileSync(opts.csvPath, "utf-8");
  const rows = parseCsv(content);
  console.log(`Parsed ${rows.length} user row(s).`);

  const allGroups = Array.from(new Set(rows.flatMap((r) => r.groups))).sort();
  console.log(`Distinct groups to ensure: ${allGroups.length === 0 ? "(none)" : allGroups.join(", ")}`);

  if (opts.dryRun) {
    console.log("--- DRY RUN mode: no writes will be made ---");
  }

  const summary: Summary = {
    cognitoCreated: 0,
    cognitoSkipped: 0,
    cognitoFailed: 0,
    dsqlUpserted: 0,
    dsqlFailed: 0,
    groupsCreated: 0,
    userGroupsCreated: 0,
  };

  const cognito = new CognitoIdentityProviderClient({ region: opts.region });

  // sub 収集フェーズ (Cognito のみ触る。DSQL 接続前に完了させ、接続時間を短くする)
  const resolved: Array<{ row: UserRow; sub: string | null }> = [];
  for (const row of rows) {
    console.log(`\n[Cognito] ${row.email} (line ${row.lineNumber})`);
    try {
      const r = await ensureCognitoUser(cognito, opts.userPoolId, row, opts.suppressInvitation, opts.dryRun);
      if (r.created) {
        summary.cognitoCreated++;
        console.log(`  created (sub=${r.sub ?? "<unknown>"})`);
      } else {
        summary.cognitoSkipped++;
        console.log(`  exists, skipping Cognito writes (sub=${r.sub ?? "<unknown>"})`);
      }
      resolved.push({ row, sub: r.sub });
    } catch (err) {
      summary.cognitoFailed++;
      console.error(`  FAILED: ${(err as Error).message}`);
    }
  }

  // DSQL フェーズ
  let pgClient: pg.Client | null = null;
  try {
    if (!opts.dryRun) {
      console.log("\nConnecting to DSQL...");
      pgClient = await connectDsql(opts.dsqlEndpoint, opts.region);
    }

    // groups 先行 upsert
    console.log("\n[DSQL] Ensuring groups...");
    let groupMap = new Map<string, string>();
    if (allGroups.length > 0) {
      if (opts.dryRun) {
        for (const g of allGroups) console.log(`  [DRY RUN] Would ensure group exists: ${g}`);
        groupMap = new Map(allGroups.map((g) => [g, "<dry-run>"]));
      } else {
        const r = await upsertGroups(pgClient!, allGroups, opts.dryRun);
        groupMap = r.map;
        summary.groupsCreated = r.created;
      }
    }

    // users + user_groups
    for (const { row, sub } of resolved) {
      if (!sub && !opts.dryRun) {
        console.error(`\n[DSQL] ${row.email}: no sub available (Cognito step failed), skipping.`);
        summary.dsqlFailed++;
        continue;
      }
      console.log(`\n[DSQL] ${row.email}`);
      try {
        const userId = opts.dryRun
          ? "<dry-run>"
          : await upsertUser(pgClient!, sub!, row, opts.dryRun);
        if (!opts.dryRun) console.log(`  upserted (user_id=${userId}, role=${row.role})`);
        summary.dsqlUpserted++;

        for (const gname of row.groups) {
          const gid = groupMap.get(gname);
          if (!gid) {
            console.error(`  group id missing for ${gname}, skipping link`);
            continue;
          }
          if (opts.dryRun) {
            console.log(`  [DRY RUN] Would link to group: ${gname}`);
          } else {
            await linkUserGroup(pgClient!, userId, gid, opts.dryRun);
            console.log(`  linked to group: ${gname}`);
          }
          summary.userGroupsCreated++;
        }
      } catch (err) {
        summary.dsqlFailed++;
        console.error(`  FAILED: ${(err as Error).message}`);
      }
    }
  } finally {
    if (pgClient) await pgClient.end();
  }

  // サマリ
  console.log("\n=== Summary ===");
  console.log(`Cognito created:  ${summary.cognitoCreated}`);
  console.log(`Cognito skipped:  ${summary.cognitoSkipped}`);
  console.log(`Cognito failed:   ${summary.cognitoFailed}`);
  console.log(`DSQL upserted:    ${summary.dsqlUpserted}`);
  console.log(`DSQL failed:      ${summary.dsqlFailed}`);
  console.log(`Groups created:   ${summary.groupsCreated}`);
  console.log(`User-group links: ${summary.userGroupsCreated}`);
  if (opts.dryRun) console.log("(dry-run: no actual writes performed)");

  if (summary.cognitoFailed > 0 || summary.dsqlFailed > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

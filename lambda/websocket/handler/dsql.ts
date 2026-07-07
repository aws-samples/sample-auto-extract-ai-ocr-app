// DSQL への軽量クライアント + 権限チェックロジック。
// Python 実装（lambda/api/app/repositories/usecase_repository.py,
// dependencies/auth.py の check_usecase_permission）と同じロジックを
// TypeScript で再実装している。
//
// 注意: ロジックを変更する場合は Python 側の実装と同期を取ること。
import { DsqlSigner } from "@aws-sdk/dsql-signer";
import * as pg from "pg";

const { Client } = pg;

const LEVEL_RANK: Record<string, number> = { viewer: 1, editor: 2, owner: 3 };

async function getClient(): Promise<pg.Client> {
  const endpoint = process.env.DSQL_ENDPOINT!;
  const region = process.env.DSQL_REGION!;
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

export interface DsqlUser {
  id: string;
  role: string;
  display_name: string | null;
}

/** cognito_sub からユーザー(id, role, display_name)を取得。lambda/api/app/repositories/user_repository.py の get_user_by_cognito_sub 相当（display_name も含めて取得） */
export async function getUserByCognitoSub(cognitoSub: string): Promise<DsqlUser | null> {
  const client = await getClient();
  try {
    const res = await client.query<DsqlUser>(
      "SELECT id, role, display_name FROM users WHERE cognito_sub = $1",
      [cognitoSub]
    );
    return res.rows[0] ?? null;
  } finally {
    await client.end();
  }
}

/** app_name に対するユーザーの最大権限を取得。usecase_repository.py の get_user_max_permission と同義 */
export async function getUserMaxPermission(userId: string, appName: string): Promise<string | null> {
  const client = await getClient();
  try {
    const res = await client.query<{ permission: string }>(
      `SELECT COALESCE(uu.permission, gu.permission) AS permission
       FROM usecases uc
       LEFT JOIN user_usecases uu ON uu.usecase_id = uc.id AND uu.user_id = $1
       LEFT JOIN (
           SELECT gu2.usecase_id, gu2.permission
           FROM group_usecases gu2
           JOIN user_groups ug ON ug.group_id = gu2.group_id AND ug.user_id = $1
       ) gu ON gu.usecase_id = uc.id
       WHERE uc.app_name = $2
         AND (uu.permission IS NOT NULL OR gu.permission IS NOT NULL)`,
      [userId, appName]
    );
    if (res.rows.length === 0) return null;
    return res.rows.reduce((best, r) =>
      (LEVEL_RANK[r.permission] ?? 0) > (LEVEL_RANK[best] ?? 0) ? r.permission : best,
      res.rows[0].permission
    );
  } finally {
    await client.end();
  }
}

/**
 * ユーザーが image_id（に紐づく app_name）を最低 viewer 権限で見られるか判定する。
 * dependencies/auth.py の check_usecase_permission(min_level="viewer") と同義。
 * admin ロールは常に true。
 * 併せてユーザーの display_name も返す（プレゼンス表示用）。
 */
export async function canViewUsecase(
  cognitoSub: string,
  appName: string
): Promise<{ allowed: boolean; displayName: string | null }> {
  const user = await getUserByCognitoSub(cognitoSub);
  if (!user) return { allowed: false, displayName: null };
  if (user.role === "admin") return { allowed: true, displayName: user.display_name };

  const perm = await getUserMaxPermission(user.id, appName);
  const allowed = !!perm && (LEVEL_RANK[perm] ?? 0) >= LEVEL_RANK["viewer"];
  return { allowed, displayName: user.display_name };
}

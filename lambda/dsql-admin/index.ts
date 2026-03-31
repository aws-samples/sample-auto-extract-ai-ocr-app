import { DsqlSigner } from "@aws-sdk/dsql-signer";
import {
  CognitoIdentityProviderClient,
  ListUsersCommand,
} from "@aws-sdk/client-cognito-identity-provider";
import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, QueryCommand } from "@aws-sdk/lib-dynamodb";
import * as pg from "pg";
import { readFileSync } from "fs";
import { join } from "path";

const { Client } = pg;

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

async function executeDdl(client: pg.Client): Promise<string[]> {
  const sql = readFileSync(join(__dirname, "ddl.sql"), "utf-8");
  const statements = sql
    .split(";")
    .map((s) => s.trim())
    .filter(
      (s) => s && !s.split("\n").every((l) => l.startsWith("--") || !l.trim())
    );

  const results: string[] = [];
  for (const stmt of statements) {
    try {
      await client.query(stmt);
      results.push(`OK: ${stmt.substring(0, 60)}...`);
    } catch (err: any) {
      if (err.code === "42P07" || err.code === "42710") {
        results.push(`SKIP (exists): ${stmt.substring(0, 60)}...`);
      } else {
        throw err;
      }
    }
  }
  return results;
}

async function executeSql(client: pg.Client, sql: string): Promise<any> {
  const result = await client.query(sql);
  return { rowCount: result.rowCount, rows: result.rows };
}

/**
 * Seed: Cognito ユーザー → DSQL users, "all" グループ作成,
 * SchemasTable → DSQL usecases, 権限割り当て。全て冪等。
 */
async function executeSeed(client: pg.Client): Promise<string[]> {
  const results: string[] = [];
  const region = process.env.DSQL_REGION!;

  // 1. Cognito ユーザーを DSQL に同期
  const cognitoClient = new CognitoIdentityProviderClient({ region });
  const userPoolId = process.env.USER_POOL_ID!;
  let paginationToken: string | undefined;
  let userCount = 0;

  do {
    const res = await cognitoClient.send(
      new ListUsersCommand({
        UserPoolId: userPoolId,
        Limit: 60,
        PaginationToken: paginationToken,
      })
    );
    for (const u of res.Users ?? []) {
      const sub = u.Username ?? u.Attributes?.find((a) => a.Name === "sub")?.Value;
      const email = u.Attributes?.find((a) => a.Name === "email")?.Value;
      if (!sub || !email) continue;
      await client.query(
        `INSERT INTO users (cognito_sub, email, role)
         VALUES ($1, $2, 'reader')
         ON CONFLICT (cognito_sub) DO NOTHING`,
        [sub, email]
      );
      userCount++;
    }
    paginationToken = res.PaginationToken;
  } while (paginationToken);

  results.push(`Users synced: ${userCount}`);

  // 2. "all" グループ作成
  const groupRes = await client.query(
    `INSERT INTO groups (name, description, source)
     VALUES ('all', '全ユーザー', 'auto')
     ON CONFLICT (name) DO NOTHING
     RETURNING id`
  );
  let allGroupId: string;
  if (groupRes.rows.length > 0) {
    allGroupId = groupRes.rows[0].id;
    results.push(`Group 'all' created: ${allGroupId}`);
  } else {
    const existing = await client.query(
      `SELECT id FROM groups WHERE name = 'all'`
    );
    allGroupId = existing.rows[0].id;
    results.push(`Group 'all' already exists: ${allGroupId}`);
  }

  // 3. 全ユーザーを all グループに追加
  const allUsers = await client.query(`SELECT id FROM users`);
  for (const row of allUsers.rows) {
    await client.query(
      `INSERT INTO user_groups (user_id, group_id, source)
       VALUES ($1, $2, 'auto')
       ON CONFLICT (user_id, group_id) DO NOTHING`,
      [row.id, allGroupId]
    );
  }
  results.push(`Users added to 'all' group: ${allUsers.rows.length}`);

  // 4. SchemasTable → DSQL usecases
  const ddbClient = DynamoDBDocumentClient.from(
    new DynamoDBClient({ region })
  );
  const schemasTableName = process.env.SCHEMAS_TABLE_NAME!;
  const ddbRes = await ddbClient.send(
    new QueryCommand({
      TableName: schemasTableName,
      KeyConditionExpression: "schema_type = :st",
      ExpressionAttributeValues: { ":st": "app" },
    })
  );

  // usecases の owner は最初のユーザー（いなければスキップ）
  const firstUser = allUsers.rows[0];
  if (!firstUser) {
    results.push("No users found, skipping usecases seed");
    return results;
  }

  let ucCount = 0;
  for (const item of ddbRes.Items ?? []) {
    const appName = item.name as string;
    if (!appName) continue;
    const ucRes = await client.query(
      `INSERT INTO usecases (app_name, created_by)
       VALUES ($1, $2)
       ON CONFLICT (app_name) DO NOTHING
       RETURNING id`,
      [appName, firstUser.id]
    );

    let ucId: string;
    if (ucRes.rows.length > 0) {
      ucId = ucRes.rows[0].id;
    } else {
      const existing = await client.query(
        `SELECT id FROM usecases WHERE app_name = $1`,
        [appName]
      );
      ucId = existing.rows[0].id;
    }

    // 作成者を owner として user_usecases に追加
    await client.query(
      `INSERT INTO user_usecases (user_id, usecase_id, permission)
       VALUES ($1, $2, 'owner')
       ON CONFLICT (user_id, usecase_id) DO UPDATE SET permission = 'owner'`,
      [firstUser.id, ucId]
    );

    // all グループに権限付与
    await client.query(
      `INSERT INTO group_usecases (group_id, usecase_id, permission)
       VALUES ($1, $2, 'viewer')
       ON CONFLICT (group_id, usecase_id) DO NOTHING`,
      [allGroupId, ucId]
    );
    ucCount++;
  }
  results.push(`Usecases synced: ${ucCount}`);

  return results;
}

export async function handler(event: any): Promise<any> {
  const action = event.ResourceProperties?.action || event.action;
  const requestType = event.RequestType;

  if (requestType === "Delete") {
    return { PhysicalResourceId: event.PhysicalResourceId || "dsql-admin" };
  }

  const client = await getClient();
  try {
    switch (action) {
      case "ddl": {
        const results = await executeDdl(client);
        return {
          PhysicalResourceId: "dsql-ddl",
          Data: { results: JSON.stringify(results) },
        };
      }
      case "seed": {
        const results = await executeSeed(client);
        return {
          PhysicalResourceId: "dsql-seed",
          Data: { results: JSON.stringify(results) },
        };
      }
      case "execute_sql": {
        const sql = event.ResourceProperties?.sql || event.sql;
        const result = await executeSql(client, sql);
        return {
          PhysicalResourceId: "dsql-sql",
          Data: result,
        };
      }
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  } finally {
    await client.end();
  }
}

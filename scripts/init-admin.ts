#!/usr/bin/env ts-node
/**
 * 指定メールアドレスのユーザーを admin に昇格する。
 * Usage: npx ts-node scripts/init-admin.ts --email user@example.com
 *
 * 必要な環境変数: DSQL_ENDPOINT, DSQL_REGION (または AWS_REGION)
 * 必要な IAM 権限: dsql:DbConnectAdmin
 */
import { DsqlSigner } from "@aws-sdk/dsql-signer";
import * as pg from "pg";

const { Client } = pg;

async function main() {
  const emailIdx = process.argv.indexOf("--email");
  if (emailIdx === -1 || !process.argv[emailIdx + 1]) {
    console.error("Usage: npx ts-node scripts/init-admin.ts --email <email>");
    process.exit(1);
  }
  const email = process.argv[emailIdx + 1];

  const endpoint = process.env.DSQL_ENDPOINT;
  const region = process.env.DSQL_REGION || process.env.AWS_REGION;
  if (!endpoint || !region) {
    console.error("DSQL_ENDPOINT and DSQL_REGION (or AWS_REGION) must be set");
    process.exit(1);
  }

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
  try {
    const res = await client.query(
      `UPDATE users SET role = 'admin', updated_at = now()
       WHERE email = $1 RETURNING id, email, role`,
      [email]
    );
    if (res.rows.length === 0) {
      console.error(`User not found: ${email}`);
      process.exit(1);
    }
    console.log(`Admin granted: ${JSON.stringify(res.rows[0])}`);
  } finally {
    await client.end();
  }
}

main();

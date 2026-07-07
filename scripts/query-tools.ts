#!/usr/bin/env ts-node
/**
 * DSQL の tools テーブルの中身を SELECT して表示する使い捨てスクリプト。
 * Usage: DSQL_ENDPOINT=... DSQL_REGION=us-east-1 AWS_PROFILE=... npx ts-node scripts/query-tools.ts
 */
import { DsqlSigner } from "@aws-sdk/dsql-signer";
import * as pg from "pg";

const { Client } = pg;

async function main() {
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
      `SELECT id, name, description, is_active FROM tools ORDER BY name`
    );
    console.log(`\n[tools] rows: ${res.rows.length}\n`);
    console.log(JSON.stringify(res.rows, null, 2));
  } finally {
    await client.end();
  }
}

main();

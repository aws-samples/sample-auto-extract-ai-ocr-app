import os
import logging
import time
import random
import boto3
import psycopg2

logger = logging.getLogger()
logger.setLevel(logging.INFO)

DSQL_ENDPOINT = os.environ["DSQL_ENDPOINT"]
DSQL_REGION = os.environ["DSQL_REGION"]
MAX_RETRIES = 3


def get_connection():
    """DSQL 接続を取得"""
    client = boto3.client("dsql", region_name=DSQL_REGION)
    token = client.generate_db_connect_admin_auth_token(DSQL_ENDPOINT, DSQL_REGION)
    return psycopg2.connect(
        host=DSQL_ENDPOINT,
        port=5432,
        user="admin",
        password=token,
        dbname="postgres",
        sslmode="require",
    )


def with_retry(conn, fn):
    """OCC リトライユーティリティ"""
    for attempt in range(MAX_RETRIES):
        try:
            conn.autocommit = False
            result = fn(conn)
            conn.commit()
            return result
        except psycopg2.errors.SerializationFailure:
            conn.rollback()
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(random.uniform(0, 0.1 * (2 ** attempt)))
        except Exception:
            conn.rollback()
            raise


def upsert_user(conn, cognito_sub, email, display_name, department):
    """users テーブルに upsert し、新規なら all グループに追加"""
    def _do(c):
        with c.cursor() as cur:
            # 既存チェック
            cur.execute("SELECT id FROM users WHERE cognito_sub = %s", (cognito_sub,))
            existing = cur.fetchone()
            is_new = existing is None

            # cognito_sub をキーに upsert。新規なら INSERT、既存なら email 等を最新化。
            # DSQL は SERIAL 非推奨のため PK は gen_random_uuid()（DDL 側で設定）。
            cur.execute("""
                INSERT INTO users (cognito_sub, email, display_name, department)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (cognito_sub) DO UPDATE SET
                    email = EXCLUDED.email,
                    display_name = EXCLUDED.display_name,
                    department = EXCLUDED.department,
                    updated_at = now()
                RETURNING id
            """, (cognito_sub, email, display_name, department))
            user_id = cur.fetchone()[0]

            if is_new:
                # all グループに追加（all グループは dsql-admin seed で作成済み前提）
                # seed 未実行の場合はスキップ（グループが見つからなければ何もしない）
                cur.execute("SELECT id FROM groups WHERE name = 'all'")
                row = cur.fetchone()
                if row:
                    # source='auto' で自動追加を明示。ON CONFLICT で冪等性を担保。
                    cur.execute("""
                        INSERT INTO user_groups (user_id, group_id, source)
                        VALUES (%s, %s, 'auto')
                        ON CONFLICT (user_id, group_id) DO NOTHING
                    """, (user_id, row[0]))

            return user_id
    return with_retry(conn, _do)


def sync_idp_groups(conn, user_id, idp_groups):
    """IdP グループの差分同期

    Cognito の custom:idp_group 属性から取得したグループ名リストを
    DSQL の groups / user_groups テーブルに同期する。
    source='idp' で IdP 由来のグループを区別し、手動追加分（source='manual'）とは独立管理。
    """
    def _do(c):
        with c.cursor() as cur:
            # IdP グループが DSQL に存在しなければ作成（source='idp'）
            for group_name in idp_groups:
                cur.execute("""
                    INSERT INTO groups (name, source)
                    VALUES (%s, 'idp')
                    ON CONFLICT (name) DO NOTHING
                """, (group_name,))

            # IdP 由来のグループ ID を取得
            cur.execute("""
                SELECT id, name FROM groups WHERE name = ANY(%s) AND source = 'idp'
            """, (idp_groups,))
            group_map = {row[1]: row[0] for row in cur.fetchall()}

            # 現在の IdP グループ所属を取得して差分計算
            cur.execute("""
                SELECT group_id FROM user_groups
                WHERE user_id = %s AND source = 'idp'
            """, (user_id,))
            current_ids = {row[0] for row in cur.fetchall()}
            desired_ids = set(group_map.values())

            # 追加: IdP にあって DSQL にない所属を追加
            for gid in desired_ids - current_ids:
                cur.execute("""
                    INSERT INTO user_groups (user_id, group_id, source, synced_at)
                    VALUES (%s, %s, 'idp', now())
                    ON CONFLICT (user_id, group_id) DO UPDATE SET
                        source = 'idp', synced_at = now()
                """, (user_id, gid))

            # 削除: IdP から外れた所属を削除（source='idp' のみ対象）
            for gid in current_ids - desired_ids:
                cur.execute("""
                    DELETE FROM user_groups
                    WHERE user_id = %s AND group_id = %s AND source = 'idp'
                """, (user_id, gid))

    with_retry(conn, _do)


def handler(event, context):
    """Cognito Post Authentication Trigger"""
    logger.info("Post auth trigger fired")

    attrs = event["request"]["userAttributes"]
    cognito_sub = attrs["sub"]
    email = attrs.get("email", "")
    display_name = attrs.get("name", attrs.get("email", ""))
    department = attrs.get("custom:department", "")
    idp_group_str = attrs.get("custom:idp_group", "")

    idp_groups = [g.strip() for g in idp_group_str.split(",") if g.strip()] if idp_group_str else []

    conn = get_connection()
    try:
        user_id = upsert_user(conn, cognito_sub, email, display_name, department)
        if idp_groups:
            sync_idp_groups(conn, user_id, idp_groups)
        logger.info(f"Synced user {cognito_sub}, idp_groups={idp_groups}")
    finally:
        conn.close()

    return event

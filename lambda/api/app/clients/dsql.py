"""DSQL 接続 + OCC リトライユーティリティ"""
import os
import time
import random
import logging
import boto3
import psycopg2
import psycopg2.extras

logger = logging.getLogger(__name__)

DSQL_ENDPOINT = os.environ.get("DSQL_ENDPOINT", "")
DSQL_REGION = os.environ.get("DSQL_REGION", "")
MAX_RETRIES = 3
TOKEN_REFRESH_SECONDS = 10 * 60  # 10分で再接続（token有効期限15分の余裕）

_conn = None
_conn_created_at = 0.0


def get_connection():
    """DSQL 接続を取得（再利用、token期限前に再接続）"""
    global _conn, _conn_created_at

    token_expired = (time.time() - _conn_created_at) >= TOKEN_REFRESH_SECONDS

    if _conn and not _conn.closed and not token_expired:
        try:
            _conn.cursor().execute("SELECT 1")
            return _conn
        except Exception:
            _conn = None

    if _conn and not _conn.closed:
        try:
            _conn.close()
        except Exception:
            pass

    client = boto3.client("dsql", region_name=DSQL_REGION)
    token = client.generate_db_connect_admin_auth_token(DSQL_ENDPOINT, DSQL_REGION)
    _conn = psycopg2.connect(
        host=DSQL_ENDPOINT,
        port=5432,
        user="admin",
        password=token,
        dbname="postgres",
        sslmode="require",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )
    _conn_created_at = time.time()
    return _conn


def with_retry(fn):
    """OCC リトライ付きでクエリ実行"""
    for attempt in range(MAX_RETRIES):
        conn = get_connection()
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


def query(sql, params=None):
    """SELECT クエリ実行"""
    conn = get_connection()
    if conn.status != psycopg2.extensions.STATUS_READY:
        conn.rollback()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def query_one(sql, params=None):
    """SELECT 1行取得"""
    rows = query(sql, params)
    return rows[0] if rows else None

"""ユーザー Repository（DSQL）"""
from clients.dsql import query, query_one, with_retry
from domains.user_profile import IdpUserProfile, needs_profile_update


def sync_user_from_idp(profile: IdpUserProfile) -> str:
    """IdP の属性で users を同期し、user_id を返す。

    - 新規: INSERT し、`all` グループに source='auto' で追加する
    - 既存: email / department に差分があるときだけ UPDATE する。
      ページロードのたびに呼ばれるため、変化が無いときは書き込まない（無駄な書き込みと OCC 競合を避ける）
    """
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, department FROM users WHERE cognito_sub = %s",
                (profile.cognito_sub,),
            )
            existing = cur.fetchone()
            if existing:
                if needs_profile_update(existing, profile):
                    cur.execute(
                        "UPDATE users SET email = %s, department = %s, updated_at = now() WHERE id = %s",
                        (profile.email, profile.department, existing["id"]),
                    )
                return str(existing["id"])

            # 複数タブから同時に初回 sync が来ても落ちないよう ON CONFLICT で冪等にする。
            # DSQL は SERIAL 非推奨のため PK は gen_random_uuid()（DDL 側で設定）。
            cur.execute("""
                INSERT INTO users (cognito_sub, email, display_name, department)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (cognito_sub) DO UPDATE SET
                    email = EXCLUDED.email,
                    department = EXCLUDED.department,
                    updated_at = now()
                RETURNING id
            """, (profile.cognito_sub, profile.email, profile.display_name, profile.department))
            user_id = cur.fetchone()["id"]

            # all グループは dsql-admin seed で作成済み前提。seed 未実行なら追加をスキップする
            cur.execute("SELECT id FROM groups WHERE name = 'all'")
            row = cur.fetchone()
            if row:
                cur.execute("""
                    INSERT INTO user_groups (user_id, group_id, source)
                    VALUES (%s, %s, 'auto')
                    ON CONFLICT (user_id, group_id) DO NOTHING
                """, (user_id, row["id"]))
            return str(user_id)
    return with_retry(_do)


def get_user_by_cognito_sub(cognito_sub: str) -> dict | None:
    """cognito_sub からユーザーを取得"""
    return query_one("SELECT id, role FROM users WHERE cognito_sub = %s", (cognito_sub,))


def get_user_detail_by_cognito_sub(cognito_sub: str) -> dict | None:
    """cognito_sub からユーザー詳細を取得"""
    return query_one(
        "SELECT id, email, display_name, department, role FROM users WHERE cognito_sub = %s",
        (cognito_sub,),
    )


def list_users() -> list[dict]:
    """全ユーザー一覧"""
    rows = query("""
        SELECT id, cognito_sub, email, display_name, department, role, is_active, created_at
        FROM users ORDER BY created_at DESC
    """)
    return [dict(r) for r in rows]


def update_user_role(user_id: str, role: str) -> bool:
    """ユーザーのロールを更新。存在しなければ False。"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET role = %s, updated_at = now() WHERE id = %s RETURNING id",
                (role, user_id),
            )
            return cur.fetchone() is not None
    return with_retry(_do)


def update_display_name(cognito_sub: str, display_name: str) -> bool:
    """display_name を更新。存在しなければ False。"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET display_name = %s, updated_at = now() WHERE cognito_sub = %s RETURNING id",
                (display_name, cognito_sub),
            )
            return cur.fetchone() is not None
    return with_retry(_do)


def search_users(pattern: str, limit: int = 10) -> list[dict]:
    """メール or 表示名で部分一致検索"""
    rows = query(
        "SELECT id, email, display_name FROM users WHERE email ILIKE %s OR display_name ILIKE %s LIMIT %s",
        (pattern, pattern, limit),
    )
    return [dict(r) for r in rows]


def get_emails_by_cognito_subs(subs: set[str]) -> dict[str, str]:
    """cognito_sub のセットからメールアドレスのマップを返す"""
    if not subs:
        return {}
    placeholders = ",".join(["%s"] * len(subs))
    rows = query(
        f"SELECT cognito_sub, email FROM users WHERE cognito_sub IN ({placeholders})",
        tuple(subs),
    )
    return {r["cognito_sub"]: r["email"] for r in rows}

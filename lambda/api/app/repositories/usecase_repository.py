"""ユースケース Repository（DSQL）"""
from dsql_client import query, query_one, with_retry


def get_usecase_by_app_name(app_name: str) -> dict | None:
    """app_name からユースケースを取得"""
    return query_one("SELECT id FROM usecases WHERE app_name = %s", (app_name,))


def list_usecases() -> list[dict]:
    """ユースケース一覧（作成者メール + owner メール付き）"""
    rows = query("""
        SELECT uc.id, uc.app_name, uc.created_at,
               creator.email as created_by_email,
               COALESCE(
                   (SELECT string_agg(u2.email, ',' ORDER BY u2.email)
                    FROM user_usecases uu2
                    JOIN users u2 ON uu2.user_id = u2.id
                    WHERE uu2.usecase_id = uc.id AND uu2.permission = 'owner'),
                   ''
               ) as owner_emails_csv
        FROM usecases uc
        LEFT JOIN users creator ON uc.created_by = creator.id
        ORDER BY uc.created_at DESC
    """)
    result = []
    for r in rows:
        d = dict(r)
        csv = d.pop("owner_emails_csv", "")
        d["owner_emails"] = [e for e in csv.split(",") if e] if csv else []
        result.append(d)
    return result


def get_usecase_owners(usecase_id: str) -> list[dict]:
    """ユースケースの owner 一覧"""
    rows = query("""
        SELECT u.email FROM user_usecases uu
        JOIN users u ON uu.user_id = u.id
        WHERE uu.usecase_id = %s AND uu.permission = 'owner'
        ORDER BY u.email
    """, (usecase_id,))
    return [dict(r) for r in rows]


def get_usecase_user_permissions(usecase_id: str) -> list[dict]:
    """ユースケースのユーザー権限一覧"""
    rows = query("""
        SELECT u.id, u.email, u.display_name, uu.permission
        FROM user_usecases uu JOIN users u ON uu.user_id = u.id
        WHERE uu.usecase_id = %s ORDER BY u.email
    """, (usecase_id,))
    return [dict(r) for r in rows]


def get_usecase_group_permissions(usecase_id: str) -> list[dict]:
    """ユースケースのグループ権限一覧"""
    rows = query("""
        SELECT g.id, g.name, gu.permission
        FROM group_usecases gu JOIN groups g ON gu.group_id = g.id
        WHERE gu.usecase_id = %s ORDER BY g.name
    """, (usecase_id,))
    return [dict(r) for r in rows]


def upsert_user_permission(user_id: str, usecase_id: str, permission: str) -> None:
    """ユーザーのユースケース権限を追加/更新"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_usecases (user_id, usecase_id, permission)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, usecase_id) DO UPDATE SET permission = EXCLUDED.permission
            """, (user_id, usecase_id, permission))
    with_retry(_do)


def delete_user_permission(user_id: str, usecase_id: str) -> None:
    """ユーザーのユースケース権限を削除"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_usecases WHERE user_id = %s AND usecase_id = %s", (user_id, usecase_id))
    with_retry(_do)


def count_owners(usecase_id: str) -> int:
    """ユースケースの owner 数を返す"""
    rows = query("SELECT user_id FROM user_usecases WHERE usecase_id = %s AND permission = 'owner'", (usecase_id,))
    return len(rows)


def is_owner(user_id: str, usecase_id: str) -> bool:
    """指定ユーザーが owner かどうか"""
    rows = query("SELECT user_id FROM user_usecases WHERE usecase_id = %s AND permission = 'owner'", (usecase_id,))
    return any(str(r["user_id"]) == user_id for r in rows)


def upsert_group_permission(group_id: str, usecase_id: str, permission: str) -> None:
    """グループのユースケース権限を追加/更新"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO group_usecases (group_id, usecase_id, permission)
                VALUES (%s, %s, %s)
                ON CONFLICT (group_id, usecase_id) DO UPDATE SET permission = EXCLUDED.permission
            """, (group_id, usecase_id, permission))
    with_retry(_do)


def delete_group_permission(group_id: str, usecase_id: str) -> None:
    """グループのユースケース権限を削除"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM group_usecases WHERE group_id = %s AND usecase_id = %s", (group_id, usecase_id))
    with_retry(_do)


def register_usecase_owner(app_name: str, user_id: str) -> None:
    """usecase を登録し、作成者を owner として追加"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO usecases (app_name, created_by)
                VALUES (%s, %s)
                ON CONFLICT (app_name) DO NOTHING
                RETURNING id
            """, (app_name, user_id))
            row = cur.fetchone()
            if row:
                uc_id = row["id"]
            else:
                cur.execute("SELECT id FROM usecases WHERE app_name = %s", (app_name,))
                uc_id = cur.fetchone()["id"]

            cur.execute("""
                INSERT INTO user_usecases (user_id, usecase_id, permission)
                VALUES (%s, %s, 'owner')
                ON CONFLICT (user_id, usecase_id) DO UPDATE SET permission = 'owner'
            """, (user_id, uc_id))
    with_retry(_do)


def get_user_max_permission(user_id: str, app_name: str) -> str | None:
    """ユーザーのユースケースに対する最大権限を返す"""
    rank = {"owner": 3, "editor": 2, "viewer": 1}
    rows = query("""
        SELECT COALESCE(uu.permission, gu.permission) AS permission
        FROM usecases uc
        LEFT JOIN user_usecases uu ON uu.usecase_id = uc.id AND uu.user_id = %s
        LEFT JOIN (
            SELECT gu2.usecase_id, gu2.permission
            FROM group_usecases gu2
            JOIN user_groups ug ON ug.group_id = gu2.group_id AND ug.user_id = %s
        ) gu ON gu.usecase_id = uc.id
        WHERE uc.app_name = %s
          AND (uu.permission IS NOT NULL OR gu.permission IS NOT NULL)
    """, (user_id, user_id, app_name))

    if not rows:
        return None
    return max((r["permission"] for r in rows), key=lambda p: rank.get(p, 0))


def get_permitted_app_names(user_id: str) -> list[str]:
    """ユーザーが何らかの権限を持つ app_name 一覧"""
    rows = query("""
        SELECT DISTINCT uc.app_name FROM usecases uc
        LEFT JOIN user_usecases uu ON uu.usecase_id = uc.id AND uu.user_id = %s
        LEFT JOIN (
            SELECT gu2.usecase_id
            FROM group_usecases gu2
            JOIN user_groups ug ON ug.group_id = gu2.group_id AND ug.user_id = %s
        ) gu ON gu.usecase_id = uc.id
        WHERE uu.user_id IS NOT NULL OR gu.usecase_id IS NOT NULL
    """, (user_id, user_id))
    return [r["app_name"] for r in rows]

def delete_usecase_by_app_name(app_name: str) -> None:
    """ユースケースと関連する中間テーブルのレコードを削除"""
    def _do(conn):
        with conn.cursor() as cur:
            # まず usecase_id を取得
            cur.execute("SELECT id FROM usecases WHERE app_name = %s", (app_name,))
            row = cur.fetchone()
            if not row:
                return
            uc_id = row["id"]
            # 中間テーブルを先に削除
            cur.execute("DELETE FROM user_usecases WHERE usecase_id = %s", (uc_id,))
            cur.execute("DELETE FROM group_usecases WHERE usecase_id = %s", (uc_id,))
            cur.execute("DELETE FROM usecase_tools WHERE usecase_id = %s", (uc_id,))
            # マスタを削除
            cur.execute("DELETE FROM usecases WHERE id = %s", (uc_id,))
    with_retry(_do)


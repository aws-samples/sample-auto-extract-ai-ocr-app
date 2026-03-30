"""ツール Repository（DSQL）"""
from dsql_client import query, with_retry


def list_tools() -> list[dict]:
    """ツール一覧"""
    rows = query("SELECT id, name, tool_name, description, is_active FROM tools ORDER BY name")
    return [dict(r) for r in rows]


def create_tool(name: str, tool_name: str, description: str | None = None) -> str:
    """ツールを作成し ID を返す"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tools (name, tool_name, description) VALUES (%s, %s, %s) RETURNING id",
                (name, tool_name, description),
            )
            return str(cur.fetchone()["id"])
    return with_retry(_do)


def update_tool(tool_id: str, updates: dict) -> bool:
    """ツールを更新。存在しなければ False。"""
    set_clauses, params = [], []
    for key in ("name", "description", "is_active"):
        if key in updates and updates[key] is not None:
            set_clauses.append(f"{key} = %s")
            params.append(updates[key])
    if not set_clauses:
        return False
    params.append(tool_id)

    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE tools SET {', '.join(set_clauses)} WHERE id = %s RETURNING id",
                params,
            )
            return cur.fetchone() is not None
    return with_retry(_do)


def get_tool_user_permissions(tool_id: str) -> list[dict]:
    """ツールのユーザー権限一覧"""
    rows = query("""
        SELECT u.id, u.email, u.display_name
        FROM user_tools ut JOIN users u ON ut.user_id = u.id
        WHERE ut.tool_id = %s ORDER BY u.email
    """, (tool_id,))
    return [dict(r) for r in rows]


def get_tool_group_permissions(tool_id: str) -> list[dict]:
    """ツールのグループ権限一覧"""
    rows = query("""
        SELECT g.id, g.name
        FROM group_tools gt JOIN groups g ON gt.group_id = g.id
        WHERE gt.tool_id = %s ORDER BY g.name
    """, (tool_id,))
    return [dict(r) for r in rows]


def get_tool_usecase_permissions(tool_id: str) -> list[dict]:
    """ツールのユースケース権限一覧"""
    rows = query("""
        SELECT uc.id, uc.app_name
        FROM usecase_tools uct JOIN usecases uc ON uct.usecase_id = uc.id
        WHERE uct.tool_id = %s ORDER BY uc.app_name
    """, (tool_id,))
    return [dict(r) for r in rows]


def add_tool_group(tool_id: str, group_id: str) -> None:
    """ツールにグループ権限を追加"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO group_tools (group_id, tool_id)
                VALUES (%s, %s)
                ON CONFLICT (group_id, tool_id) DO NOTHING
            """, (group_id, tool_id))
    with_retry(_do)


def remove_tool_group(tool_id: str, group_id: str) -> None:
    """ツールからグループ権限を削除"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM group_tools WHERE group_id = %s AND tool_id = %s", (group_id, tool_id))
    with_retry(_do)


def add_tool_user(tool_id: str, user_id: str) -> None:
    """ツールにユーザー権限を追加"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_tools (user_id, tool_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, tool_id) DO NOTHING
            """, (user_id, tool_id))
    with_retry(_do)


def remove_tool_user(tool_id: str, user_id: str) -> None:
    """ツールからユーザー権限を削除"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_tools WHERE user_id = %s AND tool_id = %s", (user_id, tool_id))
    with_retry(_do)

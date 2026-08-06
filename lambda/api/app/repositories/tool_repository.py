"""ツール Repository（DSQL）"""
from clients.dsql import query, with_retry


def list_tools() -> list[dict]:
    """ツール一覧"""
    rows = query("SELECT id, name, description, is_active FROM tools ORDER BY name")
    return [dict(r) for r in rows]


def create_tool(name: str, description: str | None = None) -> str:
    """ツールを作成し ID を返す"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tools (name, description) VALUES (%s, %s) RETURNING id",
                (name, description),
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


# --- Usecase Tools operations ---

def get_usecase_tools(usecase_id: str, active_only: bool = False) -> list[dict]:
    """ユースケースに紐付くツール一覧を取得

    Args:
        active_only: True の場合 is_active=true のツールのみ返す（表示用）
    """
    if active_only:
        rows = query("""
            SELECT t.id, t.name, t.description, t.is_active
            FROM usecase_tools uct
            JOIN tools t ON uct.tool_id = t.id
            WHERE uct.usecase_id = %s AND t.is_active = true
            ORDER BY t.name
        """, (usecase_id,))
    else:
        rows = query("""
            SELECT t.id, t.name, t.description, t.is_active
            FROM usecase_tools uct
            JOIN tools t ON uct.tool_id = t.id
            WHERE uct.usecase_id = %s
            ORDER BY t.name
        """, (usecase_id,))
    return [dict(r) for r in rows]


def get_usecase_allowed_tool_names(usecase_id: str) -> list[str]:
    """ユースケースに紐付く許可ツール名リストを取得"""
    rows = query("""
        SELECT t.name
        FROM usecase_tools uct
        JOIN tools t ON uct.tool_id = t.id
        WHERE uct.usecase_id = %s AND t.is_active = true
    """, (usecase_id,))
    return [r["name"] for r in rows]


def set_usecase_tools(usecase_id: str, tool_ids: list[str]) -> None:
    """ユースケースのツールを一括設定（既存を置換）"""
    def _do(conn):
        with conn.cursor() as cur:
            # 既存の紐付けを削除
            cur.execute("DELETE FROM usecase_tools WHERE usecase_id = %s", (usecase_id,))
            # 新しい紐付けを挿入
            for tool_id in tool_ids:
                cur.execute(
                    "INSERT INTO usecase_tools (usecase_id, tool_id) VALUES (%s, %s)",
                    (usecase_id, tool_id),
                )
    with_retry(_do)


def get_visible_tools_for_user(user_id: str) -> list[dict]:
    """user_tools + group_tools 経由でユーザーが閲覧可能なツールを取得"""
    rows = query("""
        SELECT DISTINCT t.id, t.name, t.description, t.is_active
        FROM tools t
        WHERE t.is_active = true AND (
            t.id IN (SELECT tool_id FROM user_tools WHERE user_id = %s)
            OR t.id IN (
                SELECT gt.tool_id FROM group_tools gt
                JOIN user_groups ug ON gt.group_id = ug.group_id
                WHERE ug.user_id = %s
            )
        )
        ORDER BY t.name
    """, (user_id, user_id))
    return [dict(r) for r in rows]



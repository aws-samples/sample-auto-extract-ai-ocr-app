"""グループ Repository（DSQL）"""
from clients.dsql import query, query_one, with_retry


def list_groups() -> list[dict]:
    """グループ一覧（メンバー数付き）"""
    rows = query("""
        SELECT g.id, g.name, g.description, g.source, g.created_at,
               COUNT(ug.user_id) as member_count
        FROM groups g
        LEFT JOIN user_groups ug ON g.id = ug.group_id
        GROUP BY g.id, g.name, g.description, g.source, g.created_at
        ORDER BY g.name
    """)
    return [dict(r) for r in rows]


def create_group(name: str, description: str | None = None) -> str:
    """グループを作成し ID を返す"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO groups (name, description) VALUES (%s, %s) RETURNING id",
                (name, description),
            )
            return str(cur.fetchone()["id"])
    return with_retry(_do)


def get_group(group_id: str) -> dict | None:
    """グループを取得"""
    return query_one("SELECT id, name, source FROM groups WHERE id = %s", (group_id,))


def update_group(group_id: str, name: str = None, description: str = None) -> bool:
    """グループの名前・説明を更新。存在しなければ False。"""
    set_clauses, params = [], []
    if name is not None:
        set_clauses.append("name = %s")
        params.append(name)
    if description is not None:
        set_clauses.append("description = %s")
        params.append(description)
    if not set_clauses:
        return False
    params.append(group_id)

    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE groups SET {', '.join(set_clauses)} WHERE id = %s RETURNING id",
                params,
            )
            return cur.fetchone() is not None
    return with_retry(_do)


def get_group_members(group_id: str) -> list[dict]:
    """グループメンバー一覧"""
    rows = query("""
        SELECT u.id, u.email, u.display_name, u.role, ug.source
        FROM user_groups ug
        JOIN users u ON ug.user_id = u.id
        WHERE ug.group_id = %s
        ORDER BY u.email
    """, (group_id,))
    return [dict(r) for r in rows]


def delete_group(group_id: str) -> None:
    """グループと関連データを削除"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM user_groups WHERE group_id = %s", (group_id,))
            cur.execute("DELETE FROM group_usecases WHERE group_id = %s", (group_id,))
            cur.execute("DELETE FROM group_tools WHERE group_id = %s", (group_id,))
            cur.execute("DELETE FROM groups WHERE id = %s", (group_id,))
    with_retry(_do)


def update_group_members(group_id: str, user_ids: list[str]) -> None:
    """manual メンバーを全削除して再追加"""
    def _do(conn):
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM user_groups WHERE group_id = %s AND source = 'manual'",
                (group_id,),
            )
            for uid in user_ids:
                cur.execute("""
                    INSERT INTO user_groups (user_id, group_id, source)
                    VALUES (%s, %s, 'manual')
                    ON CONFLICT (user_id, group_id) DO NOTHING
                """, (uid, group_id))
    with_retry(_do)


def search_groups(pattern: str, limit: int = 10) -> list[dict]:
    """グループ名で部分一致検索（auto 除外）"""
    rows = query(
        "SELECT id, name, description FROM groups WHERE name ILIKE %s AND source != 'auto' LIMIT %s",
        (pattern, limit),
    )
    return [dict(r) for r in rows]


def get_user_group_names(user_ids: list[str]) -> dict[str, list[dict]]:
    """ユーザー ID のリストから、各ユーザーが所属するグループ名+sourceのマップを返す。auto グループは除外。"""
    if not user_ids:
        return {}
    placeholders = ",".join(["%s"] * len(user_ids))
    rows = query(f"""
        SELECT ug.user_id, g.name, g.source
        FROM user_groups ug
        JOIN groups g ON ug.group_id = g.id
        WHERE ug.user_id IN ({placeholders}) AND g.source != 'auto'
        ORDER BY g.name
    """, tuple(user_ids))
    result: dict[str, list[dict]] = {}
    for r in rows:
        uid = str(r["user_id"])
        if uid not in result:
            result[uid] = []
        result[uid].append({"name": r["name"], "source": r["source"]})
    return result


def get_group_by_name(name: str) -> dict | None:
    """名前でグループを取得"""
    return query_one("SELECT id FROM groups WHERE name = %s", (name,))

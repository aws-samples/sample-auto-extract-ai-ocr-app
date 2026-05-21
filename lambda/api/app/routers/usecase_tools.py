"""ユースケース ツール設定 API（Editor 以上がアクセス可能）"""
import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from dependencies.auth import RequirePermission
from repositories import tool_repository
from repositories.usecase_repository import get_usecase_by_app_name, register_usecase_owner
from repositories.schema_repository import get_app_schema

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/usecases/{app_name}", tags=["Usecase Tools"])


def _resolve_usecase_id(app_name: str, user_id: str | None = None) -> str:
    usecase = get_usecase_by_app_name(app_name)
    if usecase:
        return str(usecase["id"])

    # Self-healing: DynamoDB にあるが DSQL にない場合は自動登録
    schema = get_app_schema(app_name)
    if not schema:
        raise HTTPException(404, f"Usecase not found: {app_name}")

    owner_id = user_id or "00000000-0000-0000-0000-000000000000"
    logger.info(f"Auto-registering pre-existing usecase in DSQL: {app_name}")
    register_usecase_owner(app_name, owner_id)

    usecase = get_usecase_by_app_name(app_name)
    if not usecase:
        raise HTTPException(500, f"Failed to auto-register usecase: {app_name}")
    return str(usecase["id"])


@router.get("/tools")
async def get_usecase_tools(app_name: str, user=Depends(RequirePermission("editor"))):
    """ユースケースに現在割当済みのツール一覧"""
    usecase_id = _resolve_usecase_id(app_name, user_id=str(user["id"]))
    return {"tools": tool_repository.get_usecase_tools(usecase_id)}


class UsecaseToolsUpdate(BaseModel):
    tool_ids: list[str]


@router.put("/tools")
async def set_usecase_tools(
    app_name: str,
    body: UsecaseToolsUpdate,
    user=Depends(RequirePermission("editor")),
):
    """ユースケースのツールを一括設定（Editor: 可視ツールのみ操作可）

    Editor の場合、可視範囲外の既存ツールは維持される（削除不可）。
    """
    usecase_id = _resolve_usecase_id(app_name, user_id=str(user["id"]))

    if user["role"] != "admin":
        visible = tool_repository.get_visible_tools_for_user(str(user["id"]))
        visible_ids = {str(t["id"]) for t in visible}

        # 追加しようとするツールが可視範囲内か検証
        invalid = [tid for tid in body.tool_ids if tid not in visible_ids]
        if invalid:
            raise HTTPException(403, f"Cannot assign tools not visible to you: {invalid}")

        # 既存の割当のうち、可視範囲外のツールは維持する
        current_tools = tool_repository.get_usecase_tools(usecase_id)
        preserved_ids = [str(t["id"]) for t in current_tools if str(t["id"]) not in visible_ids]

        # 最終セット = Editor が指定したツール + 可視範囲外の既存ツール
        final_tool_ids = list(set(body.tool_ids + preserved_ids))
        tool_repository.set_usecase_tools(usecase_id, final_tool_ids)
    else:
        tool_repository.set_usecase_tools(usecase_id, body.tool_ids)

    return {"ok": True}


@router.get("/available-tools")
async def get_available_tools(app_name: str, user=Depends(RequirePermission("editor"))):
    """ログインユーザーが選択可能なツール一覧（user_tools + group_tools 経由）"""
    # app_name の存在確認
    _resolve_usecase_id(app_name, user_id=str(user["id"]))

    if user["role"] == "admin":
        return {"tools": tool_repository.list_tools()}
    return {"tools": tool_repository.get_visible_tools_for_user(str(user["id"]))}

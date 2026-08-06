"""ユーザー設定 Repository（DynamoDB UserPreferencesTable）"""
import logging
import os
from clients import dynamodb_resource

logger = logging.getLogger(__name__)

_prefs_table = None


def _get_table():
    global _prefs_table
    if _prefs_table is None:
        _prefs_table = dynamodb_resource.Table(os.environ["USER_PREFERENCES_TABLE_NAME"])
    return _prefs_table


def get_stars(cognito_sub: str) -> list[str]:
    """ユーザーのスター一覧を取得"""
    table = _get_table()
    res = table.query(
        KeyConditionExpression="user_id = :uid AND begins_with(sk, :prefix)",
        ExpressionAttributeValues={":uid": cognito_sub, ":prefix": "star#"},
    )
    return [item["sk"].split("#", 1)[1] for item in res.get("Items", [])]


def add_star(cognito_sub: str, app_name: str) -> None:
    """スターを追加"""
    _get_table().put_item(Item={"user_id": cognito_sub, "sk": f"star#{app_name}"})


def remove_star(cognito_sub: str, app_name: str) -> None:
    """スターを削除"""
    _get_table().delete_item(Key={"user_id": cognito_sub, "sk": f"star#{app_name}"})

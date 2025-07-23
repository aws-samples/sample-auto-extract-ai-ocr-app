import json
import logging
import os
import boto3
from botocore.exceptions import ClientError

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# デフォルトアプリ
DEFAULT_APP = "shipping_ocr"

# DynamoDB クライアント
dynamodb = boto3.resource('dynamodb')


def load_app_schemas():
    """
    アプリケーションスキーマを取得する
    DynamoDB から全てのアプリスキーマを取得
    取得できない場合はエラーを返す
    """
    try:
        # DynamoDB からスキーマを取得
        schemas_table_name = os.environ.get('SCHEMAS_TABLE_NAME')
        if not schemas_table_name:
            logger.error("SCHEMAS_TABLE_NAME 環境変数が設定されていません")
            raise ValueError("SCHEMAS_TABLE_NAME environment variable is not set")
            
        logger.info(f"DynamoDB からスキーマを取得します: {schemas_table_name}")
        schemas_table = dynamodb.Table(schemas_table_name)
        
        # schema_type='app' の全てのレコードを取得
        response = schemas_table.query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key('schema_type').eq('app')
        )
        
        if 'Items' in response and response['Items']:
            # 各アプリのデータを配列に格納
            apps = []
            for item in response['Items']:
                # 新しい構造: 直接アプリデータとして扱う
                app_data = {
                    'name': item.get('name'),
                    'display_name': item.get('display_name', item.get('name')),
                    'description': item.get('description', ''),
                    'fields': item.get('fields', []),
                    'input_methods': item.get('input_methods', {'file_upload': True, 's3_sync': False}),
                    'custom_prompt': item.get('custom_prompt', '')
                }
                apps.append(app_data)
            
            logger.info(f"DynamoDB から {len(apps)} 個のアプリスキーマを読み込みました")
            return {"apps": apps}
        else:
            logger.warning("DynamoDB からスキーマを取得できませんでした")
            # スキーマが見つからない場合は空の配列を返す
            return {"apps": []}
    
    except ClientError as e:
        logger.error(f"DynamoDB からのスキーマ取得エラー: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"スキーマ取得エラー: {str(e)}")
        raise


# グローバル変数を削除し、代わりに毎回DynamoDBから取得する関数を使用
def get_app_schemas():
    """
    アプリケーションスキーマをDynamoDBから取得する
    毎回呼び出されるたびに最新のデータを取得
    """
    return load_app_schemas()


def get_extraction_fields_for_app(app_name):
    """指定されたアプリ用の抽出フィールドを取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return {"fields": app["fields"]}

    logger.warning(f"App '{app_name}' not found in schemas")
    # アプリが見つからない場合は空のフィールドリストを返す
    return {"fields": []}


def get_field_names_for_app(app_name):
    """指定されたアプリの抽出フィールド名リストを取得（階層構造対応）"""
    fields = get_extraction_fields_for_app(app_name)["fields"]
    field_names = []
    
    def extract_field_names(fields, prefix=""):
        for field in fields:
            field_name = field["name"]
            full_name = f"{prefix}{field_name}" if prefix else field_name
            field_names.append(full_name)
            
            # map型の場合は再帰的に処理
            if field.get("type") == "map" and "fields" in field:
                extract_field_names(field["fields"], f"{full_name}.")
            
            # list型の場合、itemsがmap型なら再帰的に処理
            if field.get("type") == "list" and "items" in field:
                items = field["items"]
                if items.get("type") == "map" and "fields" in items:
                    # リスト内の各項目のフィールド名を取得
                    for item_field in items["fields"]:
                        field_names.append(f"{full_name}.{item_field['name']}")
    
    extract_field_names(fields)
    return field_names


def get_app_display_name(app_name):
    """アプリの表示名を取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return app.get("display_name", app_name)
    return app_name


def get_app_input_methods(app_name):
    """アプリの入力方法設定を取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            input_methods = app.get("input_methods", {"file_upload": True, "s3_sync": False})
            # 後方互換性のために、s3_bucket_pathがある場合はs3_uriに変換
            if "s3_bucket_path" in input_methods and not "s3_uri" in input_methods:
                bucket_name = os.environ.get("BUCKET_NAME", "my-ocr-bucket")
                path = input_methods["s3_bucket_path"]
                input_methods["s3_uri"] = f"s3://{bucket_name}/{path}"
            return input_methods
    # アプリが見つからない場合はデフォルト設定を返す
    return {"file_upload": True, "s3_sync": False}
    
def get_custom_prompt_for_app(app_name):
    """指定されたアプリ用のカスタムプロンプトを取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return app.get("custom_prompt", "")
    return ""


def update_app_schema(app_name, app_data):
    """
    アプリケーションスキーマを更新する
    """
    try:
        schemas_table_name = os.environ.get('SCHEMAS_TABLE_NAME')
        if not schemas_table_name:
            logger.error("SCHEMAS_TABLE_NAME 環境変数が設定されていません")
            return False
            
        schemas_table = dynamodb.Table(schemas_table_name)
        
        # 現在の日時を取得
        from datetime import datetime
        current_time = datetime.now().isoformat()
        
        # 既存のレコードを取得して created_at を保持
        try:
            existing_response = schemas_table.get_item(
                Key={
                    'schema_type': 'app',
                    'name': app_name
                }
            )
            created_at = existing_response.get('Item', {}).get('created_at', current_time)
        except:
            created_at = current_time
        
        # 新しい構造でスキーマを保存
        item = {
            'schema_type': 'app',
            'name': app_name,
            'display_name': app_data.get('display_name', app_name),
            'description': app_data.get('description', ''),
            'fields': app_data.get('fields', []),
            'input_methods': app_data.get('input_methods', {'file_upload': True, 's3_sync': False}),
            'created_at': created_at,
            'updated_at': current_time
        }
        
        # custom_prompt がある場合のみ追加
        if 'custom_prompt' in app_data and app_data['custom_prompt']:
            item['custom_prompt'] = app_data['custom_prompt']
        
        schemas_table.put_item(Item=item)
        
        logger.info(f"スキーマを更新しました: {app_name}")
        return True
        
    except Exception as e:
        logger.error(f"スキーマ更新エラー: {str(e)}")
        return False


def delete_app_schema(app_name):
    """
    アプリケーションスキーマを削除する
    """
    try:
        schemas_table_name = os.environ.get('SCHEMAS_TABLE_NAME')
        if not schemas_table_name:
            logger.error("SCHEMAS_TABLE_NAME 環境変数が設定されていません")
            return False
            
        schemas_table = dynamodb.Table(schemas_table_name)
        
        # スキーマを削除
        schemas_table.delete_item(
            Key={
                'schema_type': 'app',
                'name': app_name
            }
        )
        
        logger.info(f"スキーマを削除しました: {app_name}")
        return True
        
    except Exception as e:
        logger.error(f"スキーマ削除エラー: {str(e)}")
        return False

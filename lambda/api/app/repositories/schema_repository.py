import logging
import os
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from clients import dynamodb_resource
from exceptions import ConflictError

logger = logging.getLogger(__name__)


def _get_schemas_table():
    """SchemasTable のリソースを取得"""
    table_name = os.environ.get('SCHEMAS_TABLE_NAME')
    if not table_name:
        logger.error("SCHEMAS_TABLE_NAME 環境変数が設定されていません")
        raise RuntimeError("SCHEMAS_TABLE_NAME environment variable is not set")
    return dynamodb_resource.Table(table_name)


def load_app_schemas():
    """
    アプリケーションスキーマを取得する
    DynamoDB から全てのアプリスキーマを取得
    取得できない場合はエラーを返す
    """
    try:
        logger.info("DynamoDB からスキーマを取得します")
        schemas_table = _get_schemas_table()
        
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
                    'custom_prompt': item.get('custom_prompt', ''),
                    'agent_enabled': item.get('agent_enabled', False),
                    'agent_auto_run': item.get('agent_auto_run', False),
                    'sample_image_s3_key': item.get('sample_image_s3_key'),
                    'sample_image_filename': item.get('sample_image_filename'),
                    'schema_instructions': item.get('schema_instructions', ''),
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


def get_app_schema(app_name):
    """指定されたアプリのスキーマを取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return app
    
    logger.warning(f"App '{app_name}' not found in schemas")
    return None


def get_extraction_fields_for_app(app_name):
    """指定されたアプリ用の抽出フィールドを取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return {"fields": app["fields"]}

    logger.warning(f"App '{app_name}' not found in schemas")
    # アプリが見つからない場合は空のフィールドリストを返す
    return {"fields": []}


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




def create_app_schema(app_name, app_data):
    """
    アプリケーションスキーマを新規作成する（同名が既に存在する場合は ClientError を raise）
    """
    try:
        schemas_table = _get_schemas_table()
        current_time = datetime.now().isoformat()

        item = {
            'schema_type': 'app',
            'name': app_name,
            'display_name': app_data.get('display_name', app_name),
            'description': app_data.get('description', ''),
            'fields': app_data.get('fields', []),
            'input_methods': app_data.get('input_methods', {'file_upload': True, 's3_sync': False}),
            'agent_enabled': app_data.get('agent_enabled', False),
            'agent_auto_run': app_data.get('agent_auto_run', False),
            'created_at': current_time,
            'updated_at': current_time
        }

        if 'custom_prompt' in app_data and app_data['custom_prompt']:
            item['custom_prompt'] = app_data['custom_prompt']

        # サンプル画像 (スキーマ生成に使った画像) の紐付け
        if app_data.get('sample_image_s3_key'):
            item['sample_image_s3_key'] = app_data['sample_image_s3_key']
            item['sample_image_filename'] = app_data.get('sample_image_filename', '')

        # スキーマ生成に使った指示プロンプト
        if app_data.get('schema_instructions') is not None:
            item['schema_instructions'] = app_data['schema_instructions']

        schemas_table.put_item(
            Item=item,
            ConditionExpression='attribute_not_exists(schema_type) AND attribute_not_exists(#n)',
            ExpressionAttributeNames={'#n': 'name'}
        )

        logger.info(f"スキーマを新規作成しました: {app_name}")
        return True

    except ClientError as e:
        if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
            raise ConflictError(f"アプリ名 '{app_name}' は既に使用されています")
        raise
    except Exception as e:
        logger.error(f"スキーマ作成エラー: {str(e)}")
        raise


def update_app_schema(app_name, app_data):
    """
    アプリケーションスキーマを更新する
    """
    try:
        schemas_table = _get_schemas_table()
        
        # 現在の日時を取得
        current_time = datetime.now().isoformat()
        
        # 既存のレコードを取得して created_at, custom_prompt を保持
        existing_item = {}
        try:
            existing_response = schemas_table.get_item(
                Key={
                    'schema_type': 'app',
                    'name': app_name
                }
            )
            existing_item = existing_response.get('Item', {})
        except Exception:
            pass
        created_at = existing_item.get('created_at', current_time)
        
        # 新しい構造でスキーマを保存
        item = {
            'schema_type': 'app',
            'name': app_name,
            'display_name': app_data.get('display_name', app_name),
            'description': app_data.get('description', ''),
            'fields': app_data.get('fields', []),
            'input_methods': app_data.get('input_methods', {'file_upload': True, 's3_sync': False}),
            'agent_enabled': app_data.get('agent_enabled', False),
            'agent_auto_run': app_data.get('agent_auto_run', False),
            'created_at': created_at,
            'updated_at': current_time
        }

        # custom_prompt: リクエストに含まれていればそれを使い、なければ既存値を保持
        if 'custom_prompt' in app_data and app_data['custom_prompt']:
            item['custom_prompt'] = app_data['custom_prompt']
        elif existing_item.get('custom_prompt'):
            item['custom_prompt'] = existing_item['custom_prompt']

        # sample_image: リクエストに含まれていれば差し替え、なければ既存値を保持
        # (画像を変更しない編集で紐付けが消えないようにするため)
        if app_data.get('sample_image_s3_key'):
            item['sample_image_s3_key'] = app_data['sample_image_s3_key']
            item['sample_image_filename'] = app_data.get('sample_image_filename', '')
        elif existing_item.get('sample_image_s3_key'):
            item['sample_image_s3_key'] = existing_item['sample_image_s3_key']
            item['sample_image_filename'] = existing_item.get('sample_image_filename', '')

        # schema_instructions: None でなければ差し替え (空文字はクリア扱い)、None なら既存値保持
        if app_data.get('schema_instructions') is not None:
            item['schema_instructions'] = app_data['schema_instructions']
        elif existing_item.get('schema_instructions'):
            item['schema_instructions'] = existing_item['schema_instructions']
        
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
        schemas_table = _get_schemas_table()
        
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

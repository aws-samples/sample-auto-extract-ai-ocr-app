import json
import logging
import os
import boto3
import re
import imghdr
from botocore.exceptions import ClientError

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


def get_app_schema(app_name):
    """指定されたアプリのスキーマを取得"""
    app_schemas = get_app_schemas()
    for app in app_schemas.get("apps", []):
        if app["name"] == app_name:
            return app
    
    logger.warning(f"App '{app_name}' not found in schemas")
    # アプリが見つからない場合はデフォルトの空スキーマを返す
    return {"name": app_name, "fields": []}


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

def generate_schema_fields_from_image(image_data, instructions=None):
    """
    画像からスキーマのフィールド部分のみを生成する関数

    Args:
        image_data (bytes): 画像データ
        instructions (str, optional): スキーマ生成の指示

    Returns:
        dict: 生成されたフィールド定義 {"fields": [...]} の形式
    """
    try:
        # extraction.pyから必要な関数をimport
        from utils.bedrock import call_bedrock, parse_converse_response
        
        # 画像のMIMEタイプを判定
        image_type = imghdr.what(None, h=image_data)
        if not image_type:
            image_type = 'jpeg'  # デフォルト
        content_type = f"image/{image_type}"

        # システムプロンプト
        system_prompts = [{
            "text": """
            あなたはOCR処理された文書から情報抽出のためのフィールド定義を生成するアシスタントです。
            ユーザーが提供する画像を分析し、その文書タイプに適したフィールド構造を生成してください。
            """
        }]

        # フィールド定義の説明
        fields_explanation = """
        フィールド型は主に以下の3種類があります：
        
        1. string型: 単一の文字列値を格納するフィールド（日付、番号、名前など）
        2. map型: 複数の関連フィールドをグループ化するための階層構造（会社情報、住所情報など）
        3. list型: 表形式のデータなど、同じ構造を持つ複数の項目を格納するためのフィールド（明細行、商品リストなど）
        
        基本的には、単一の値は string 型、関連する複数の値をグループ化する場合は map 型、
        表形式のデータ（明細行など）は list 型を使用してください。
        """

        # フィールド定義の例
        fields_definition = """
        フィールドは以下の構造に従って定義してください：
        
        {
          "fields": [
            {
              "name": "フィールド名（英数字、アンダースコア）",
              "display_name": "フィールド表示名（日本語可）",
              "type": "string | map | list"  // フィールドの型
            },
            // map型の場合は子フィールドを定義
            {
              "name": "company_info",
              "display_name": "会社情報",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "会社名",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                }
              ]
            },
            // list型の場合はitemsを定義（表形式のデータ向け）
            {
              "name": "items",
              "display_name": "明細",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品目",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  }
                ]
              }
            }
          ]
        }
        """

        # 実際のフィールド例
        fields_examples = """
        以下は実際のフィールド定義例です：
        
        1. 請求書処理アプリケーションのフィールド例：
        {
          "fields": [
            {
              "name": "invoice_date",
              "display_name": "請求日",
              "type": "string"
            },
            {
              "name": "invoice_number",
              "display_name": "請求番号",
              "type": "string"
            },
            {
              "name": "company_info",
              "display_name": "会社情報",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "会社名",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "items",
              "display_name": "明細",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品目",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  },
                  {
                    "name": "unit_price",
                    "display_name": "単価",
                    "type": "string"
                  },
                  {
                    "name": "amount",
                    "display_name": "金額",
                    "type": "string"
                  }
                ]
              }
            },
            {
              "name": "total_amount",
              "display_name": "合計金額",
              "type": "string"
            },
            {
              "name": "tax_amount",
              "display_name": "消費税",
              "type": "string"
            }
          ]
        }
        
        2. 輸送伝票処理アプリケーションのフィールド例：
        {
          "fields": [
            {
              "name": "shipping_date",
              "display_name": "出荷日",
              "type": "string"
            },
            {
              "name": "tracking_number",
              "display_name": "追跡番号",
              "type": "string"
            },
            {
              "name": "sender",
              "display_name": "送り主",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "名前",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "receiver",
              "display_name": "受取人",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "名前",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "items",
              "display_name": "荷物情報",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品名",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  },
                  {
                    "name": "weight",
                    "display_name": "重量",
                    "type": "string"
                  }
                ]
              }
            }
          ]
        }
        """

        # ユーザーからの指示があれば追加
        instruction_text = ""
        if instructions:
            instruction_text = f"""
            ユーザーからの追加指示：
            {instructions}
            """

        # ユーザーメッセージ
        user_message = {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": content_type.split('/')[1],
                        "source": {
                            "bytes": image_data
                        }
                    }
                },
                {
                    "text": f"""
                    この画像に写っている文書を分析し、情報抽出に適したフィールド定義を生成してください。
                    
                    {instruction_text}
                    
                    {fields_explanation}
                    
                    {fields_definition}
                    
                    {fields_examples}
                    
                    以下の点に注意してフィールドを生成してください：
                    1. 文書の種類（請求書、納品書、輸送伝票など）を特定し、適切なフィールド構造を設計する
                    2. 文書から抽出可能な全ての重要情報をフィールドとして定義する
                    3. 階層構造が必要な場合は適切にmap型を使用する
                    4. 表形式のデータ（明細行など）がある場合はlist型を使用する
                    5. フィールド名は英数字とアンダースコアのみを使用し、日本語は表示名に使用する
                    
                    必ず {{"fields": [...]}} の形式でJSONを出力してください。説明や補足は不要です。
                    """
                }
            ]
        }

        messages = [user_message]

        # Bedrock APIを呼び出し
        response = call_bedrock(messages, system_prompts)

        # レスポンスからテキストを抽出
        fields_text = parse_converse_response(response)

        # JSONテキストからフィールド定義を抽出
        json_match = re.search(r'```json\s*(.*?)\s*```',
                               fields_text, re.DOTALL)
        if json_match:
            fields_json = json_match.group(1)
        else:
            fields_json = fields_text

        # JSONをパース
        try:
            schema = json.loads(fields_json)

            # スキーマが {"fields": [...]} の形式になっているか確認
            if "fields" not in schema:
                # fieldsキーがない場合は、配列を受け取ったと仮定して包む
                if isinstance(schema, list):
                    schema = {"fields": schema}
                else:
                    # それ以外の場合はエラー
                    raise ValueError("生成されたスキーマに 'fields' キーがありません")

            return schema
        except json.JSONDecodeError as e:
            logger.error(f"フィールド定義のJSONパースエラー: {str(e)}")
            raise ValueError(f"生成されたフィールド定義が有効なJSONではありません: {fields_json}")

    except Exception as e:
        logger.error(f"フィールド生成エラー: {str(e)}")
        raise

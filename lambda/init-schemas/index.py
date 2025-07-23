import os
import json
import boto3
import logging
from datetime import datetime

# ロガーの設定
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# DynamoDB クライアントの初期化
dynamodb = boto3.resource('dynamodb')
schemas_table_name = os.environ.get('SCHEMAS_TABLE_NAME')

def handler(event, context):
    """
    デフォルトスキーマを DynamoDB に初期化するハンドラー関数
    各アプリを個別のレコードとして保存します
    """
    try:
        logger.info(f"Event: {json.dumps(event)}")
        
        # スキーマテーブル名の取得
        if not schemas_table_name:
            raise ValueError("SCHEMAS_TABLE_NAME environment variable is not set")
        
        schemas_table = dynamodb.Table(schemas_table_name)
        
        # スキーマファイルの読み込み
        # S3からスキーマファイルを取得する場合
        s3_bucket = event.get('s3_bucket') or os.environ.get('DEFAULT_S3_BUCKET')
        s3_key = event.get('s3_key') or os.environ.get('DEFAULT_S3_KEY')
        
        logger.info(f"S3バケット: {s3_bucket}, S3キー: {s3_key}")
        
        if s3_bucket and s3_key:
            # S3 からスキーマファイルを取得
            logger.info(f"S3からスキーマファイルを取得します: {s3_bucket}/{s3_key}")
            s3_client = boto3.client('s3')
            try:
                response = s3_client.get_object(Bucket=s3_bucket, Key=s3_key)
                schema_content = response['Body'].read().decode('utf-8')
                schema_data = json.loads(schema_content)
                logger.info(f"S3からスキーマファイルを正常に読み込みました: {s3_bucket}/{s3_key}")
            except Exception as e:
                logger.error(f"S3からのスキーマファイル読み込みエラー: {str(e)}")
                raise ValueError(f"S3からのスキーマファイル読み込みエラー: {str(e)}")
        else:
            # デフォルトスキーマを使用
            logger.info("S3パラメータが指定されていないため、デフォルトスキーマを使用します")
            schema_data = {
                "apps": [
                    {
                        "name": "shipping_ocr",
                        "display_name": "運送発注書",
                        "description": "運送発注書の情報を抽出します",
                        "input_methods": {
                            "file_upload": True,
                            "s3_sync": True,
                            "s3_uri": "s3://my-ocr-bucket/shipping-documents/"
                        },
                        "fields": [
                            {
                                "name": "order_date",
                                "display_name": "注文日",
                                "type": "string"
                            },
                            {
                                "name": "operation_info",
                                "display_name": "運行情報",
                                "type": "map",
                                "fields": [
                                    {
                                        "name": "contract_work",
                                        "display_name": "委託業務内容",
                                        "type": "string"
                                    },
                                    {
                                        "name": "operation_date",
                                        "display_name": "運行日",
                                        "type": "string"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            }
        
        # 現在の日時を取得
        current_time = datetime.now().isoformat()
        
        # 各アプリを個別のレコードとして保存
        apps = schema_data.get("apps", [])
        logger.info(f"{len(apps)}個のアプリスキーマを処理します")
        
        for app in apps:
            app_name = app.get("name")
            if not app_name:
                logger.warning(f"アプリ名が指定されていないスキーマをスキップします: {app}")
                continue
                
            # 既存のスキーマを確認
            response = schemas_table.get_item(
                Key={
                    'schema_type': 'app',
                    'name': app_name
                }
            )
            
            # 既存のスキーマがある場合はスキップ
            if 'Item' in response:
                logger.info(f"既存のスキーマが見つかりました。スキップします: {app_name}")
                continue
            
            # アプリスキーマを DynamoDB に投入（新しい構造）
            logger.info(f"スキーマをDynamoDBに保存します: {app_name}")
            
            item = {
                'schema_type': 'app',
                'name': app_name,
                'display_name': app.get('display_name', app_name),
                'description': app.get('description', ''),
                'fields': app.get('fields', []),
                'input_methods': app.get('input_methods', {'file_upload': True, 's3_sync': False}),
                'created_at': current_time,
                'updated_at': current_time
            }
            
            # custom_prompt がある場合のみ追加
            if 'custom_prompt' in app and app['custom_prompt']:
                item['custom_prompt'] = app['custom_prompt']
            
            schemas_table.put_item(Item=item)
            logger.info(f"スキーマの保存が完了しました: {app_name}")
        
        logger.info(f"全てのスキーマの初期化が完了しました: {len(apps)}個")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f'Schema initialization completed for {len(apps)} apps')
        }
    
    except Exception as e:
        logger.error(f"スキーマの初期化中にエラーが発生しました: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f'Error initializing schema: {str(e)}')
        }

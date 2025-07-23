import os
import logging
import boto3
from boto3.dynamodb.conditions import Key
from botocore.exceptions import ClientError
from fastapi import HTTPException
from datetime import datetime
import uuid
import json

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DynamoDB クライアントの初期化
dynamodb = boto3.resource('dynamodb')


def get_images_table():
    """
    画像テーブルのリソースを取得する

    Returns:
        boto3.resources.factory.dynamodb.Table: DynamoDB テーブルリソース
    """
    table_name = os.getenv("IMAGES_TABLE_NAME")
    if not table_name:
        logger.error("IMAGES_TABLE_NAME 環境変数が設定されていません")
        raise HTTPException(
            status_code=500, detail="Database configuration error")

    return dynamodb.Table(table_name)


def get_jobs_table():
    """
    ジョブテーブルのリソースを取得する

    Returns:
        boto3.resources.factory.dynamodb.Table: DynamoDB テーブルリソース
    """
    table_name = os.getenv("JOBS_TABLE_NAME")
    if not table_name:
        logger.error("JOBS_TABLE_NAME 環境変数が設定されていません")
        raise HTTPException(
            status_code=500, detail="Database configuration error")

    return dynamodb.Table(table_name)


def get_db_info():
    """
    データベース接続情報を取得する

    Returns:
        dict: データベース接続情報
    """
    return {
        "type": "DynamoDB",
        "images_table": os.getenv("IMAGES_TABLE_NAME"),
        "jobs_table": os.getenv("JOBS_TABLE_NAME"),
        "region": os.getenv("AWS_REGION", "us-east-1")
    }


def get_db_version():
    """
    データベースのバージョン情報を取得する

    Returns:
        dict: データベース情報
    """
    db_info = get_db_info()

    try:
        # DynamoDB のサービス情報を取得
        client = boto3.client('dynamodb')
        tables = client.list_tables()
        db_info["tables"] = tables.get("TableNames", [])
        db_info["status"] = "connected"
    except Exception as e:
        db_info["error"] = str(e)
        db_info["status"] = "error"

    return {"db_info": db_info}


def create_image_record(image_id, filename, s3_key, app_name="default", status="pending", converted_s3_key=None):
    """
    画像レコードを作成する

    Args:
        image_id (str): 画像ID
        filename (str): ファイル名
        s3_key (str): 元ファイルのS3キー
        app_name (str): アプリケーション名
        status (str): 画像の処理ステータス
        converted_s3_key (str, optional): 変換後画像のS3キー

    Returns:
        str: 作成された画像のID
    """
    if not image_id:
        image_id = str(uuid.uuid4())

    table = get_images_table()
    current_time = datetime.now().isoformat()

    try:
        item = {
            "id": image_id,
            "filename": filename,
            "original_s3_key": s3_key,
            "s3_key": s3_key,  # 後方互換性のために残す
            "upload_time": current_time,
            "status": status,
            "app_name": app_name
        }

        # 変換後のS3キーがある場合は追加
        if converted_s3_key:
            item["converted_s3_key"] = converted_s3_key
            item["s3_key"] = converted_s3_key  # 変換後のキーを優先

        table.put_item(Item=item)
        return image_id
    except Exception as e:
        logger.error(f"画像レコード作成エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def get_images(app_name=None):
    """
    画像一覧を取得する

    Args:
        app_name (str, optional): アプリケーション名でフィルタリング

    Returns:
        list: 画像レコードのリスト
    """
    table = get_images_table()

    try:
        if app_name:
            # GSIを使用してアプリ名でフィルタリング
            response = table.query(
                IndexName="AppNameIndex",
                KeyConditionExpression=Key('app_name').eq(app_name),
                ScanIndexForward=False  # 降順（新しい順）
            )
        else:
            # 全件取得（注意: 大規模データの場合はページネーションが必要）
            response = table.scan()

        images = []
        for item in response.get('Items', []):
            images.append({
                "id": item.get("id"),
                "name": item.get("filename"),
                "s3_key": item.get("s3_key"),
                "uploadTime": item.get("upload_time"),
                "status": item.get("status"),
                "jobId": item.get("job_id"),
                "appName": item.get("app_name")
            })

        return images
    except Exception as e:
        logger.error(f"画像一覧取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_image_status(image_id, status, job_id=None):
    """
    画像ステータスを更新する

    Args:
        image_id (str): 画像ID
        status (str): 新しいステータス
        job_id (str, optional): ジョブID
    """
    table = get_images_table()

    update_expression = "SET #status = :status"
    expression_attribute_names = {"#status": "status"}
    expression_attribute_values = {":status": status}

    if job_id:
        update_expression += ", job_id = :job_id"
        expression_attribute_values[":job_id"] = job_id

    try:
        table.update_item(
            Key={"id": image_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )
    except Exception as e:
        logger.error(f"画像ステータス更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_ocr_result(image_id, ocr_result, extraction_status="processing"):
    """
    OCR結果を更新する

    Args:
        image_id (str): 画像ID
        ocr_result (dict): OCR結果
        extraction_status (str): 抽出ステータス
    """
    table = get_images_table()

    try:
        table.update_item(
            Key={"id": image_id},
            UpdateExpression="SET ocr_result = :ocr_result, extraction_status = :extraction_status",
            ExpressionAttributeValues={
                ":ocr_result": ocr_result,
                ":extraction_status": extraction_status
            }
        )
        logger.info(f"OCR結果を更新しました: {image_id}")
    except Exception as e:
        logger.error(f"OCR結果更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_extracted_info(image_id, extracted_info, extraction_mapping, status="completed"):
    """
    抽出情報を更新する（Map型で保存）

    Args:
        image_id (str): 画像ID
        extracted_info (dict): 抽出情報
        extraction_mapping (dict): 抽出マッピング
        status (str): 抽出ステータス
    """
    table = get_images_table()

    try:
        table.update_item(
            Key={"id": image_id},
            UpdateExpression="SET extracted_info = :extracted_info, extraction_mapping = :extraction_mapping, extraction_status = :status",
            ExpressionAttributeValues={
                ":extracted_info": extracted_info,
                ":extraction_mapping": extraction_mapping,
                ":status": status
            }
        )
        logger.info(f"抽出情報を更新しました: {image_id}")
    except Exception as e:
        logger.error(f"抽出情報更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def get_image(image_id):
    """
    画像情報を取得する

    Args:
        image_id (str): 画像ID

    Returns:
        dict: 画像情報
    """
    table = get_images_table()

    try:
        response = table.get_item(Key={"id": image_id})
        item = response.get("Item")

        if not item:
            raise HTTPException(status_code=404, detail="Image not found")

        return item
    except ClientError as e:
        logger.error(f"画像取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def create_job(job_id=None, status="processing"):
    """
    ジョブを作成する

    Args:
        job_id (str, optional): ジョブID
        status (str): ジョブステータス

    Returns:
        str: 作成されたジョブのID
    """
    if not job_id:
        job_id = str(uuid.uuid4())

    table = get_jobs_table()
    current_time = datetime.now().isoformat()

    try:
        item = {
            "id": job_id,
            "status": status,
            "created_at": current_time,
            "updated_at": current_time
        }

        table.put_item(Item=item)
        return job_id
    except Exception as e:
        logger.error(f"ジョブ作成エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_job_status(job_id, status):
    """
    ジョブステータスを更新する

    Args:
        job_id (str): ジョブID
        status (str): 新しいステータス
    """
    table = get_jobs_table()
    current_time = datetime.now().isoformat()

    try:
        table.update_item(
            Key={"id": job_id},
            UpdateExpression="SET #status = :status, updated_at = :updated_at",
            ExpressionAttributeNames={"#status": "status"},
            ExpressionAttributeValues={
                ":status": status,
                ":updated_at": current_time
            }
        )
    except Exception as e:
        logger.error(f"ジョブステータス更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def get_job(job_id):
    """
    ジョブ情報を取得する

    Args:
        job_id (str): ジョブID

    Returns:
        dict: ジョブ情報
    """
    table = get_jobs_table()

    try:
        response = table.get_item(Key={"id": job_id})
        item = response.get("Item")

        if not item:
            raise HTTPException(status_code=404, detail="Job not found")

        return item
    except ClientError as e:
        logger.error(f"ジョブ取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def get_images_by_job_id(job_id):
    """
    ジョブIDに関連する画像を取得する

    Args:
        job_id (str): ジョブID

    Returns:
        list: 画像リスト
    """
    table = get_images_table()

    try:
        # job_id でフィルタリングするにはスキャンが必要
        # 頻繁に使用する場合は GSI を追加すべき
        response = table.scan(
            FilterExpression=Key('job_id').eq(job_id)
        )

        images = []
        for item in response.get('Items', []):
            images.append({
                "id": item.get("id"),
                "filename": item.get("filename"),
                "status": item.get("status")
            })

        return images
    except Exception as e:
        logger.error(f"ジョブ関連画像取得エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_converted_image(image_id, converted_s3_key, status=None, original_size=None, resized_size=None):
    """
    変換後の画像情報を更新する

    Args:
        image_id (str): 画像ID
        converted_s3_key (str): 変換後画像のS3キー
        status (str, optional): 更新するステータス
        original_size (tuple, optional): 元の画像サイズ (width, height)
        resized_size (tuple, optional): リサイズ後の画像サイズ (width, height)

    Returns:
        bool: 更新が成功したかどうか
    """
    table = get_images_table()

    try:
        update_expression = "SET converted_s3_key = :converted_s3_key, s3_key = :converted_s3_key"
        expression_values = {
            ":converted_s3_key": converted_s3_key
        }

        if status:
            update_expression += ", #status = :status"
            expression_values[":status"] = status
            
        if original_size:
            update_expression += ", original_size = :original_size"
            expression_values[":original_size"] = {
                "width": original_size[0],
                "height": original_size[1]
            }
            
        if resized_size:
            update_expression += ", resized_size = :resized_size"
            expression_values[":resized_size"] = {
                "width": resized_size[0],
                "height": resized_size[1]
            }

        expression_names = {}
        if status:
            expression_names["#status"] = "status"

        response = table.update_item(
            Key={"id": image_id},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values,
            ExpressionAttributeNames=expression_names if expression_names else {},
            ReturnValues="UPDATED_NEW"
        )

        logger.info(f"変換後画像情報を更新しました: {image_id}, {converted_s3_key}")
        return True
    except Exception as e:
        logger.error(f"変換後画像情報の更新エラー: {str(e)}")
        return False


def delete_images_by_app_name(app_name: str):
    """
    指定されたアプリ名に関連する全ての画像データを削除する
    
    Args:
        app_name (str): アプリ名
        
    Returns:
        bool: 削除が成功したかどうか
    """
    try:
        table = get_images_table()
        
        # GSIを使用してアプリ名でクエリ
        response = table.query(
            IndexName="AppNameIndex",
            KeyConditionExpression=Key('app_name').eq(app_name)
        )
        
        # 取得した画像を削除
        deleted_count = 0
        for item in response.get('Items', []):
            table.delete_item(Key={'id': item['id']})
            deleted_count += 1
            
        logger.info(f"アプリ '{app_name}' に関連する {deleted_count} 件の画像データを削除しました")
        return True
        
    except Exception as e:
        logger.error(f"画像データ削除エラー (app_name: {app_name}): {str(e)}")
        return False


def delete_jobs_by_app_name(app_name: str):
    """
    指定されたアプリ名に関連する全てのジョブデータを削除する
    
    Args:
        app_name (str): アプリ名
        
    Returns:
        bool: 削除が成功したかどうか
    """
    try:
        table = get_jobs_table()
        
        # アプリ名でフィルタリングしてスキャン
        response = table.scan(
            FilterExpression=Key('app_name').eq(app_name)
        )
        
        # 取得したジョブを削除
        deleted_count = 0
        for item in response.get('Items', []):
            table.delete_item(Key={'id': item['id']})
            deleted_count += 1
            
        logger.info(f"アプリ '{app_name}' に関連する {deleted_count} 件のジョブデータを削除しました")
        return True
        
    except Exception as e:
        logger.error(f"ジョブデータ削除エラー (app_name: {app_name}): {str(e)}")
        return False

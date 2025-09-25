from clients import dynamodb_resource
import logging
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from fastapi import HTTPException
from datetime import datetime
import uuid
from config import settings

import logging

logger = logging.getLogger(__name__)


def get_images_table():
    """
    画像テーブルのリソースを取得する

    Returns:
        boto3.resources.factory.dynamodb_resource.Table: DynamoDB テーブルリソース
    """
    table_name = settings.IMAGES_TABLE_NAME
    if not table_name:
        logger.error("IMAGES_TABLE_NAME 環境変数が設定されていません")
        raise HTTPException(
            status_code=500, detail="Database configuration error")

    return dynamodb_resource.Table(table_name)


def get_jobs_table():
    """
    ジョブテーブルのリソースを取得する

    Returns:
        boto3.resources.factory.dynamodb_resource.Table: DynamoDB テーブルリソース
    """
    table_name = settings.JOBS_TABLE_NAME
    if not table_name:
        logger.error("JOBS_TABLE_NAME 環境変数が設定されていません")
        raise HTTPException(
            status_code=500, detail="Database configuration error")

    return dynamodb_resource.Table(table_name)


def create_image_record(image_id, filename, s3_key, app_name="default", status="pending", converted_s3_key=None,
                        page_processing_mode="combined", total_pages=None, page_number=None, parent_document_id=None):
    """
    画像レコードを作成する

    Args:
        image_id (str): 画像ID
        filename (str): ファイル名
        s3_key (str): 元ファイルのS3キー
        app_name (str): アプリケーション名
        status (str): 画像の処理ステータス
        converted_s3_key (str, optional): 変換後画像のS3キー
        page_processing_mode (str): ページ処理モード ("combined" | "individual")
        total_pages (int, optional): 総ページ数
        page_number (int, optional): ページ番号（個別処理の場合）
        parent_document_id (str, optional): 親ドキュメントID（個別処理の場合）

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
            "s3_key": s3_key,
            "upload_time": current_time,
            "status": status,
            "app_name": app_name,
            "page_processing_mode": page_processing_mode
        }

        # ページ関連の情報を追加
        if total_pages is not None:
            item["total_pages"] = total_pages
        if page_number is not None:
            item["page_number"] = page_number
        if parent_document_id is not None:
            item["parent_document_id"] = parent_document_id

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
                                 指定時はGSI(AppNameIndex)でquery実行
                                 未指定時はscanで全件取得（1MB制限あり）

    Returns:
        list: 画像レコードのリスト

    注意:
        app_name未指定時はDynamoDB scanを使用するため、1MBのデータサイズ制限があります。
        大量のレコードがある場合、全てのデータを取得できない可能性があります。
        本番環境では必ずapp_nameを指定してGSI経由でのquery使用を推奨します。
    """
    table = get_images_table()

    try:
        if app_name:
            # GSI(AppNameIndex)を使用してアプリ名でフィルタリング
            # queryは効率的で1MB制限の影響を受けにくい
            response = table.query(
                IndexName="AppNameIndex",
                KeyConditionExpression=Key('app_name').eq(app_name),
                ScanIndexForward=False  # 降順（新しい順）
            )
            logger.info(f"GSI経由でアプリ '{app_name}' の画像を取得")
        else:
            # 全件取得（警告: DynamoDB scanは1MB制限があり、大規模データでは不完全な結果になる）
            response = table.scan()
            logger.warning("scanで全件取得中 - 大量データがある場合は一部のレコードが取得されない可能性があります")

        images = []
        for item in response.get('Items', []):
            images.append({
                "id": item.get("id"),
                "name": item.get("filename"),
                "s3_key": item.get("s3_key"),
                "uploadTime": item.get("upload_time"),
                "status": item.get("status"),
                "jobId": item.get("job_id"),
                "appName": item.get("app_name"),
                "pageProcessingMode": item.get("page_processing_mode"),
                "totalPages": item.get("total_pages"),
                "pageNumber": item.get("page_number"),
                "parentDocumentId": item.get("parent_document_id")
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

        # 親ドキュメントのステータス更新チェック
        image_data = get_image(image_id)
        if image_data and image_data.get("parent_document_id"):
            check_and_update_parent_status(image_data["parent_document_id"])

    except Exception as e:
        logger.error(f"画像ステータス更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_ocr_result(image_id: str, ocr_result: dict, extraction_status: str = "processing") -> None:
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

        # 親ドキュメントのステータス更新チェック
        image_data = get_image(image_id)
        if image_data and image_data.get("parent_document_id"):
            check_and_update_parent_status(image_data["parent_document_id"])

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


def update_converted_image(image_id, converted_s3_key, status=None, original_size=None, resized_size=None,
                           page_processing_mode=None, total_pages=None):
    """
    変換後の画像情報を更新する

    Args:
        image_id (str): 画像ID
        converted_s3_key (str | List[str]): 変換後画像のS3キー（単一またはリスト）
        status (str, optional): 更新するステータス
        original_size (tuple, optional): 元の画像サイズ (width, height)
        resized_size (tuple, optional): リサイズ後の画像サイズ (width, height)
        page_processing_mode (str, optional): ページ処理モード
        total_pages (int, optional): 総ページ数

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

        if page_processing_mode:
            update_expression += ", page_processing_mode = :page_processing_mode"
            expression_values[":page_processing_mode"] = page_processing_mode

        if total_pages is not None:
            update_expression += ", total_pages = :total_pages"
            expression_values[":total_pages"] = total_pages

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


def create_individual_page_record(page_id: str, parent_image_id: str, filename: str,
                                  converted_s3_key: str,
                                  page_number: int, total_pages: int, app_name: str,
                                  original_size: tuple, new_size: tuple):
    """
    個別ページのレコードを作成する

    Args:
        page_id (str): ページID
        parent_image_id (str): 親ドキュメントID
        filename (str): ファイル名
        original_s3_key (str): 元のS3キー
        converted_s3_key (str): 変換後のS3キー
        page_number (int): ページ番号
        total_pages (int): 総ページ数
        app_name (str): アプリケーション名
        original_size (tuple): 元のサイズ
        new_size (tuple): 新しいサイズ
    """
    table = get_images_table()
    current_time = datetime.now().isoformat()

    try:
        item = {
            "id": page_id,
            "filename": filename,
            "s3_key": converted_s3_key,
            "converted_s3_key": converted_s3_key,
            "upload_time": current_time,
            "status": "pending",
            "app_name": app_name,
            "page_processing_mode": "individual",
            "page_number": page_number,
            "total_pages": total_pages,
            "parent_document_id": parent_image_id,
            "original_size": list(original_size) if original_size else None,
            "new_size": list(new_size) if new_size else None
        }

        table.put_item(Item=item)
        logger.info(
            f"個別ページレコード作成完了: {page_id} (ページ {page_number}/{total_pages})")

    except Exception as e:
        logger.error(f"個別ページレコード作成エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def update_parent_document_status(parent_id: str, status: str, total_pages: int = None):
    """
    親ドキュメントのステータスを更新する

    Args:
        parent_id (str): 親ドキュメントID
        status (str): 新しいステータス
        total_pages (int, optional): 総ページ数
    """
    table = get_images_table()

    try:
        update_expression = "SET #status = :status"
        expression_attribute_names = {"#status": "status"}
        expression_attribute_values = {":status": status}

        if total_pages is not None:
            update_expression += ", total_pages = :total_pages"
            expression_attribute_values[":total_pages"] = total_pages

        table.update_item(
            Key={"id": parent_id},
            UpdateExpression=update_expression,
            ExpressionAttributeNames=expression_attribute_names,
            ExpressionAttributeValues=expression_attribute_values
        )

        logger.info(f"親ドキュメントステータス更新完了: {parent_id} -> {status}")

    except Exception as e:
        logger.error(f"親ドキュメントステータス更新エラー: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Database error: {str(e)}")


def get_children_by_parent_id(parent_id: str):
    """
    親ドキュメントIDから子ページ一覧を取得する

    Args:
        parent_id (str): 親ドキュメントID

    Returns:
        list: 子ページのリスト
    """
    table = get_images_table()

    try:
        response = table.scan(
            FilterExpression=Attr('parent_document_id').eq(parent_id)
        )
        return response.get('Items', [])
    except Exception as e:
        logger.error(f"子ページ取得エラー: {str(e)}")
        return []


def determine_parent_status(children):
    """
    子ページのステータスから親ドキュメントのステータスを判定する

    Args:
        children (list): 子ページのリスト

    Returns:
        str: 親ドキュメントのステータス
    """
    if not children:
        return "converting"

    statuses = [child.get("status") for child in children]

    if all(status == "completed" for status in statuses):
        return "completed"
    elif any(status == "failed" for status in statuses):
        return "failed"  # 一つでも失敗したら親も失敗
    elif any(status == "processing" for status in statuses):
        return "processing"
    else:
        return "converting"  # pending状態


def check_and_update_parent_status(parent_id: str):
    """
    親ドキュメントのステータスをチェックして更新する

    Args:
        parent_id (str): 親ドキュメントID
    """
    try:
        children = get_children_by_parent_id(parent_id)
        new_status = determine_parent_status(children)

        # 現在の親ステータスを取得
        parent_data = get_image(parent_id)
        current_status = parent_data.get("status")

        # ステータスが変更された場合のみ更新
        if current_status != new_status:
            update_parent_document_status(parent_id, new_status)
            logger.info(
                f"親ドキュメントステータス更新: {parent_id} {current_status} -> {new_status}")

    except Exception as e:
        logger.error(f"親ステータス更新エラー: {str(e)}")

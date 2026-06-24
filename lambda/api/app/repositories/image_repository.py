from clients import dynamodb_resource, s3_client
import logging
from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
from datetime import datetime
import uuid
from config import settings

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
        raise ValueError("IMAGES_TABLE_NAME environment variable is not set")

    return dynamodb_resource.Table(table_name)


def create_image_record(image_id, filename, s3_key, app_name="default", status="pending", converted_s3_key=None,
                        page_processing_mode="combined", total_pages=None, page_number=None, parent_document_id=None,
                        sync_source_path=None, uploaded_by=None):
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
        sync_source_path (str, optional): S3同期元のパス
        uploaded_by (str, optional): アップロードしたユーザーの cognito_sub

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

        # アップロードユーザー情報を追加
        if uploaded_by:
            item["uploaded_by"] = uploaded_by

        # ページ関連の情報を追加
        if total_pages is not None:
            item["total_pages"] = total_pages
        if page_number is not None:
            item["page_number"] = page_number
        if parent_document_id is not None:
            item["parent_document_id"] = parent_document_id

        # S3同期元パスがある場合は追加
        if sync_source_path is not None:
            item["sync_source_path"] = sync_source_path

        # 変換後のS3キーがある場合は追加
        if converted_s3_key:
            item["converted_s3_key"] = converted_s3_key
            item["s3_key"] = converted_s3_key  # 変換後のキーを優先

        table.put_item(Item=item)
        return image_id
    except Exception as e:
        logger.error(f"画像レコード作成エラー: {str(e)}")
        raise


def get_images(app_name=None, uploaded_by=None):
    """
    画像一覧を取得する

    Args:
        app_name (str, optional): アプリケーション名でフィルタリング
                                 指定時はGSI(AppNameIndex)でquery実行
                                 未指定時はscanで全件取得（1MB制限あり）
        uploaded_by (str, optional): アップロードユーザーの cognito_sub でフィルタリング
                                    指定時はGSI(UploadedByIndex)でquery実行

    Returns:
        list: 画像レコードのリスト

    注意:
        app_name/uploaded_by未指定時はDynamoDB scanを使用するため、1MBのデータサイズ制限があります。
        大量のレコードがある場合、全てのデータを取得できない可能性があります。
        本番環境では必ずapp_nameを指定してGSI経由でのquery使用を推奨します。
    """
    table = get_images_table()

    try:
        if uploaded_by and app_name:
            # UploadedByIndex で取得し、app_name でクライアント側フィルタ
            # （GSI の PK が uploaded_by のみのため、app_name は KeyCondition に含められない）
            response = table.query(
                IndexName="UploadedByIndex",
                KeyConditionExpression=Key('uploaded_by').eq(uploaded_by),
                ScanIndexForward=False  # 降順（新しい順）
            )
            items = [i for i in response.get('Items', []) if i.get('app_name') == app_name]
            logger.info("UploadedByIndex + app_name フィルタで画像を取得")
        elif uploaded_by:
            # UploadedByIndex で自分の画像のみ取得
            response = table.query(
                IndexName="UploadedByIndex",
                KeyConditionExpression=Key('uploaded_by').eq(uploaded_by),
                ScanIndexForward=False  # 降順（新しい順）
            )
            items = response.get('Items', [])
            logger.info("UploadedByIndex 経由でユーザーの画像を取得")
        elif app_name:
            # GSI(AppNameIndex)を使用してアプリ名でフィルタリング
            # queryは効率的で1MB制限の影響を受けにくい
            response = table.query(
                IndexName="AppNameIndex",
                KeyConditionExpression=Key('app_name').eq(app_name),
                ScanIndexForward=False  # 降順（新しい順）
            )
            items = response.get('Items', [])
            logger.info(f"GSI経由でアプリ '{app_name}' の画像を取得")
        else:
            # 全件取得（警告: DynamoDB scanは1MB制限があり、大規模データでは不完全な結果になる）
            response = table.scan()
            items = response.get('Items', [])
            logger.warning("scanで全件取得中 - 大量データがある場合は一部のレコードが取得されない可能性があります")

        # DynamoDB の Item をそのまま返す（camelCase 変換はサービス層/Pydantic で行う）
        return items
    except Exception as e:
        logger.error(f"画像一覧取得エラー: {str(e)}")
        raise


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
        raise


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

    except Exception as e:
        logger.error(f"OCR結果更新エラー: {str(e)}")
        raise


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
        raise


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
        return response.get("Item")
    except ClientError as e:
        logger.error(f"画像取得エラー: {str(e)}")
        raise


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

        update_kwargs = {
            "Key": {"id": image_id},
            "UpdateExpression": update_expression,
            "ExpressionAttributeValues": expression_values,
            "ReturnValues": "UPDATED_NEW",
        }
        if status:
            update_kwargs["ExpressionAttributeNames"] = {"#status": "status"}

        table.update_item(**update_kwargs)

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


def delete_image(image_id: str) -> bool:
    """
    画像レコードを削除する

    Args:
        image_id (str): 画像ID

    Returns:
        bool: 削除が成功したかどうか
    """
    try:
        table = get_images_table()
        table.delete_item(Key={'id': image_id})
        logger.info(f"Deleted image: {image_id}")
        return True
    except Exception as e:
        logger.error(f"Error deleting image {image_id}: {str(e)}")
        return False


def update_verification_status(image_id: str, verification_completed: bool, verified_by: str = None) -> None:
    """
    確認完了ステータスを更新する

    Args:
        image_id (str): 画像ID
        verification_completed (bool): 確認完了フラグ
        verified_by (str, optional): 確認者の cognito_sub
    """
    table = get_images_table()
    current_time = datetime.now().isoformat()
    
    try:
        table.update_item(
            Key={"id": image_id},
            UpdateExpression="SET verification_completed = :completed, verification_completed_at = :timestamp, verified_by = :verified_by",
            ExpressionAttributeValues={
                ":completed": verification_completed,
                ":timestamp": current_time if verification_completed else None,
                ":verified_by": verified_by if verification_completed else None
            }
        )
        logger.info(f"確認完了ステータスを更新: {image_id} -> {verification_completed} (by: {verified_by})")
    except Exception as e:
        logger.error(f"確認完了ステータス更新エラー: {str(e)}")
        raise


def update_agent_status(image_id: str, status: str, suggestions_count: int = None) -> None:
    """agent_status を更新する"""
    table = get_images_table()
    try:
        if suggestions_count is not None:
            table.update_item(
                Key={"id": image_id},
                UpdateExpression="SET agent_status = :s, agent_suggestions_count = :c",
                ExpressionAttributeValues={":s": status, ":c": suggestions_count},
            )
        else:
            table.update_item(
                Key={"id": image_id},
                UpdateExpression="SET agent_status = :s",
                ExpressionAttributeValues={":s": status},
            )
    except Exception as e:
        logger.error(f"agent_status 更新エラー: {str(e)}")
        raise


def reset_processing_status(image_id: str) -> None:
    """再処理のため extraction_status / agent_status / agent_suggestions_count をリセットする"""
    table = get_images_table()
    try:
        table.update_item(
            Key={"id": image_id},
            UpdateExpression=(
                "SET extraction_status = :proc, agent_status = :proc, "
                "agent_suggestions_count = :zero"
            ),
            ExpressionAttributeValues={":proc": "processing", ":zero": 0},
        )
    except Exception as e:
        logger.error(f"処理ステータスリセットエラー: {str(e)}")
        raise


def create_individual_page_record(page_id: str, parent_image_id: str, filename: str,
                                  converted_s3_key: str,
                                  page_number: int, total_pages: int, app_name: str,
                                  original_size: tuple, new_size: tuple, uploaded_by: str = None):
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

        if uploaded_by:
            item["uploaded_by"] = uploaded_by

        table.put_item(Item=item)
        logger.info(
            f"個別ページレコード作成完了: {page_id} (ページ {page_number}/{total_pages})")

    except Exception as e:
        logger.error(f"個別ページレコード作成エラー: {str(e)}")
        raise


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
        raise


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


def create_s3_sync_folder(app_name: str) -> None:
    """S3同期フォルダを作成する"""
    try:
        sync_bucket = settings.SYNC_BUCKET_NAME
        
        # フォルダ作成用の空オブジェクトを作成
        folder_key = f"{app_name}/"
        s3_client.put_object(
            Bucket=sync_bucket,
            Key=folder_key,
            Body=b'',
            ContentType='application/x-directory'
        )
        
        logger.info(f"Created S3 sync folder: s3://{sync_bucket}/{folder_key}")
    except Exception as e:
        logger.warning(f"Failed to create S3 sync folder for {app_name}: {str(e)}")
        # S3フォルダ作成失敗はアプリ作成を止めない


def get_images_by_sync_source(filename: str, sync_source_path: str, app_name: str = None):
    """
    ファイル名と同期元パスで既存ファイルを検索する

    Args:
        filename (str): 検索するファイル名
        sync_source_path (str): 同期元のS3パス
        app_name (str, optional): アプリ名でフィルタ

    Returns:
        list: マッチする画像のリスト
    """
    try:
        table = get_images_table()

        # ファイル名と同期元パスで検索
        filter_expression = Attr('filename').eq(filename) & Attr('sync_source_path').eq(sync_source_path)

        # アプリ名でのフィルタも追加
        if app_name:
            filter_expression = filter_expression & Attr('app_name').eq(app_name)

        response = table.scan(FilterExpression=filter_expression)
        return response.get('Items', [])

    except Exception as e:
        logger.error(f"同期元パスでの画像検索エラー: {str(e)}")
        return []

def get_existing_sync_sources(app_name: str) -> set[str]:
    """指定アプリの既存 sync_source_path のセットを返す（バッチ重複チェック用）

    1 回の scan で全件取得し、Python 側でセット化する。
    ファイルごとに scan するより効率的。

    警告: DynamoDB scan は 1MB 制限があり、大量データでは不完全な結果になる可能性がある。
    """
    try:
        table = get_images_table()
        filter_expr = Attr("app_name").eq(app_name) & Attr("sync_source_path").exists()
        response = table.scan(
            FilterExpression=filter_expr,
            ProjectionExpression="sync_source_path",
        )
        return {item["sync_source_path"] for item in response.get("Items", [])}
    except Exception as e:
        logger.error(f"同期元パス一括取得エラー: {str(e)}")
        return set()


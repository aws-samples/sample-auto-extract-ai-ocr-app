import uuid
import logging
import json
import base64
from typing import Optional, Dict, Any

from repositories import (
    get_images,
    get_image, update_ocr_result as db_update_ocr_result,
    update_image_status,
)
from clients import get_inference_component_status, trigger_endpoint_wakeup
from schemas import OcrResult, OcrResultResponse
from config import settings
from background import BackgroundTaskExtension
from clients import s3_client, sagemaker_runtime_client, sfn_client
from domains.ocr_engine import parse_ocr_response
from services.pdf_conversion_service import sync_parent_status
from utils.helpers import float_to_decimal

logger = logging.getLogger(__name__)


class EndpointNotReadyError(Exception):
    """OCR エンドポイントが起動中の場合のエラー"""
    pass


class OcrService:
    """OCR処理を管理するサービスクラス"""

    def __init__(self, background_task: Optional[BackgroundTaskExtension] = None):
        self.enable_ocr = settings.ENABLE_OCR
        self.background_task = background_task

    def get_endpoint_status(self) -> dict:
        """OCR エンドポイントの状態を返す"""
        if not self.enable_ocr:
            return {"ready": True, "status": "ocr_disabled"}
        return get_inference_component_status(settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)

    def _invoke_ocr(self, image_data: bytes) -> dict:
        """SageMaker OCR エンドポイントを呼び出し、整形済み結果を返す

        Args:
            image_data: 画像のバイトデータ

        Returns:
            整形済み OCR 結果
        """
        if not self.enable_ocr:
            raise ValueError("OCR is disabled in this deployment")
        if not settings.SAGEMAKER_ENDPOINT_NAME:
            raise ValueError("SageMaker endpoint not configured")

        try:
            image_base64 = base64.b64encode(image_data).decode("utf-8")
            response = sagemaker_runtime_client.invoke_endpoint(
                EndpointName=settings.SAGEMAKER_ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps({"image": image_base64}),
                InferenceComponentName=settings.SAGEMAKER_INFERENCE_COMPONENT_NAME,
            )
            response_body = json.loads(response["Body"].read().decode("utf-8"))
            return parse_ocr_response(response_body)
        except Exception as e:
            logger.error(f"SageMaker OCR 呼び出しエラー: {str(e)}")
            return {"error": f"SageMaker endpoint error: {str(e)}", "text": "", "words": [], "word_count": 0}

    async def get_ocr_result(self, image_id: str) -> OcrResultResponse:
        """OCR結果を取得する"""
        image_data = get_image(image_id)
        if not image_data:
            raise ValueError("Image not found")

        ocr_result = image_data.get("ocr_result", {})
        # OCR無効時はocr_resultが存在しない
        if ocr_result is None:
            ocr_result = {}

        image_url = f"{settings.API_BASE_URL}/image/{image_id}"
        s3_key = image_data.get("s3_key")
        if isinstance(s3_key, list):
            s3_key = s3_key[0] if s3_key else ""

        return OcrResultResponse(
            filename=image_data.get("filename"),
            s3_key=s3_key,
            uploadTime=image_data.get("upload_time"),
            status=image_data.get("status"),
            ocrResult=OcrResult(**ocr_result) if ocr_result else OcrResult(words=[]),
            imageUrl=image_url,
            app_name=image_data.get("app_name")
        )

    async def update_ocr_result(self, image_id: str, edited_ocr_data: dict) -> None:
        """OCR結果を更新する"""
        db_update_ocr_result(image_id, edited_ocr_data)

    def process_image_ocr(self, image_id: str) -> None:
        """画像のOCR処理を実行（処理モード自動判定）"""
        try:
            logger.info(f"Processing single image: {image_id}")
            image_data = get_image(image_id)
            if not image_data:
                raise ValueError(f"Image not found: {image_id}")

            update_image_status(image_id, "processing")
            sync_parent_status(image_id)

            page_processing_mode = image_data.get("page_processing_mode", "combined")
            converted_s3_keys = image_data.get("converted_s3_key")
            is_multiimage_combined = (
                page_processing_mode == "combined"
                and isinstance(converted_s3_keys, list)
                and len(converted_s3_keys) > 1
            )
            is_individual_page = image_data.get("parent_document_id") is not None

            logger.info(f"Processing image {image_id} (mode: {page_processing_mode})")

            if is_multiimage_combined:
                self._process_ocr_multipage(image_id, image_data)
            elif is_individual_page:
                self._process_ocr_individual_page(image_id, image_data)
            else:
                self._process_ocr_single_image(image_id, image_data)

            logger.info(f"Successfully completed OCR for image {image_id}")

        except Exception as e:
            logger.error(f"Error processing OCR for image {image_id}: {str(e)}")
            update_image_status(image_id, "failed")
            sync_parent_status(image_id)
            raise

    # ========================================
    # OCR オーケストレーション（元 domains/ocr_engine.py から移動）
    # ========================================

    def _process_ocr_multipage(self, image_id: str, image_data: dict) -> list:
        """複数ページのOCR処理"""
        logger.info(f"複数ページOCR処理を開始: {image_id}")

        converted_s3_keys = image_data.get("converted_s3_key")
        if not converted_s3_keys or not isinstance(converted_s3_keys, list):
            raise ValueError("複数ページの変換済み画像が見つかりません")

        ocr_results = []
        for i, s3_key in enumerate(converted_s3_keys):
            try:
                logger.info(f"ページ {i+1}/{len(converted_s3_keys)} OCR処理中: {s3_key}")
                s3_response = s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=s3_key)
                image_bytes = s3_response['Body'].read()
                ocr_result = self._invoke_ocr(image_bytes)
                if "error" in ocr_result:
                    raise ValueError(f"OCR処理エラー: {ocr_result['error']}")
                page_result = {
                    "page": i + 1,
                    "words": ocr_result.get("words", []),
                    "text": ocr_result.get("text", "")
                }
                ocr_results.append(page_result)
                logger.info(f"ページ {i+1} OCR完了")
            except Exception as e:
                logger.error(f"ページ {i+1} OCR処理エラー: {str(e)}")
                ocr_results.append({"page": i + 1, "words": [], "text": "", "error": str(e)})
                continue

        self._save_multipage_ocr_result(image_id, ocr_results)
        logger.info(f"複数ページOCR処理完了: {image_id}")
        return ocr_results

    def _process_ocr_individual_page(self, image_id: str, image_data: dict) -> None:
        """個別ページのOCR処理"""
        logger.info(f"個別ページ処理を実行: {image_id}")
        s3_key = image_data.get("s3_key")
        self._process_ocr_page(image_id, s3_key)

    def _process_ocr_single_image(self, image_id: str, image_data: dict) -> None:
        """単一画像のOCR処理"""
        logger.info(f"単一画像処理を実行: {image_id}")
        s3_key = image_data.get("converted_s3_key") or image_data.get("s3_key")
        self._process_ocr_page(image_id, s3_key)

    def _process_ocr_page(self, image_id: str, s3_key) -> None:
        """単一S3キーに対するOCR処理（共通ロジック）"""
        if isinstance(s3_key, list):
            s3_key = s3_key[0]

        s3_response = s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=s3_key)
        image_bytes = s3_response['Body'].read()

        ocr_result = self._invoke_ocr(image_bytes)

        if "error" in ocr_result:
            logger.error(f"OCR処理でエラーが発生: {ocr_result['error']}")
            update_image_status(image_id, "failed")
            return

        logger.info(f"Successfully processed {len(ocr_result.get('words', []))} words for image {image_id}")
        db_update_ocr_result(image_id, ocr_result, "processing")

    @staticmethod
    def _save_multipage_ocr_result(image_id: str, ocr_results: list) -> None:
        """複数ページOCR結果を保存"""
        try:
            # 統合OCR結果を作成（全ページ通してユニークなIDを付与）
            all_words = []
            global_word_id = 0

            for page_result in ocr_results:
                page_words = page_result.get("words", [])
                for word in page_words:
                    word["page"] = page_result["page"]
                    word["id"] = global_word_id
                    global_word_id += 1
                all_words.extend(page_words)

            # ページ別結果のIDも更新（参照用）
            updated_pages = []
            for page_result in ocr_results:
                updated_page = page_result.copy()
                page_words = []
                for word in page_result.get("words", []):
                    for updated_word in all_words:
                        if (updated_word.get("page") == page_result["page"]
                                and updated_word.get("content") == word.get("content")
                                and updated_word.get("points") == word.get("points")):
                            page_words.append(updated_word)
                            break
                updated_page["words"] = page_words
                updated_pages.append(updated_page)

            combined_result = float_to_decimal({
                "words": all_words,
                "pages": updated_pages,
                "total_pages": len(ocr_results)
            })

            db_update_ocr_result(image_id, combined_result, "completed")
            sync_parent_status(image_id)
            logger.info(f"複数ページOCR結果保存完了: {image_id}, 総単語数: {len(all_words)}, ID範囲: 0-{global_word_id-1}")

        except Exception as e:
            logger.error(f"複数ページOCR結果保存エラー: {str(e)}")
            raise

    # ========================================
    # Step Functions 起動
    # ========================================

    async def start_step_functions_job(self, request) -> Dict[str, Any]:
        """Step FunctionsでOCRジョブを開始する"""
        try:
            # OCR有効時のみエンドポイント状態確認
            if self.enable_ocr:
                status = get_inference_component_status(settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)

                if not status['ready']:
                    trigger_endpoint_wakeup(settings.SAGEMAKER_ENDPOINT_NAME, settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)
                    raise EndpointNotReadyError('Endpoint warming up')

            job_id = str(uuid.uuid4())
            app_name = request.app_name

            # pending画像を取得
            images = get_images(app_name)
            pending_images = [img for img in images if img.get('status') == 'pending']

            logger.info(f"Found {len(pending_images)} pending images for app: {app_name}")

            if not pending_images:
                logger.warning(f"No pending images found for app: {app_name}")
                return {"jobId": job_id}

            # ステータスを更新
            for img in pending_images:
                update_image_status(img['id'], 'processing', job_id)

            # Step Functions起動
            execution_response = sfn_client.start_execution(
                stateMachineArn=settings.STATE_MACHINE_ARN,
                name=f"ocr-job-{job_id}",
                input=json.dumps({
                    'job_id': job_id,
                    'images': [{'image_id': img['id']} for img in pending_images]
                })
            )

            logger.info(f"Started Step Functions execution: {execution_response['executionArn']}")

            return {"jobId": job_id}

        except Exception as e:
            logger.error(f"OCR job start error: {str(e)}")
            raise

    async def start_step_functions_for_image(self, image_id: str, skip_ocr: bool = False) -> Dict[str, Any]:
        """指定画像のStep Functions OCR処理を開始する"""
        try:
            # OCRをスキップしない場合かつOCR有効時のみエンドポイント状態確認
            if not skip_ocr and self.enable_ocr:
                status = get_inference_component_status(settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)

                if not status['ready']:
                    trigger_endpoint_wakeup(settings.SAGEMAKER_ENDPOINT_NAME, settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)
                    raise EndpointNotReadyError('Endpoint warming up')

            job_id = str(uuid.uuid4())

            # ステータスをprocessingに更新
            update_image_status(image_id, 'processing', job_id)

            # 再抽出のため extraction_status と agent_status をリセット
            # （前回の completed 値が残っていると、フロントのポーリングが
            #  新しい処理の完了を待たずに即 completed と表示してしまう）
            from repositories.image_repository import get_images_table
            get_images_table().update_item(
                Key={"id": image_id},
                UpdateExpression=(
                    "SET extraction_status = :proc, agent_status = :proc, "
                    "agent_suggestions_count = :zero"
                ),
                ExpressionAttributeValues={":proc": "processing", ":zero": 0},
            )

            # Step Functions起動（単一画像）
            execution_response = sfn_client.start_execution(
                stateMachineArn=settings.STATE_MACHINE_ARN,
                name=f"ocr-single-{image_id}-{job_id[:8]}",
                input=json.dumps({
                    'job_id': job_id,
                    'images': [{'image_id': image_id, 'skip_ocr': skip_ocr}]
                })
            )

            logger.info(f"Started Step Functions execution for image {image_id}: {execution_response['executionArn']}")

            return {"status": "processing", "image_id": image_id, "job_id": job_id}

        except Exception as e:
            logger.error(f"Error starting OCR for image: {str(e)}")
            raise

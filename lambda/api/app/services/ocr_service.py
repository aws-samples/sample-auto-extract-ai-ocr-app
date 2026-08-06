import uuid
import logging
import json
import base64
from typing import Dict, Any, List, Optional

from exceptions import EndpointNotReadyError, NotFoundError, BadRequestError
from repositories import (
    get_images,
    get_image, update_ocr_result as db_update_ocr_result,
    update_image_status,
    update_agent_status,
)
from clients import get_inference_component_status, get_endpoint_status_direct, trigger_endpoint_wakeup
from schemas import OcrResult, OcrResultResponse
from config import settings
from clients import s3_client, sagemaker_runtime_client, sfn_client
from domains.ocr_engine import parse_ocr_response, parse_yomitoku_mp_response
from domains.image_status import ImageStatus, AgentStatus, PageProcessingMode
from services.pdf_conversion_service import sync_parent_status
from utils.helpers import compress_image_for_payload

logger = logging.getLogger(__name__)

# Step Functions StartExecution の入力ペイロードは 256KB 上限。
# 1 画像あたり {"image_id": "<uuid>"} ≒ 55-60 バイトのため、余裕をもって
# 1 実行あたりの画像数を制限する（2000 件で約 120KB 相当）。
MAX_IMAGES_PER_EXECUTION = 2000


def _guess_image_content_type(image_data: bytes) -> str:
    """画像バイナリのマジックバイトから Content-Type を判定する"""
    if image_data[:3] == b'\xff\xd8\xff':
        return "image/jpeg"
    if image_data[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    if image_data[:4] in (b'II*\x00', b'MM\x00*'):
        return "image/tiff"
    return "image/jpeg"


class OcrService:
    """OCR処理を管理するサービスクラス"""

    def __init__(self):
        self.enable_ocr = settings.ENABLE_OCR

    def get_endpoint_status(self) -> dict:
        """OCR エンドポイントの状態を返す"""
        if not self.enable_ocr:
            return {"ready": True, "status": "ocr_disabled"}
        if settings.OCR_ENGINE == "yomitoku-mp":
            return get_endpoint_status_direct(settings.SAGEMAKER_ENDPOINT_NAME)
        return get_inference_component_status(settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)

    def _invoke_ocr(self, image_data: bytes) -> dict:
        """SageMaker OCR エンドポイントを呼び出し、整形済み結果を返す"""
        if not self.enable_ocr:
            raise BadRequestError("この環境では OCR が無効です")
        if not settings.SAGEMAKER_ENDPOINT_NAME:
            raise RuntimeError("SageMaker endpoint not configured")

        if settings.OCR_ENGINE == "yomitoku-mp":
            return self._invoke_yomitoku_mp(image_data)

        try:
            image_data = compress_image_for_payload(image_data)
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

    def _invoke_yomitoku_mp(self, image_data: bytes) -> dict:
        """Yomitoku Marketplace エンドポイントを呼び出す"""
        try:
            image_data = compress_image_for_payload(image_data, max_bytes=5 * 1024 * 1024)
            content_type = _guess_image_content_type(image_data)
            response = sagemaker_runtime_client.invoke_endpoint(
                EndpointName=settings.SAGEMAKER_ENDPOINT_NAME,
                ContentType=content_type,
                Body=image_data,
            )
            response_body = json.loads(response["Body"].read().decode("utf-8"))
            return parse_yomitoku_mp_response(response_body)
        except Exception as e:
            logger.error(f"Yomitoku MP 呼び出しエラー: {str(e)}")
            return {"error": f"SageMaker endpoint error: {str(e)}", "text": "", "words": [], "word_count": 0}

    async def get_ocr_result(self, image_id: str) -> OcrResultResponse:
        """OCR結果を取得する"""
        image_data = get_image(image_id)
        if not image_data:
            raise NotFoundError("画像が見つかりません")

        image_url = f"{settings.API_BASE_URL}/image/{image_id}"
        s3_key = image_data.get("s3_key")
        if isinstance(s3_key, list):
            s3_key = s3_key[0] if s3_key else ""

        return OcrResultResponse(
            filename=image_data.get("filename"),
            s3_key=s3_key,
            uploadTime=image_data.get("upload_time"),
            status=image_data.get("status"),
            ocrResult=self._to_ocr_result(image_data.get("ocr_result")),
            imageUrl=image_url,
            app_name=image_data.get("app_name")
        )

    @staticmethod
    def _to_ocr_result(ocr_result) -> OcrResult:
        """保存済みの OCR 結果を OcrResult に変換する。

        OCR 無効時は ocr_result 自体が無く、処理が失敗した画像には words を持たない
        `{"error": ..., "timestamp": ...}` が保存されている。words が取れないときは
        単語 0 件 + error として返す（必須項目不足で 500 にしない）。
        """
        if not isinstance(ocr_result, dict):
            return OcrResult(words=[])

        if not isinstance(ocr_result.get("words"), list):
            error = ocr_result.get("error")
            return OcrResult(words=[], error=str(error) if error is not None else None)

        return OcrResult(**ocr_result)

    async def update_ocr_result(self, image_id: str, edited_ocr_data: dict) -> None:
        """OCR結果を更新する"""
        db_update_ocr_result(image_id, edited_ocr_data)

    def process_image_ocr(self, image_id: str) -> None:
        """画像のOCR処理を実行（処理モード自動判定）"""
        try:
            logger.info(f"Processing single image: {image_id}")
            image_data = get_image(image_id)
            if not image_data:
                raise NotFoundError(f"画像が見つかりません: {image_id}")

            update_image_status(image_id, ImageStatus.OCR)
            sync_parent_status(image_id)

            page_processing_mode = image_data.get("page_processing_mode", PageProcessingMode.COMBINED)
            converted_s3_keys = image_data.get("converted_s3_key")
            is_multiimage_combined = (
                page_processing_mode == PageProcessingMode.COMBINED
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
            update_image_status(image_id, ImageStatus.FAILED)
            sync_parent_status(image_id)
            raise

    # ========================================
    # OCR オーケストレーション
    # ========================================

    def _process_ocr_multipage(self, image_id: str, image_data: dict) -> list:
        """複数ページのOCR処理"""
        logger.info(f"複数ページOCR処理を開始: {image_id}")

        converted_s3_keys = image_data.get("converted_s3_key")
        if not converted_s3_keys or not isinstance(converted_s3_keys, list):
            raise NotFoundError("複数ページの変換済み画像が見つかりません")

        ocr_results = []
        for i, s3_key in enumerate(converted_s3_keys):
            try:
                logger.info(f"ページ {i+1}/{len(converted_s3_keys)} OCR処理中: {s3_key}")
                s3_response = s3_client.get_object(Bucket=settings.BUCKET_NAME, Key=s3_key)
                image_bytes = s3_response['Body'].read()
                ocr_result = self._invoke_ocr(image_bytes)
                if "error" in ocr_result:
                    raise RuntimeError(f"OCR処理エラー: {ocr_result['error']}")
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
            raise RuntimeError(f"OCR処理でエラーが発生: {ocr_result['error']}")

        logger.info(f"Successfully processed {len(ocr_result.get('words', []))} words for image {image_id}")
        db_update_ocr_result(image_id, ocr_result)

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

            combined_result = {
                "words": all_words,
                "pages": updated_pages,
                "total_pages": len(ocr_results)
            }

            db_update_ocr_result(image_id, combined_result)
            sync_parent_status(image_id)
            logger.info(f"複数ページOCR結果保存完了: {image_id}, 総単語数: {len(all_words)}, ID範囲: 0-{global_word_id-1}")

        except Exception as e:
            logger.error(f"複数ページOCR結果保存エラー: {str(e)}")
            raise

    # ========================================
    # Step Functions 起動
    # ========================================

    def _check_endpoint_ready(self, skip_ocr: bool) -> None:
        """OCR エンドポイントが処理可能か確認する。未起動なら wakeup を促し例外を投げる。"""
        if skip_ocr or not self.enable_ocr:
            return
        status = self.get_endpoint_status()
        if not status['ready']:
            if settings.OCR_ENGINE != "yomitoku-mp":
                trigger_endpoint_wakeup(settings.SAGEMAKER_ENDPOINT_NAME, settings.SAGEMAKER_INFERENCE_COMPONENT_NAME)
            raise EndpointNotReadyError('OCR エンドポイントが起動中です。しばらくお待ちください。')

    async def _start_pipeline(
        self, image_ids: List[str], skip_ocr: bool, job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """指定画像群の OCR→抽出パイプラインを 1 つの Step Functions 実行で開始する。

        バッチ起動・単一起動の共通実体。エンドポイント確認 → ステータス更新 → SFn 起動を
        この順で行う。SFn 起動に失敗したら各画像を元のステータスへ戻す。
        """
        job_id = job_id or str(uuid.uuid4())

        # 重複除去（順序は保持）
        unique_ids = list(dict.fromkeys(image_ids))
        if not unique_ids:
            return {"jobId": job_id}

        # エンドポイント確認はステータスを触る前に行う（未起動なら何も変更せず例外）。
        self._check_endpoint_ready(skip_ocr)

        # Step Functions の StartExecution 入力は 256KB 制限があるため上限で切る。
        if len(unique_ids) > MAX_IMAGES_PER_EXECUTION:
            logger.warning(
                f"Image count ({len(unique_ids)}) exceeds per-execution limit "
                f"({MAX_IMAGES_PER_EXECUTION}). Processing first {MAX_IMAGES_PER_EXECUTION}."
            )
            unique_ids = unique_ids[:MAX_IMAGES_PER_EXECUTION]

        # ロールバック用に各画像の現ステータスを控える。
        prior_statuses = {img_id: self._get_image_status(img_id) for img_id in unique_ids}

        try:
            for img_id in unique_ids:
                # 再処理の起点。抽出フェーズへの遷移は extract() 冒頭が担う。
                update_image_status(img_id, ImageStatus.OCR, job_id)
                # 過去の agent 結果を破棄する意味で idle に戻す。AgentKick が実行時に更新する。
                update_agent_status(img_id, AgentStatus.IDLE, suggestions_count=0)

            execution_response = sfn_client.start_execution(
                stateMachineArn=settings.STATE_MACHINE_ARN,
                name=f"ocr-job-{job_id}",
                input=json.dumps({
                    'job_id': job_id,
                    'images': [{'image_id': img_id, 'skip_ocr': skip_ocr} for img_id in unique_ids]
                })
            )
        except Exception as start_error:
            logger.error(f"Step Functions start failed, reverting image statuses: {start_error}")
            for img_id, prior in prior_statuses.items():
                if prior:
                    update_image_status(img_id, prior)
            raise

        logger.info(f"Started Step Functions execution: {execution_response['executionArn']}")
        return {"jobId": job_id}

    @staticmethod
    def _get_image_status(image_id: str) -> Optional[str]:
        """ロールバック用に画像の現ステータスを取得する（取得失敗時は None）。"""
        image = get_image(image_id)
        return image.get('status') if image else None

    async def start_step_functions_job(self, request) -> Dict[str, Any]:
        """app 単位で OCR パイプラインを開始する。

        request.image_ids 省略時はその app の PENDING 画像全件、指定時はその画像群
        （app に属することを検証）を対象にする。
        """
        app_name = request.app_name
        image_ids = getattr(request, "image_ids", None)
        skip_ocr = getattr(request, "skip_ocr", False)

        app_images = get_images(app_name)

        if image_ids:
            # 指定 ID が app に属することを検証（属さない ID があれば弾く）。
            known_ids = {img['id'] for img in app_images}
            invalid = [i for i in image_ids if i not in known_ids]
            if invalid:
                raise BadRequestError(f"指定された画像がアプリ '{app_name}' に存在しません: {invalid}")
            target_ids = image_ids
        else:
            # 省略時は PENDING 全件。individual モードの親コンテナ（子を束ねる表示用・
            # parent_document_id を持たない親）は OCR 対象でないため除外する。
            target_ids = [
                img['id'] for img in app_images
                if img.get('status') == ImageStatus.PENDING and not self._is_parent_container(img, app_images)
            ]
            logger.info(f"Found {len(target_ids)} pending images for app: {app_name}")

        return await self._start_pipeline(target_ids, skip_ocr)

    @staticmethod
    def _is_parent_container(image: Dict[str, Any], all_images: List[Dict[str, Any]]) -> bool:
        """individual モードの親コンテナ（自身は子を持ち、親を持たない）か判定する。"""
        if image.get('parent_document_id'):
            return False
        return any(other.get('parent_document_id') == image['id'] for other in all_images)

    async def start_step_functions_for_image(self, image_id: str, skip_ocr: bool = False) -> Dict[str, Any]:
        """指定画像 1 件の OCR パイプラインを開始する（再処理・再抽出用）。"""
        return await self._start_pipeline([image_id], skip_ocr)

import logging
from typing import Dict, Any

from repositories import (
    get_image, get_images,
    get_children_by_parent_id, delete_image as repo_delete_image
)
from utils import decimal_to_float
from repositories.usecase_repository import get_permitted_app_names
from repositories import user_repository
from schemas.image import ImageInfo
from domains.image_status import PageProcessingMode

logger = logging.getLogger(__name__)


class ImageListService:
    """画像一覧取得・削除を管理するサービスクラス"""

    @staticmethod
    def _serialize_images(images: list[dict]) -> list[dict]:
        """DynamoDB の画像レコードを API レスポンス形式（camelCase）に変換する"""
        result = []
        for img in images:
            try:
                converted = decimal_to_float(img)
                info = ImageInfo.model_validate(converted)
                result.append(info.model_dump(by_alias=True))
            except Exception as e:
                logger.error(f"Image serialization error for {img.get('id', '?')}: {e}; raw_keys={sorted(img.keys())}")
                result.append({"id": img.get("id", ""), "name": img.get("filename", ""), "status": img.get("status", "")})
        return result

    @staticmethod
    def _enrich_uploaded_by_email(images: list[dict]) -> None:
        """画像リストに uploaded_by_email / verified_by_email を付与する"""
        subs = set()
        for img in images:
            if img.get("uploaded_by"):
                subs.add(img["uploaded_by"])
            if img.get("verified_by"):
                subs.add(img["verified_by"])
        if not subs:
            return
        email_map = user_repository.get_emails_by_cognito_subs(subs)
        for img in images:
            img["uploaded_by_email"] = email_map.get(img.get("uploaded_by", ""), "")
            img["verified_by_email"] = email_map.get(img.get("verified_by", ""), "")

    async def get_images_list(self, app_name: str = None, uploaded_by: str = None) -> Dict[str, Any]:
        """画像一覧を取得する"""
        try:
            images = get_images(app_name, uploaded_by=uploaded_by)
            self._enrich_uploaded_by_email(images)

            serialized = self._serialize_images(images)
            result = {
                "images": serialized,
                "total": len(serialized)
            }

            logger.info(f"Retrieved {len(serialized)} images")
            return result

        except Exception as e:
            logger.error(f"Error getting images list: {str(e)}")
            raise

    async def get_images_for_user(self, user_id: str, role: str, app_name: str = None) -> Dict[str, Any]:
        """ユーザーの権限に応じた画像一覧を取得する"""
        if role == "admin":
            return await self.get_images_list(app_name)

        permitted = get_permitted_app_names(user_id)
        if not permitted:
            return {"images": [], "total": 0}

        return await self.get_images_for_permitted_apps(permitted, app_name_filter=app_name)

    async def get_images_for_permitted_apps(self, app_names: list[str], app_name_filter: str = None) -> Dict[str, Any]:
        """権限のあるユースケースの画像一覧を取得する"""
        try:
            if app_name_filter:
                if app_name_filter not in app_names:
                    return {"images": [], "total": 0}
                target_apps = [app_name_filter]
            else:
                target_apps = app_names

            all_images = []
            for name in target_apps:
                images = get_images(app_name=name)
                all_images.extend(images)

            all_images.sort(key=lambda x: x.get("upload_time", ""), reverse=True)
            self._enrich_uploaded_by_email(all_images)

            serialized = self._serialize_images(all_images)
            return {
                "images": serialized,
                "total": len(serialized)
            }
        except Exception as e:
            logger.error(f"Error getting images for permitted apps: {str(e)}")
            raise

    async def delete_image(self, image_id: str, cognito_sub: str = None, is_admin: bool = False) -> Dict[str, Any]:
        """画像を削除する"""
        try:
            image = get_image(image_id)
            if not image:
                raise ValueError("Image not found")

            if not is_admin:
                if not cognito_sub or image.get("uploaded_by") != cognito_sub:
                    raise PermissionError("Forbidden: not the owner")

            parent_document_id = image.get("parent_document_id")
            page_processing_mode = image.get("page_processing_mode")
            total_pages = image.get("total_pages", 0)

            is_parent = (not parent_document_id and
                        page_processing_mode == PageProcessingMode.INDIVIDUAL and
                        total_pages > 1)

            if is_parent:
                children = get_children_by_parent_id(image_id)
                for child in children:
                    repo_delete_image(child['id'])
                    logger.info(f"Deleted child image: {child['id']}")

            remaining_count = 0
            if parent_document_id:
                all_children = get_children_by_parent_id(parent_document_id)
                remaining_count = len([c for c in all_children if c['id'] != image_id])
                logger.info(f"Remaining children count (before deletion): {remaining_count}")

            repo_delete_image(image_id)
            logger.info(f"Deleted image: {image_id}")

            if parent_document_id and remaining_count == 0:
                repo_delete_image(parent_document_id)
                logger.info(f"Deleted parent image: {parent_document_id}")

            return {"status": "success", "message": "Image deleted successfully"}

        except Exception as e:
            logger.error(f"Error deleting image: {str(e)}")
            raise

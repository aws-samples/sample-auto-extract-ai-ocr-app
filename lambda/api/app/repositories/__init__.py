"""
Data access layer for DynamoDB and DSQL operations
"""
from .image_repository import (
    create_image_record,
    get_images,
    get_image,
    update_image_status,
    update_ocr_result,
    update_extracted_info,
    update_converted_image,
    delete_images_by_app_name,
    delete_image,
    update_verification_status,
    create_individual_page_record,
    update_parent_document_status,
    get_children_by_parent_id,
)

from .job_repository import (
    get_job,
)
from .schema_repository import (
    load_app_schemas,
    get_app_schemas,
    get_app_schema,
    get_extraction_fields_for_app,
    get_app_display_name,
    get_app_input_methods,
    get_custom_prompt_for_app,
    update_app_schema,
    delete_app_schema,
)

from . import user_repository
from . import group_repository
from . import usecase_repository
from . import tool_repository

__all__ = [
    # Image operations
    "create_image_record",
    "get_images",
    "get_image",
    "update_image_status",
    "update_ocr_result",
    "update_extracted_info",
    "update_converted_image",
    "delete_images_by_app_name",
    "delete_image",
    "update_verification_status",
    "create_individual_page_record",
    "update_parent_document_status",
    "get_children_by_parent_id",
    # Job operations
    "get_job",
    # Schema operations
    "load_app_schemas",
    "get_app_schemas",
    "get_app_schema",
    "get_extraction_fields_for_app",
    "get_app_display_name",
    "get_app_input_methods",
    "get_custom_prompt_for_app",
    "update_app_schema",
    "delete_app_schema",
    # DSQL repositories
    "user_repository",
    "group_repository",
    "usecase_repository",
    "tool_repository",
]

"""
Domain logic layer - Pure business logic（外部サービス呼び出しなし）
"""
from .extraction_engine import (
    build_single_image_with_ocr_request,
    build_multi_images_with_ocr_request,
    build_multi_images_without_ocr_request,
    build_single_image_without_ocr_request,
    parse_extraction_response,
    finalize_extraction_result,
)
from .ocr_engine import (
    parse_ocr_response,
)
from .schema_generator import (
    build_schema_generation_request,
    parse_schema_generation_response,
)
from .schema_fields import (
    extract_field_names,
)
from .image_status import (
    determine_parent_status,
    determine_parent_agent_status,
    ImageStatus,
    AgentStatus,
    PageProcessingMode,
    validate_image_status,
    validate_agent_status,
    validate_page_processing_mode,
)
from .prompts import (
    create_single_with_ocr_prompt,
    create_single_without_ocr_prompt,
    create_multi_with_ocr_prompt,
    create_multi_without_ocr_prompt,
)
from .template import (
    generate_unified_template,
    generate_json_template,
    generate_indices_template,
)

__all__ = [
    # Extraction — メッセージ構築
    "build_single_image_with_ocr_request",
    "build_multi_images_with_ocr_request",
    "build_multi_images_without_ocr_request",
    "build_single_image_without_ocr_request",
    # Extraction — レスポンスパース
    "parse_extraction_response",
    "finalize_extraction_result",
    # OCR
    "parse_ocr_response",
    # Schema
    "build_schema_generation_request",
    "parse_schema_generation_response",
    "extract_field_names",
    # Image status
    "determine_parent_status",
    "determine_parent_agent_status",
    "ImageStatus",
    "AgentStatus",
    "PageProcessingMode",
    "validate_image_status",
    "validate_agent_status",
    "validate_page_processing_mode",
    # Prompts
    "create_single_with_ocr_prompt",
    "create_single_without_ocr_prompt",
    "create_multi_with_ocr_prompt",
    "create_multi_without_ocr_prompt",
    # Template
    "generate_unified_template",
    "generate_json_template",
    "generate_indices_template",
]

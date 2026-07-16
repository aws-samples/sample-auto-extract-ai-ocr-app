import re

from pydantic import BaseModel, StringConstraints, field_validator, model_validator
from typing import Annotated, Optional, List, Dict, Any, Literal


FieldType = Literal["string", "number", "map", "list"]

# フィールド名 / アプリ名に許可する文字。英数字とアンダースコアのみ。
# keep in sync: services/schema_service.py（アプリ名検証）, web/src/utils/schemaValidation.ts（NAME_PATTERN）
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_]+$")

FieldName = Annotated[str, StringConstraints(pattern=NAME_PATTERN.pattern)]
DisplayName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _validate_sibling_field_names(fields: List["SchemaField"]) -> None:
    """同じ階層でフィールド名が重複していないか検証する（重複で ValueError）。"""
    seen = set()
    for f in fields:
        if f.name in seen:
            raise ValueError(f"同じ階層でフィールド名が重複しています: {f.name}")
        seen.add(f.name)


class FieldItems(BaseModel):
    """list 型フィールドの要素定義"""
    type: FieldType
    fields: Optional[List["SchemaField"]] = None  # 要素が map の場合の子フィールド
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_structure(self) -> "FieldItems":
        if self.type == "list":
            raise ValueError("list の要素として list 型は指定できません")
        if self.type == "map":
            if not self.fields:
                raise ValueError("list の map 要素には子フィールド(fields)が必要です")
            _validate_sibling_field_names(self.fields)
        else:  # string / number
            if self.fields is not None:
                raise ValueError("list の string / number 要素に fields は指定できません")
        return self


class SchemaField(BaseModel):
    """抽出スキーマのフィールド定義（再帰構造）"""
    name: FieldName
    display_name: DisplayName
    type: FieldType
    fields: Optional[List["SchemaField"]] = None  # map 型の子フィールド
    items: Optional[FieldItems] = None            # list 型の要素定義
    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _check_structure(self) -> "SchemaField":
        # エラーメッセージに display_name を埋め込み、どのフィールドが原因か特定しやすくする
        label = self.display_name
        if self.type == "map":
            if not self.fields:
                raise ValueError(f"map 型「{label}」には子フィールド(fields)が必要です")
            if self.items is not None:
                raise ValueError(f"map 型「{label}」に items は指定できません")
            _validate_sibling_field_names(self.fields)
        elif self.type == "list":
            if self.items is None:
                raise ValueError(f"list 型「{label}」には要素定義(items)が必要です")
            if self.fields is not None:
                raise ValueError(f"list 型「{label}」に fields は指定できません")
        else:  # string / number
            if self.fields is not None or self.items is not None:
                raise ValueError(f"{self.type} 型「{label}」に fields や items は指定できません")
        return self


class SchemaGenerateRequest(BaseModel):
    """スキーマ生成リクエスト"""
    s3_key: str
    filename: str
    instructions: Optional[str] = None


class SchemaSaveRequest(BaseModel):
    """スキーマ保存リクエスト"""
    name: str
    display_name: str
    description: Optional[str] = None
    fields: List[SchemaField]
    input_methods: Dict[str, Any]
    agent_enabled: bool = False
    # スキーマ生成に使ったサンプル画像の S3 キー (schema-uploads/ 配下)。
    # None の場合、update 時は既存値を保持する (画像未変更の編集で消えないように)。
    sample_image_s3_key: Optional[str] = None
    sample_image_filename: Optional[str] = None
    # スキーマ生成に使った指示プロンプト。編集画面で復元し「プロンプトだけ変えて再生成」を可能にする。
    schema_instructions: Optional[str] = None

    @field_validator("fields")
    @classmethod
    def _check_fields(cls, v: List[SchemaField]) -> List[SchemaField]:
        if not v:
            raise ValueError("スキーマには最低1つのフィールドが必要です")
        _validate_sibling_field_names(v)
        return v


class SchemaGenerateStartResponse(BaseModel):
    """スキーマ生成の非同期ジョブ起動レスポンス"""
    job_id: str
    status: str  # "processing"


class SchemaGenerateStatusResponse(BaseModel):
    """スキーマ生成ジョブの状態確認レスポンス"""
    status: str  # "processing" | "completed" | "failed"
    result: Optional[Dict[str, Any]] = None  # 完了時のみ {"fields": [...]}
    error: Optional[str] = None  # 失敗時のみ

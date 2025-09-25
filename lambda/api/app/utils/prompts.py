"""
プロンプト生成関数
"""
import json
from utils.helpers import decimal_to_float, safe_get_from_dynamo_data
from utils.template import generate_unified_template
import logging

logger = logging.getLogger(__name__)


def create_multi_without_ocr_prompt(extraction_fields, field_names, custom_prompt=""):
    """OCRなし複数画像モード用のプロンプトを作成"""

    # 抽出対象の情報を整理（fieldsはリスト形式）
    extraction_targets = ""
    for field in extraction_fields:
        field_name = field.get("name", "")
        field_type = field.get("type", "text")
        description = field.get("description", "")
        extraction_targets += f"- {field_name} ({field_type}): {description}\n"

    # JSONテンプレートを作成
    json_template = json.dumps(field_names, ensure_ascii=False, indent=2)

    prompt = f"""複数ページの画像から以下の情報を抽出してください。OCR処理は行わず、画像を直接解析してください。

<extraction_fields>
{extraction_targets}
</extraction_fields>

{f'''
<custom_instructions>
{custom_prompt}
</custom_instructions>
''' if custom_prompt else ''}

<output_format>
以下のJSON形式で出力してください:
{json_template}
</output_format>

注意事項:
- 複数の画像から直接情報を読み取ってください
- 複数ページにまたがる情報は適切に統合してください
- 不明な項目は空文字列("")にしてください
- 数値は文字列として出力してください
- 日付は YYYY-MM-DD 形式で出力してください
- JSONのみを出力し、余計な説明は不要です
"""

    return prompt


def create_single_without_ocr_prompt(extraction_fields, field_names, custom_prompt=""):
    """OCRなし単一画像モード用のプロンプトを作成"""

    # 抽出対象の情報を整理（fieldsはリスト形式）
    extraction_targets = ""
    for field in extraction_fields:
        field_name = field.get("name", "")
        field_type = field.get("type", "text")
        description = field.get("description", "")
        extraction_targets += f"- {field_name} ({field_type}): {description}\n"

    # JSONテンプレートを作成
    json_template = json.dumps(field_names, ensure_ascii=False, indent=2)

    prompt = f"""画像から以下の情報を抽出してください。OCR処理は行わず、画像を直接解析してください。

<extraction_fields>
{extraction_targets}
</extraction_fields>

{f'''
<custom_instructions>
{custom_prompt}
</custom_instructions>
''' if custom_prompt else ''}

<output_format>
以下のJSON形式で出力してください:
{json_template}
</output_format>

注意事項:
- 画像から直接情報を読み取ってください
- 不明な項目は空文字列("")にしてください
- 数値は文字列として出力してください
- 日付は YYYY-MM-DD 形式で出力してください
- JSONのみを出力し、余計な説明は不要です
"""

    return prompt


def create_single_with_ocr_prompt(extraction_targets, unified_template, example_ocr, example_output,
                                  ocr_result, custom_prompt):
    """OCRあり単一画像用のプロンプトを作成"""

    prompt = f"""
    次のOCR結果から指定された情報を抽出してください。
    
    抽出対象情報には以下の型があります：
    - string型: 単一の文字列値
    - map型: 複数のフィールドを持つオブジェクト
    - list型: 複数の項目を持つ配列
    
    抽出した各情報について、対応するOCR結果の単語IDも指定してください。
    階層構造のデータについては、各フィールドごとにIDを指定してください。
    リスト型データについては各項目の各フィールドごとにIDを指定してください。
    
    情報が見つからない場合は該当項目を空文字列にし、IDは空の配列にしてください。
    アウトプットのフォーマット以外のテキストは一切出力しないでください。

    <extraction_example>
    例えば、以下のようなOCR結果があった場合：
    {json.dumps(example_ocr, ensure_ascii=False, indent=0)}
    
    以下のような抽出結果を期待します：
    {json.dumps(example_output, ensure_ascii=False, indent=0)}
    </extraction_example>

    <extraction_fields>
    {extraction_targets}
    </extraction_fields>

    <ocr_result>
    {json.dumps(decimal_to_float(ocr_result), ensure_ascii=False, indent=0)}
    </ocr_result>
    
    {f'''
    <custom_instructions>
    {custom_prompt}
    </custom_instructions>
    ''' if custom_prompt else ''}

    <output_format>
    {unified_template}
    </output_format>
    """

    return prompt


def create_multi_with_ocr_prompt(ocr_results: list, schema: dict, instructions: str, custom_prompt: str = ""):
    """OCRあり複数画像用のプロンプト生成（マッピング対応、カスタムプロンプト対応）"""

    # OCR結果をページ別に整理し、全単語にIDを付与
    ocr_text_by_page = []
    all_words_with_ids = []  # 全単語にIDを付与
    word_id = 0

    for i, page_result in enumerate(ocr_results):
        # page_resultが辞書でない場合はスキップ
        if not isinstance(page_result, dict):
            logger.warning(
                f"ページ結果が辞書形式ではありません: {type(page_result)} - {page_result}")
            continue

        page_num = safe_get_from_dynamo_data(page_result, "page", 1)
        page_words = safe_get_from_dynamo_data(page_result, "words", [])

        # page_wordsがリストでない場合は空リストに
        if not isinstance(page_words, list):
            logger.warning(
                f"ページ単語がリスト形式ではありません: {type(page_words)} - {page_words}")
            page_words = []

        # 各単語にIDを付与
        page_text_parts = []
        for word in page_words:
            # wordが辞書でない場合はスキップ
            if not isinstance(word, dict):
                logger.warning(f"単語が辞書形式ではありません: {type(word)} - {word}")
                continue

            word_content = safe_get_from_dynamo_data(
                word, "content", "").strip()
            if word_content:
                page_text_parts.append(f"[ID:{word_id}] {word_content}")
                all_words_with_ids.append({
                    "id": word_id,
                    "content": word_content,
                    "page": page_num,
                    "points": safe_get_from_dynamo_data(word, "points", [])
                })
                word_id += 1

        page_text = " ".join(page_text_parts)
        ocr_text_by_page.append(f"=== ページ {page_num} OCRテキスト ===\n{page_text}")

    combined_ocr_text = "\n\n".join(ocr_text_by_page)

    # スキーマから統合テンプレートを生成
    unified_template = generate_unified_template(schema)

    prompt = f"""
複数ページのOCR結果から以下の情報を抽出してください。

<instructions>
{instructions}
</instructions>

{f'''
<custom_instructions>
{custom_prompt}
</custom_instructions>
''' if custom_prompt else ''}

<ocr_results>
{combined_ocr_text}
</ocr_results>

<output_format>
{unified_template}
</output_format>

重要：回答は必ずJSONオブジェクトのみを返してください。説明文、コメント、マークダウン記法は一切含めないでください。
"""

    logger.info(f"複数ページプロンプト生成完了: {len(ocr_results)}ページ, {word_id}個の単語にID付与")
    return prompt

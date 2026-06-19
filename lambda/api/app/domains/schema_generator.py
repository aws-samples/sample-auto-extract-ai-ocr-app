import io
import json
import logging
import re

from PIL import Image

logger = logging.getLogger(__name__)


def build_schema_generation_request(image_data, instructions=None):
    """画像からスキーマ生成用の Bedrock リクエストを構築する（純粋関数）

    Args:
        image_data (bytes): 画像データ
        instructions (str, optional): スキーマ生成の指示

    Returns:
        tuple: (messages, system_prompts)
    """
    try:
        try:
            img = Image.open(io.BytesIO(image_data))
            image_type = (img.format or 'JPEG').lower()
        except Exception:
            image_type = 'jpeg'
        content_type = f"image/{image_type}"

        # システムプロンプト
        system_prompts = [{
            "text": """
            あなたはOCR処理された文書から情報抽出のためのフィールド定義を生成するアシスタントです。
            ユーザーが提供する画像を分析し、その文書タイプに適したフィールド構造を生成してください。
            """
        }]

        # フィールド定義の説明
        fields_explanation = """
        フィールド型は主に以下の3種類があります：
        
        1. string型: 単一の文字列値を格納するフィールド（日付、番号、名前など）
        2. map型: 複数の関連フィールドをグループ化するための階層構造（会社情報、住所情報など）
        3. list型: 表形式のデータなど、同じ構造を持つ複数の項目を格納するためのフィールド（明細行、商品リストなど）
        
        基本的には、単一の値は string 型、関連する複数の値をグループ化する場合は map 型、
        表形式のデータ（明細行など）は list 型を使用してください。
        """

        # フィールド定義の例
        fields_definition = """
        フィールドは以下の構造に従って定義してください：
        
        {
          "fields": [
            {
              "name": "フィールド名（英数字、アンダースコア）",
              "display_name": "フィールド表示名（日本語可）",
              "type": "string | map | list"  // フィールドの型
            },
            // map型の場合は子フィールドを定義
            {
              "name": "company_info",
              "display_name": "会社情報",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "会社名",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                }
              ]
            },
            // list型の場合はitemsを定義（表形式のデータ向け）
            {
              "name": "items",
              "display_name": "明細",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品目",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  }
                ]
              }
            }
          ]
        }
        """

        # 実際のフィールド例
        fields_examples = """
        以下は実際のフィールド定義例です：
        
        1. 請求書処理アプリケーションのフィールド例：
        {
          "fields": [
            {
              "name": "invoice_date",
              "display_name": "請求日",
              "type": "string"
            },
            {
              "name": "invoice_number",
              "display_name": "請求番号",
              "type": "string"
            },
            {
              "name": "company_info",
              "display_name": "会社情報",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "会社名",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "items",
              "display_name": "明細",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品目",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  },
                  {
                    "name": "unit_price",
                    "display_name": "単価",
                    "type": "string"
                  },
                  {
                    "name": "amount",
                    "display_name": "金額",
                    "type": "string"
                  }
                ]
              }
            },
            {
              "name": "total_amount",
              "display_name": "合計金額",
              "type": "string"
            },
            {
              "name": "tax_amount",
              "display_name": "消費税",
              "type": "string"
            }
          ]
        }
        
        2. 輸送伝票処理アプリケーションのフィールド例：
        {
          "fields": [
            {
              "name": "shipping_date",
              "display_name": "出荷日",
              "type": "string"
            },
            {
              "name": "tracking_number",
              "display_name": "追跡番号",
              "type": "string"
            },
            {
              "name": "sender",
              "display_name": "送り主",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "名前",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "receiver",
              "display_name": "受取人",
              "type": "map",
              "fields": [
                {
                  "name": "name",
                  "display_name": "名前",
                  "type": "string"
                },
                {
                  "name": "address",
                  "display_name": "住所",
                  "type": "string"
                },
                {
                  "name": "phone",
                  "display_name": "電話番号",
                  "type": "string"
                }
              ]
            },
            {
              "name": "items",
              "display_name": "荷物情報",
              "type": "list",
              "items": {
                "type": "map",
                "fields": [
                  {
                    "name": "description",
                    "display_name": "品名",
                    "type": "string"
                  },
                  {
                    "name": "quantity",
                    "display_name": "数量",
                    "type": "string"
                  },
                  {
                    "name": "weight",
                    "display_name": "重量",
                    "type": "string"
                  }
                ]
              }
            }
          ]
        }
        """

        # ユーザーからの指示があれば追加
        instruction_text = ""
        if instructions:
            instruction_text = f"""
            ユーザーからの追加指示：
            {instructions}
            """

        # ユーザーメッセージ
        user_message = {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": content_type.split('/')[1],
                        "source": {
                            "bytes": image_data
                        }
                    }
                },
                {
                    "text": f"""
                    この画像に写っている文書を分析し、情報抽出に適したフィールド定義を生成してください。
                    
                    {instruction_text}
                    
                    {fields_explanation}
                    
                    {fields_definition}
                    
                    {fields_examples}
                    
                    以下の点に注意してフィールドを生成してください：
                    1. 文書の種類（請求書、納品書、輸送伝票など）を特定し、適切なフィールド構造を設計する
                    2. 文書から抽出可能な全ての重要情報をフィールドとして定義する
                    3. 階層構造が必要な場合は適切にmap型を使用する
                    4. 表形式のデータ（明細行など）がある場合はlist型を使用する
                    5. フィールド名は英数字とアンダースコアのみを使用し、日本語は表示名に使用する
                    
                    必ず {{"fields": [...]}} の形式でJSONを出力してください。説明や補足は不要です。
                    """
                }
            ]
        }

        messages = [user_message]

        return messages, system_prompts

    except Exception as e:
        logger.error(f"スキーマ生成リクエスト構築エラー: {str(e)}")
        raise


def parse_schema_generation_response(fields_text: str) -> dict:
    """Bedrock レスポンスからスキーマフィールド定義をパースする（純粋関数）

    Args:
        fields_text: Bedrock からのレスポンステキスト

    Returns:
        {"fields": [...]} 形式のスキーマ定義
    """
    try:
        # JSONテキストからフィールド定義を抽出
        json_match = re.search(r'```json\s*(.*?)\s*```', fields_text, re.DOTALL)
        if json_match:
            fields_json = json_match.group(1)
        else:
            fields_json = fields_text

        schema = json.loads(fields_json)

        # スキーマが {"fields": [...]} の形式になっているか確認
        if "fields" not in schema:
            # fieldsキーがない場合は、配列を受け取ったと仮定して包む
            if isinstance(schema, list):
                schema = {"fields": schema}
            else:
                raise ValueError("生成されたスキーマに 'fields' キーがありません")

        return schema
    except json.JSONDecodeError as e:
        logger.error(f"フィールド定義のJSONパースエラー: {str(e)}")
        raise ValueError(f"生成されたフィールド定義が有効なJSONではありません: {fields_json}")

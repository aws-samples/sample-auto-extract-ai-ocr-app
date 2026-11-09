// =============================================================================
// Bedrock OCR（ocrEngine: "bedrock"）で使ってよいモデル
//
// 座標付きの転記は動作確認したモデルでしか精度を保証できないため、許可リスト外は
// synth 時に止める。実行時の同じ検証は lambda/api/app/domains/ocr_engine.py にあり、
// 片方を変えたらもう片方も揃えること。
// =============================================================================

/**
 * モデル ID → そのモデルを呼び出せるソースリージョン（ocrModelRegion）。
 *
 * GPT-6 Astra は bedrock-runtime では in-region 推論ができず、推論プロファイル経由でしか
 * Converse を呼べない（素の `openai.gpt-6-astra` はオンデマンド非対応で失敗する）。
 * リージョンはモデルカードの Regional Availability（bedrock-runtime）に合わせている。
 * https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html
 */
export const BEDROCK_OCR_MODELS: Readonly<Record<string, readonly string[]>> = {
  // Geo（US）推論プロファイル。US / カナダのリージョンからのみ呼べる。
  "us.openai.gpt-6-astra": [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "ca-west-1",
  ],
  // Global 推論プロファイル。処理リージョンは世界中のいずれかになる。
  "global.openai.gpt-6-astra": [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "ca-west-1",
    "eu-central-1",
    "eu-north-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-southeast-1",
    "ap-southeast-2",
    "sa-east-1",
  ],
};

/** ocrModelId 未指定時に使うモデル */
export const DEFAULT_BEDROCK_OCR_MODEL_ID = "us.openai.gpt-6-astra";

/**
 * OCR 用モデルとリージョンの組が許可されているか検証する。
 *
 * @throws 許可リスト外のモデル、またはそのモデルを呼べないリージョンのとき
 */
export function validateBedrockOcrModel(modelId: string, region: string): void {
  const regions = BEDROCK_OCR_MODELS[modelId];
  if (!regions) {
    throw new Error(
      `ocrModelId "${modelId}" は Bedrock OCR に使えません。` +
        `使えるモデル: ${Object.keys(BEDROCK_OCR_MODELS).join(", ")}`
    );
  }
  if (!regions.includes(region)) {
    throw new Error(
      `ocrModelId "${modelId}" は ocrModelRegion "${region}" から呼び出せません。` +
        `呼び出せるリージョン: ${regions.join(", ")}`
    );
  }
}

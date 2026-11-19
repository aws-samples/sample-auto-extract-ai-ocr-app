// =============================================================================
// Amazon Textract（ocrEngine: "textract"）を使えるリージョン
//
// OCR ワーカー Lambda はスタックと同じリージョンの Textract を呼ぶため、Textract が
// 提供されていないリージョンへのデプロイは synth 時に止める。
// 一覧は Textract の提供リージョン（2026-11 時点で 18 リージョン）に合わせている。
// https://aws.amazon.com/about-aws/whats-new/2026/11/textract-tokyo-zurich/
// =============================================================================

export const TEXTRACT_REGIONS: readonly string[] = [
  "us-east-1",
  "us-east-2",
  "us-west-1",
  "us-west-2",
  "ca-central-1",
  "eu-central-1",
  "eu-central-2",
  "eu-west-1",
  "eu-west-2",
  "eu-west-3",
  "eu-south-2",
  "ap-northeast-1",
  "ap-northeast-2",
  "ap-south-1",
  "ap-southeast-1",
  "ap-southeast-2",
  "us-gov-east-1",
  "us-gov-west-1",
];

/**
 * Textract を呼び出せるリージョンか検証する。
 *
 * @throws Textract が提供されていないリージョンのとき
 */
export function validateTextractRegion(region: string): void {
  if (!TEXTRACT_REGIONS.includes(region)) {
    throw new Error(
      `ocrEngine "textract" はリージョン "${region}" で使えません。` +
        `使えるリージョン: ${TEXTRACT_REGIONS.join(", ")}`
    );
  }
}

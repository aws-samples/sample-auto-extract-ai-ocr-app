// =============================================================================
// アプリケーションパラメータ設定
//
// 編集箇所:
// 1. defaultParameters — 全環境共通のデフォルト値
// 2. envOverrides     — 環境ごとの差分（base / dev / stg / prod）
//
// ENV 未指定時は base が使われる。
// envOverrides に記述のないパラメータは defaultParameters の値がそのまま適用される。
// =============================================================================

import {
  DEFAULT_BEDROCK_OCR_MODEL_ID,
  validateBedrockOcrModel,
} from "./bedrock-ocr-models";

export interface AppParameters {
  // Bedrock（情報抽出・スキーマ生成）
  modelId: string;
  modelRegion: string;

  // OCR
  enableOcr: boolean;
  /**
   * "bedrock" は Bedrock 上の LLM（ocrModelId のモデル）で OCR を行う。
   * SageMaker エンドポイントは作成されず、sagemakerZeroScale 等は無視される。
   */
  ocrEngine: "paddle" | "deepseek" | "yomitoku-mp" | "bedrock";
  /**
   * ocrEngine が "bedrock" のときに OCR に使うモデル。lib/bedrock-ocr-models.ts の
   * 許可リスト外を指定すると synth 時にエラーになる。未指定時は DEFAULT_BEDROCK_OCR_MODEL_ID。
   * 抽出用の modelId にはフォールバックしない（抽出用に別モデルを選んだとき、それが OCR に
   * 流れて許可リストで弾かれるのを避けるため）。
   */
  ocrModelId?: string;
  /** OCR で Bedrock を呼び出すリージョン。未指定時は modelRegion を使う。 */
  ocrModelRegion?: string;
  sagemakerZeroScale: boolean;
  sagemakerScaleInCooldownSeconds: number;
  marketplaceModelPackageArn?: string;

  // Agent
  enableAgentDemo: boolean;
  /** エージェント検証用モデル。未指定時は modelId / modelRegion を使う。 */
  agentModelId?: string;
  agentModelRegion?: string;

  // Cognito
  selfSignUpEnabled: boolean;
  /** 空配列 = 制限なし */
  allowedSignUpEmailDomains: string[];

  // SAML 認証（手順は docs/SAML_WITH_ENTRA_ID.md）
  /** true にするとログイン画面を SAML 専用にし、メールアドレス・パスワードのログインとセルフサインアップを無効にする */
  samlAuthEnabled: boolean;
  /** Cognito ドメイン。スキームなしで指定する（例: your-preferred-name.auth.ap-northeast-1.amazoncognito.com） */
  samlCognitoDomainName?: string;
  /** UserPool に登録した SAML IdP のプロバイダ名（例: EntraID） */
  samlCognitoFederatedIdentityProviderName?: string;

  // WAF
  waf: WafOptions;
}

export interface WafOptions {
  /**
   * WAF を有効にするか。
   * true にすると AWS Managed Rules (Common Rule Set) が常に適用される。
   * IP/Geo の制限は追加で指定した場合のみ有効。
   */
  enabled: boolean;
  allowedIpV4AddressRanges?: string[];
  allowedIpV6AddressRanges?: string[];
  allowedCountryCodes?: string[];
}

// =============================================================================
// デフォルト値
// =============================================================================
const defaultParameters: AppParameters = {
  modelId: "us.anthropic.claude-sonnet-4-6",
  modelRegion: "us-east-1",
  enableOcr: true,
  ocrEngine: "paddle",
  sagemakerZeroScale: true,
  sagemakerScaleInCooldownSeconds: 3600,
  enableAgentDemo: true,
  selfSignUpEnabled: true,
  allowedSignUpEmailDomains: [],
  samlAuthEnabled: false,
  waf: {
    enabled: false,
  },
};

// =============================================================================
// 環境ごとの差分
// =============================================================================
const envOverrides: Record<string, Partial<AppParameters>> = {
  base: {
    
  },
  dev: {

  },
  stg: {

  },
  prod: {
    sagemakerZeroScale: false,
    selfSignUpEnabled: false,
    waf: {
      enabled: true,
      allowedCountryCodes:["JP"]
    }
  },
};

export function getParameters(env?: string): AppParameters {
  const target = envOverrides[env || "base"] ?? {};
  const params = resolveOcrModel({ ...defaultParameters, ...target });
  validateSamlAuth(params);
  return params;
}

/**
 * samlAuthEnabled のとき Cognito ドメインと IdP 名を検証する。欠けていると誰もログインできないため synth で止める。
 *
 * @throws ドメインか IdP 名が空、またはドメインにスキームやパスが付いているとき
 */
export function validateSamlAuth(params: AppParameters): void {
  if (!params.samlAuthEnabled) return;
  const domain = params.samlCognitoDomainName ?? "";
  if (!domain || !params.samlCognitoFederatedIdentityProviderName) {
    throw new Error(
      "samlAuthEnabled が true のときは samlCognitoDomainName と samlCognitoFederatedIdentityProviderName を指定してください",
    );
  }
  if (domain.includes("://") || domain.includes("/")) {
    throw new Error(
      `samlCognitoDomainName はスキームなしのホスト名で指定してください（例: your-preferred-name.auth.ap-northeast-1.amazoncognito.com）: ${domain}`,
    );
  }
}

/**
 * ocrEngine が "bedrock" のとき OCR 用モデルの既定値を埋め、許可リストで検証する。
 * それ以外のエンジンでは ocrModelId / ocrModelRegion を使わないため外す。
 *
 * @throws 許可リスト外のモデル、またはそのモデルを呼べないリージョンのとき
 */
export function resolveOcrModel(params: AppParameters): AppParameters {
  if (!params.enableOcr || params.ocrEngine !== "bedrock") {
    return { ...params, ocrModelId: undefined, ocrModelRegion: undefined };
  }
  const ocrModelId = params.ocrModelId || DEFAULT_BEDROCK_OCR_MODEL_ID;
  const ocrModelRegion = params.ocrModelRegion || params.modelRegion;
  validateBedrockOcrModel(ocrModelId, ocrModelRegion);
  return { ...params, ocrModelId, ocrModelRegion };
}

export function getStackName(env?: string): string {
  if (!env || env === "base") return "OcrAppStack";
  return `OcrAppStack-${env}`;
}

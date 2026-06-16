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

export interface AppParameters {
  // Bedrock
  modelId: string;
  modelRegion: string;

  // OCR
  enableOcr: boolean;
  ocrEngine: "paddle" | "deepseek";
  sagemakerZeroScale: boolean;
  sagemakerScaleInCooldownSeconds: number;

  // Agent
  enableAgent: boolean;
  enableAgentDemo: boolean;

  // Cognito
  selfSignUpEnabled: boolean;
  /** 空配列 = 制限なし */
  allowedSignUpEmailDomains: string[];

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
  enableAgent: true,
  enableAgentDemo: true,
  selfSignUpEnabled: true,
  allowedSignUpEmailDomains: [],
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
  return { ...defaultParameters, ...target };
}

export function getStackName(env?: string): string {
  if (!env || env === "base") return "OcrAppStack";
  return `OcrAppStack-${env}`;
}

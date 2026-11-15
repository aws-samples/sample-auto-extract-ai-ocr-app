/**
 * validateSamlAuth のテスト。
 *
 * 想定している正しい挙動:
 * - samlAuthEnabled が false なら、ドメインや IdP 名が空でも例外にしない（既定の構成）。
 * - samlAuthEnabled が true なら、Cognito ドメインと IdP 名の両方が必要。欠けていれば synth 時に例外にする
 *   （ログインボタンが動かず誰もログインできない画面をデプロイしない）。
 * - Cognito ドメインはスキームなしのホスト名で受け取る。https:// やパスが付いていたら例外にする
 *   （Amplify が https:// を付けるため不正な URL になる）。
 */
import { AppParameters, getParameters, validateSamlAuth } from '../lib/parameters';

const params = (overrides: Partial<AppParameters> = {}): AppParameters => ({
  ...getParameters('dev'),
  samlAuthEnabled: true,
  samlCognitoDomainName: 'your-preferred-name.auth.ap-northeast-1.amazoncognito.com',
  samlCognitoFederatedIdentityProviderName: 'EntraID',
  ...overrides,
});

describe('validateSamlAuth', () => {
  test('passes when SAML is disabled even without domain or provider', () => {
    expect(() =>
      validateSamlAuth(
        params({ samlAuthEnabled: false, samlCognitoDomainName: undefined, samlCognitoFederatedIdentityProviderName: undefined }),
      ),
    ).not.toThrow();
  });

  test('passes with a host-only domain and a provider name', () => {
    expect(() => validateSamlAuth(params())).not.toThrow();
  });

  test.each([
    ['domain', { samlCognitoDomainName: undefined }],
    ['domain (empty)', { samlCognitoDomainName: '' }],
    ['provider name', { samlCognitoFederatedIdentityProviderName: undefined }],
  ])('rejects a missing %s', (_label, overrides) => {
    expect(() => validateSamlAuth(params(overrides))).toThrow(/samlCognitoDomainName と samlCognitoFederatedIdentityProviderName/);
  });

  test.each([
    'https://your-preferred-name.auth.ap-northeast-1.amazoncognito.com',
    'your-preferred-name.auth.ap-northeast-1.amazoncognito.com/',
  ])('rejects a domain with a scheme or path: %s', (domain) => {
    expect(() => validateSamlAuth(params({ samlCognitoDomainName: domain }))).toThrow(/スキームなしのホスト名/);
  });
});

"""ID トークンのクレームからユーザー同期用のプロフィールを組み立てるドメインロジック"""
from dataclasses import dataclass
from urllib.parse import unquote_plus


@dataclass(frozen=True)
class IdpUserProfile:
    """IdP（Cognito）が持つユーザー属性。DSQL の users / IdP グループ同期の入力。"""
    cognito_sub: str
    email: str
    display_name: str
    department: str
    idp_groups: tuple[str, ...]


def parse_idp_groups(raw: str) -> tuple[str, ...]:
    """`custom:idp_group` の値をグループ名の列にする（空要素・重複は除き、順序は保つ）。

    Cognito は複数の値を URL エンコードした `[a, b]` 形式で渡すため、その形式のときだけデコードする。
    """
    value = (raw or "").strip()
    if value.startswith("[") and value.endswith("]"):
        items = (unquote_plus(v.strip()) for v in value[1:-1].split(","))
    else:
        items = (v for v in value.split(","))
    return tuple(dict.fromkeys(g.strip() for g in items if g.strip()))


def profile_from_claims(claims: dict) -> IdpUserProfile:
    """ID トークンのクレームから IdpUserProfile を作る。"""
    email = claims.get("email", "") or ""
    return IdpUserProfile(
        cognito_sub=claims.get("sub", "") or "",
        email=email,
        display_name=claims.get("name") or email,
        department=claims.get("custom:department", "") or "",
        idp_groups=parse_idp_groups(claims.get("custom:idp_group", "") or ""),
    )


def needs_profile_update(existing: dict, profile: IdpUserProfile) -> bool:
    """既存の users 行を IdP の値で更新する必要があるか。

    比較するのは IdP が正となる email / department だけ。display_name は利用者が
    PATCH /user/me で変えられるため、role は管理画面で変えるため、IdP の値と違っても更新しない。
    department は未設定の行が NULL、クレーム側は空文字なので同じものとして扱う。
    """
    return (
        (existing.get("email") or "") != profile.email
        or (existing.get("department") or "") != profile.department
    )

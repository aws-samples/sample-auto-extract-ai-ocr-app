"""ID トークンのクレームからユーザー同期用のプロフィールを組み立てるドメインロジック"""
from dataclasses import dataclass


@dataclass(frozen=True)
class IdpUserProfile:
    """IdP（Cognito）が持つユーザー属性。DSQL の users / IdP グループ同期の入力。"""
    cognito_sub: str
    email: str
    display_name: str
    department: str
    idp_groups: tuple[str, ...]


def profile_from_claims(claims: dict) -> IdpUserProfile:
    """ID トークンのクレームから IdpUserProfile を作る。

    `custom:idp_group` はカンマ区切りの文字列で届く。空要素と重複は落とす。
    """
    email = claims.get("email", "") or ""
    raw_groups = claims.get("custom:idp_group", "") or ""
    groups = (g.strip() for g in raw_groups.split(","))
    return IdpUserProfile(
        cognito_sub=claims.get("sub", "") or "",
        email=email,
        display_name=claims.get("name") or email,
        department=claims.get("custom:department", "") or "",
        idp_groups=tuple(dict.fromkeys(g for g in groups if g)),
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

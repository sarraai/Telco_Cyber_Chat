from __future__ import annotations

from typing import Dict

from langgraph_sdk.auth import Auth

auth = Auth()

# ---------------------------------------------------------
# 1) Demo tokens: one per role
#    ⚠️ Later you can move tokens to env / DB.
# ---------------------------------------------------------
VALID_TOKENS: Dict[str, Dict] = {
    # Admin (e.g., supervisor / teacher)
    "admin-token-123": {
        "identity": "admin_user",
        "role": "admin",
    },
    # IT specialist
    "it-token-123": {
        "identity": "it_specialist_user",
        "role": "it_specialist",
    },
    # Network admin
    "netadmin-token-123": {
        "identity": "network_admin_user",
        "role": "network_admin",
    },
    # End user
    "enduser-token-123": {
        "identity": "end_user_user",
        "role": "end_user",
    },
}


# ---------------------------------------------------------
# 2) Authentication: map Authorization header → MinimalUserDict
# ---------------------------------------------------------
@auth.authenticate
async def get_current_user(
    authorization: str | None,
) -> auth.types.MinimalUserDict:
    """
    Parse the `Authorization` header and map it to a user + role.

    Expected header:
      Authorization: Bearer <TOKEN>

    Where <TOKEN> is one of the keys in VALID_TOKENS above.
    """
    if not authorization:
        raise auth.exceptions.HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use 'Authorization: Bearer <token>'.",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid Authorization header. Expected 'Bearer <token>'.",
        )

    user = VALID_TOKENS.get(token.strip())
    if user is None:
        raise auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid or unknown API token.",
        )

    role = user["role"]

    # MinimalUserDict: identity + optional metadata/permissions
    return auth.types.MinimalUserDict(
        identity=user["identity"],
        permissions=[role],          # you can use this in authorize()
        metadata={"role": role},     # useful later if you want
    )


# ---------------------------------------------------------
# 3) Authorization (optional, for now allow everything)
# ---------------------------------------------------------

@auth.authorize("threads")
async def authorize_threads(
    ctx: auth.types.AuthContext,
    value: dict,
) -> dict:
    """
    Called whenever the client interacts with the /threads resource.

    For now, we allow all roles. Later you can use:
        role = ctx.user.metadata.get("role")
    to restrict some actions based on the role.
    """
    return value

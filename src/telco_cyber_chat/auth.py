from __future__ import annotations

from typing import Dict

from langgraph_sdk import Auth

# This is the object LangGraph / LangSmith will use
auth = Auth()

# ---------------------------------------------------------
# 1) Demo tokens: one per role
#    ⚠️ Later you can move tokens to env / DB.
# ---------------------------------------------------------
VALID_TOKENS: Dict[str, Dict[str, str]] = {
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
) -> Auth.types.MinimalUserDict:
    """
    Parse the `Authorization` header and map it to a user + role.

    Expected header:
      Authorization: Bearer <TOKEN>

    Where <TOKEN> is one of the keys in VALID_TOKENS above.
    """
    if not authorization:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Missing Authorization header. Use 'Authorization: Bearer <token>'.",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid Authorization header. Expected 'Bearer <token>'.",
        )

    user = VALID_TOKENS.get(token.strip())
    if user is None:
        raise Auth.exceptions.HTTPException(
            status_code=401,
            detail="Invalid or unknown API token.",
        )

    role = user["role"]

    # MinimalUserDict: identity + optional metadata/permissions
    # (we just return a plain dict; LangGraph wraps it for you)
    return {
        "identity": user["identity"],
        "permissions": [role],         # you can read this later if needed
        "metadata": {"role": role},    # useful for RBAC / logging, etc.
    }


# ---------------------------------------------------------
# 3) Authorization (threads) – for now allow everything
# ---------------------------------------------------------
@auth.on.threads
async def authorize_threads(
    ctx: Auth.types.AuthContext,
    value: dict,
) -> dict:
    """
    Called whenever the client interacts with the `threads` resource.

    For now, we **do not restrict anything** – all roles can do all actions.
    Later you can inspect `ctx.user` and enforce stricter rules.

    Example:
        role = (getattr(ctx.user, "metadata", {}) or {}).get("role")
        if role != "admin" and ctx.action == "delete":
            raise Auth.exceptions.HTTPException(...)

    Returning `{}` means "no additional filters".
    """
    # If you want to start tagging threads with owner/role, you can uncomment this:

    # metadata = value.setdefault("metadata", {})
    # metadata.setdefault("owner", ctx.user.identity)
    # user_meta = getattr(ctx.user, "metadata", {}) or {}
    # role = user_meta.get("role")
    # if role:
    #     metadata.setdefault("role", role)

    return  # no restriction / filtering for now

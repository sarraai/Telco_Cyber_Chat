from __future__ import annotations

import os
import httpx
from langgraph_sdk import Auth

auth = Auth()

# Loaded from env (LangSmith app env or local .env)
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

# Only these roles are allowed in your Telco app
ALLOWED_ROLES = {"admin", "it_specialist", "network_admin", "end_user"}


@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    """
    Validate JWT tokens with Supabase and extract user information + telco role.
    """

    if authorization is None:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Missing Authorization header"
        )

    # Expect "Bearer <token>"
    try:
        scheme, token = authorization.split()
    except ValueError:
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Invalid Authorization header format"
        )

    if scheme.lower() != "bearer":
        raise Auth.exceptions.HTTPException(
            status_code=401, detail="Authorization scheme must be Bearer"
        )

    try:
        # 1) Ask Supabase which user this token belongs to
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{SUPABASE_URL}/auth/v1/user",
                headers={
                    "Authorization": authorization,   # "Bearer <access_token>"
                    "apiKey": SUPABASE_SERVICE_KEY,   # secret project key
                },
            )

        if response.status_code != 200:
            raise Auth.exceptions.HTTPException(
                status_code=401,
                detail=f"Supabase auth failed: {response.status_code} {response.text}",
            )

        user = response.json()

        # 2) Extract / normalize role from metadata
        user_metadata = user.get("user_metadata") or {}
        app_metadata = user.get("app_metadata") or {}

        raw_role = (
            user_metadata.get("role")
            or app_metadata.get("role")
            or "end_user"
        )

        # Make sure it matches one of your 4 roles
        role = raw_role if raw_role in ALLOWED_ROLES else "end_user"

        return {
            "identity": user["id"],            # unique user id from Supabase
            "email": user.get("email"),
            "role": role,                      # admin / it_specialist / ...
            "is_authenticated": True,
        }

    except Auth.exceptions.HTTPException:
        # Already formatted; just bubble up
        raise
    except Exception as e:
        # Any other error → 401
        raise Auth.exceptions.HTTPException(status_code=401, detail=str(e))

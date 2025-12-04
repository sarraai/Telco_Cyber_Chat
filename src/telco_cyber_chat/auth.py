from __future__ import annotations
from typing import Dict
from langgraph_sdk import Auth

auth = Auth()

VALID_TOKENS: Dict[str, Dict] = {
    "admin-token-123": {
        "identity": "admin_user",
        "role": "admin",
    },
    "it-token-123": {
        "identity": "it_specialist_user",
        "role": "it_specialist",
    },
    "netadmin-token-123": {
        "identity": "network_admin_user",
        "role": "network_admin",
    },
    "enduser-token-123": {
        "identity": "end_user_user",
        "role": "end_user",
    },
}

@auth.authenticate
async def get_current_user(authorization: str | None) -> Auth.types.MinimalUserDict:
    if authorization is None:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Missing Authorization header")

    try:
        scheme, token = authorization.split()
    except ValueError:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid Authorization header format")

    if scheme.lower() != "bearer":
        raise Auth.exceptions.HTTPException(status_code=401, detail="Authorization scheme must be Bearer")

    if token not in VALID_TOKENS:
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid token")

    user_data = VALID_TOKENS[token]
    # This dict is what LangGraph uses as "user"
    return {
        "identity": user_data["identity"],
        "role": user_data["role"],
    }

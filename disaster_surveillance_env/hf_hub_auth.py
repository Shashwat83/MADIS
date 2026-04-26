from __future__ import annotations

import os
from typing import Optional


def hub_token(explicit_token: Optional[str] = None) -> Optional[str]:
    """Return a Hugging Face Hub token when configured.

    We intentionally keep this as a pure env-var lookup so library code does not
    depend on Colab-specific secret vault mechanisms (which can time out unless
    running from the Colab UI).
    """

    token = explicit_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    token = (token or "").strip()
    return token or None


def from_pretrained_token_kwargs(explicit_token: Optional[str] = None) -> dict:
    """Build kwargs for transformers/datasets Hub calls.

    When no token is configured, we return an empty dict to keep behavior
    explicitly unauthenticated (but stable and predictable).
    """

    token = hub_token(explicit_token)
    return {"token": token} if token else {}


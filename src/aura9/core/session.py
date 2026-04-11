"""Session identity management for Aura-9.

Every session receives a unique ID stamped on all data:
    sess-{uuid4}-{unix_timestamp}
"""

from __future__ import annotations

import time
import uuid


def generate_session_id() -> str:
    """Return a new session ID in the canonical format."""
    return f"sess-{uuid.uuid4()}-{int(time.time())}"

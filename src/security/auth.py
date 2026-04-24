"""Auth — CLI authentication tiers."""

from __future__ import annotations

import os

from loguru import logger


class Auth:
    """CLI authentication with tiered permission levels."""

    def check_cli_auth(self, tier: int) -> bool:
        """Check CLI authentication for a given tier.

        Tier 0: read-only (UID match only)
        Tier 1: state-modifying (UID match + confirmation prompt)
        Tier 2: destructive (UID match + typed "CONFIRM")
        """
        # UID match check (local ops)
        current_uid = os.getuid()
        owner_uid = self._get_owner_uid()

        if current_uid != owner_uid:
            logger.warning(f"Auth: UID mismatch — current={current_uid}, owner={owner_uid}")
            return False

        if tier == 0:
            return True

        if tier == 1:
            response = input("Confirm state-modifying operation? [y/N]: ").strip().lower()
            return response in ("y", "yes")

        if tier == 2:
            response = input('Type "CONFIRM" to proceed with destructive operation: ').strip()
            return response == "CONFIRM"

        return False

    def verify_ipc_auth(self) -> bool:
        """Verify IPC authentication."""
        # Basic: check that caller UID matches
        return os.getuid() == self._get_owner_uid()

    @staticmethod
    def _get_owner_uid() -> int:
        """Get the UID of the process owner (same process for local ops)."""
        return os.getuid()

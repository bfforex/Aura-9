"""Audit Trail — JSONL append-only audit log with PII scrubbing."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from loguru import logger


class AuditTrail:
    """Writes JSONL audit events with PII scrubbing and monthly compression."""

    def __init__(self, audit_path: str = "./logs/audit.log", pii_scrubber=None) -> None:
        self._path = Path(audit_path)
        self._scrubber = pii_scrubber
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            logger.warning(f"AuditTrail: could not create audit dir: {exc}")

    async def write(
        self,
        event_type: str,
        data: dict,
        session_id: str,
        task_id: str | None = None,
        confidence: float | None = None,
        failure_class: str | None = None,
        watchdog_status: str = "CLEAR",
        tais_status: str = "NORMAL",
    ) -> None:
        """Append an audit event to the JSONL log."""
        detail = json.dumps(data)

        if self._scrubber:
            detail, _ = self._scrubber.scrub(detail)

        result_hash = hashlib.sha256(detail.encode()).hexdigest()

        event = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "session_id": session_id,
            "task_id": task_id,
            "action_type": event_type,
            "action_detail": detail,
            "result_hash": result_hash,
            "confidence_score": confidence,
            "failure_class": failure_class,
            "watchdog_status": watchdog_status,
            "tais_status": tais_status,
        }

        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as exc:
            logger.error(f"AuditTrail: write failed: {exc}")

    def compress_monthly(self) -> None:
        """Compress audit log to .gz (called on monthly schedule)."""
        if not self._path.exists():
            return
        month = datetime.now(UTC).strftime("%Y-%m")
        archive = self._path.with_suffix(f".{month}.log.gz")
        try:
            with open(self._path, "rb") as f_in:
                with gzip.open(archive, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            os.truncate(self._path, 0)
            logger.info(f"AuditTrail: compressed to {archive}")
        except Exception as exc:
            logger.error(f"AuditTrail: compress failed: {exc}")

"""Financial Gate — dual-confirmation for financial transactions."""

from __future__ import annotations

from loguru import logger

_FINANCIAL_SERVER_PREFIXES = ["stripe", "paypal", "plaid", "banking", "payment", "finance"]
_FINANCIAL_TOOL_KEYWORDS = [
    "transfer", "pay", "charge", "refund", "withdraw", "deposit",
    "purchase", "transaction", "invoice", "billing",
]
_FINANCIAL_ARG_KEYS = ["amount", "currency", "account", "card_number", "routing", "iban"]


class FinancialGate:
    """Detects and gates financial transaction tool calls."""

    def __init__(self, audit_trail=None, human_gate=None) -> None:
        self._audit = audit_trail
        self._human_gate = human_gate

    def check(self, server_id: str, tool_name: str, arguments: dict) -> bool:
        """Return True if this call involves a financial action."""
        server_lower = server_id.lower()
        if any(server_lower.startswith(p) for p in _FINANCIAL_SERVER_PREFIXES):
            return True

        tool_lower = tool_name.lower()
        if any(kw in tool_lower for kw in _FINANCIAL_TOOL_KEYWORDS):
            return True

        arg_keys_lower = [k.lower() for k in arguments.keys()]
        if any(fk in arg_keys_lower for fk in _FINANCIAL_ARG_KEYS):
            return True

        return False

    async def request_confirmation(self, action_details: dict) -> bool:
        """Request dual-confirmation for a financial action."""
        logger.warning(f"FinancialGate: requiring confirmation for: {action_details}")

        if self._audit:
            await self._audit.write(
                event_type="FINANCIAL_GATE_REQUEST",
                data=action_details,
                session_id=action_details.get("session_id", ""),
            )

        if self._human_gate:
            response = await self._human_gate.request(
                question=f"FINANCIAL TRANSACTION REQUIRES APPROVAL:\n{action_details}",
                context="Dual-confirmation required for financial action",
                task_id=action_details.get("task_id", ""),
                session_id=action_details.get("session_id", ""),
            )
            approved = response.approved
        else:
            # No gate available — block by default
            approved = False

        if self._audit:
            await self._audit.write(
                event_type="FINANCIAL_GATE_RESULT",
                data={**action_details, "approved": approved},
                session_id=action_details.get("session_id", ""),
            )

        return approved

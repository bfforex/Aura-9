"""Unit tests for the PII Scrubber."""

from __future__ import annotations

import pytest

from src.security.pii_scrubber import PIIScrubber


@pytest.mark.unit
class TestPIIScrubber:
    def setup_method(self):
        self.scrubber = PIIScrubber()

    def test_email_scrubbed(self):
        text = "Contact john.doe@example.com for support"
        result, categories = self.scrubber.scrub(text)
        assert "[EMAIL_REDACTED]" in result
        assert "john.doe@example.com" not in result
        assert "email" in categories

    def test_phone_scrubbed(self):
        text = "Call us at 555-123-4567"
        result, categories = self.scrubber.scrub(text)
        assert "[PHONE_REDACTED]" in result
        assert "555-123-4567" not in result
        assert "phone" in categories

    def test_ssn_scrubbed(self):
        text = "SSN: 123-45-6789"
        result, categories = self.scrubber.scrub(text)
        assert "[SSN_REDACTED]" in result
        assert "123-45-6789" not in result
        assert "ssn" in categories

    def test_credit_card_scrubbed(self):
        text = "Card: 4532 1234 5678 9012"
        result, categories = self.scrubber.scrub(text)
        assert "[CARD_REDACTED]" in result
        assert "4532 1234 5678 9012" not in result
        assert "credit_card" in categories

    def test_ip_address_scrubbed(self):
        text = "Server at 192.168.1.100 is down"
        result, categories = self.scrubber.scrub(text)
        assert "[IP_REDACTED]" in result
        assert "192.168.1.100" not in result
        assert "ip_address" in categories

    def test_clean_text_unchanged(self):
        text = "Hello world, this is a normal message"
        result, categories = self.scrubber.scrub(text)
        assert result == text
        assert categories == []

    def test_multiple_categories_in_one_text(self):
        text = "Email: user@test.com, Phone: 555-987-6543, SSN: 987-65-4321"
        result, categories = self.scrubber.scrub(text)
        assert "email" in categories
        assert "phone" in categories
        assert "ssn" in categories
        assert "[EMAIL_REDACTED]" in result
        assert "[PHONE_REDACTED]" in result
        assert "[SSN_REDACTED]" in result

    def test_returns_tuple(self):
        result = self.scrubber.scrub("test")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_categories_is_list(self):
        _, categories = self.scrubber.scrub("no pii here")
        assert isinstance(categories, list)

    def test_email_with_plus_sign(self):
        text = "user+tag@domain.co.uk is the email"
        result, categories = self.scrubber.scrub(text)
        assert "email" in categories

    def test_credit_card_with_dashes(self):
        text = "Card number: 1234-5678-9012-3456"
        result, categories = self.scrubber.scrub(text)
        assert "credit_card" in categories

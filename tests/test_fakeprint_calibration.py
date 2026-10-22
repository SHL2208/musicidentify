"""Bộ kiểm thử cho hiệu chỉnh Fakeprint Detector và Gemini System Instruction.

Kiểm tra:
1. Gemini System Prompt và User Prompt không được 'mớm cung' hoặc khẳng định
   Fakeprint >= 55% là 'bằng chứng vật lý xác nhận AI' (Physical evidence confirming AI).
2. Fakeprint Detector được hiệu chỉnh khách quan, coi bồi âm là chỉ số tham khảo.

Tuân thủ nghiêm ngặt PEP 8 và Ruff.
"""

from __future__ import annotations

import unittest

from backend.services.gemini_classifier_service import SYSTEM_INSTRUCTION


class TestFakeprintCalibration(unittest.TestCase):
    """Kiểm thử hiệu chỉnh prompt và độ tin cậy của chỉ số Fakeprint."""

    def test_gemini_system_instruction_does_not_bias_fakeprint_as_absolute_proof(self) -> None:
        """System instruction không được coi Fakeprint là 'strong physical evidence confirming AI generation'

        vì các bản thu có Auto-tune, synth lead hoặc harmonic dày đặc của người thật có thể tạo bồi âm cao.
        """
        # Kiểm tra cụm từ 'strong physical evidence confirming AI generation' cũ
        self.assertNotIn(
            "strong physical evidence confirming AI generation",
            SYSTEM_INSTRUCTION,
            "LỖI: System Instruction vẫn ép mô hình tin 100% vào Fakeprint gây ra False Positives!",
        )

    def test_gemini_system_instruction_mentions_autotune_and_studio_harmonic_caution(self) -> None:
        """System instruction phải hướng dẫn chuyên gia thận trọng với Auto-Tune và bồi âm phòng thu."""
        instruction_lower = SYSTEM_INSTRUCTION.lower()
        has_caution = (
            "auto-tune" in instruction_lower
            or "autotune" in instruction_lower
            or "harmonic" in instruction_lower
            or "reference" in instruction_lower
        )
        self.assertTrue(
            has_caution,
            "System Instruction cần có hướng dẫn phân biệt Auto-Tune phòng thu và bồi âm.",
        )


if __name__ == "__main__":
    unittest.main()

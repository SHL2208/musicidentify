"""Bộ kiểm thử TDD cho các tính năng nhận diện AI Vocals đời mới, Nhạc không lời và Tinh chỉnh Safeguards.

Tuân thủ nghiêm ngặt PEP 8, Ruff và quy trình Red-Green-Refactor.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.services.detection_service import DetectionService
from backend.services.gemini_classifier_service import GeminiClassifierService
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestModernAIVocalsAndSafeguards(unittest.TestCase):
    """Test suite cho nhận diện AI Vocals đời mới, Nhạc không lời và Safeguard Anti-FN/Anti-FP."""

    def setUp(self) -> None:
        self.mock_audio_engine = MagicMock()
        self.mock_signal_analyzer = MagicMock()
        self.mock_classifier = MagicMock()
        self.mock_resolver = MagicMock()
        self.mock_track_repo = MagicMock()

        self.detection_service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.mock_track_repo,
        )

    def test_acoustic_metrics_entity_has_modern_ai_and_instrumental_fields(self) -> None:
        """Kiểm tra AcousticMetrics có đầy đủ các trường mới: is_instrumental, vocal_energy_ratio, modern_ai_vocal_suspect."""
        metrics = AcousticMetrics(
            duration_sec=30.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=9500.0,
            spectral_rolloff_85_hz=6000.0,
            stereo_correlation=0.9,
            harmonic_ratio=0.88,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_rigidity_ratio=0.58,
            pitch_jitter=0.08,
            fakeprint_score=0.25,
            fakeprint_detected=False,
            is_instrumental=False,
            vocal_energy_ratio=0.45,
            modern_ai_vocal_suspect=True,
        )
        self.assertFalse(metrics.is_instrumental)
        self.assertAlmostEqual(metrics.vocal_energy_ratio, 0.45)
        self.assertTrue(metrics.modern_ai_vocal_suspect)

    def test_safeguard_anti_fn_catches_chang_can_ai_thuong_nua(self) -> None:
        """Case A: Nhạc AI đời mới có Ultrasonic Cutoff + Rigidity cao + Harmonic cao nhưng Fakeprint thấp.

        Phải được Safeguard Anti-FN nâng cấp từ PROCESSED_HUMAN lên HYBRID_AI (is_ai=True).
        Ví dụ thực tế: 'Chẳng Cần Ai Thương Nữa - Lý Nguyên Tiêu' (Fakeprint: 0.195, Cutoff: True, Rigidity: 0.63, Rolloff: 11.3kHz).
        """
        track = TrackItem(
            id="trk_test_lynguyentieu",
            raw_query="Chẳng Cần Ai Thương Nữa Lý Nguyên Tiêu",
            track_title="Chẳng Cần Ai Thương Nữa",
            artist="Lý Nguyên Tiêu",
            input_type=InputType.SONG_NAME,
            distributor="Mixus",
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=11333.4,
            spectral_rolloff_85_hz=7377.7,
            stereo_correlation=1.0,
            harmonic_ratio=0.894,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_rigidity_ratio=0.63,
            pitch_jitter=0.0738,
            fakeprint_score=0.195,  # Fakeprint thấp, không đạt ngưỡng cũ 0.65
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=True,  # Có Ultrasonic Cutoff!
            is_instrumental=False,
            modern_ai_vocal_suspect=True,
        )
        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title or track.raw_query,
            source_url="https://youtube.com/watch?v=xL_xKxuc4XE",
            input_type=track.input_type,
            raw_query=track.raw_query,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.94,
            suspected_engine="None",
            reasons=["Vocal features authentic human breath dynamics"],
            detailed_explanation="Human studio production",
            acoustic_metrics=metrics,
        )

        final_result = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={"distributor": "Mixus", "title": "Chẳng Cần Ai Thương Nữa"},
            metrics=metrics,
        )

        # PHẢI ĐƯỢC NÂNG CẤP LÊN AI!
        self.assertTrue(
            final_result.is_ai, "Phải nâng cấp thành AI do có Ultrasonic Cutoff + Rigidity cao"
        )
        self.assertIn(final_result.verdict_type, [VerdictType.HYBRID_AI, VerdictType.AI_GENERATED])
        self.assertTrue(
            any("CẢNH BÁO ÂM HỌC VẬT LÝ" in r or "AI" in r for r in final_result.reasons)
        )

    def test_safeguard_anti_fn_catches_nguoi_tot_nhat_tren_doi(self) -> None:
        """Case B: Nhạc AI đời mới có Sub-10kHz Rolloff sụt giảm nặng + Rigidity cao + Harmonic cao.

        Ví dụ thực tế: 'Người Tốt Nhất Trên Đời - Huyền Tranng' (Rolloff 95%: 9479 Hz, Rigidity: 0.568, Harmonic: 0.909, Fakeprint: 0.427).
        """
        track = TrackItem(
            id="trk_test_huyentranng",
            raw_query="Người Tốt Nhất Trên Đời Huyền Tranng",
            track_title="Người Tốt Nhất Trên Đời",
            artist="Huyền Tranng",
            input_type=InputType.SONG_NAME,
            distributor="Believe",
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=9479.9,  # Sụt giảm sâu dưới 10kHz!
            spectral_rolloff_85_hz=5744.4,
            stereo_correlation=1.0,
            harmonic_ratio=0.909,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_rigidity_ratio=0.568,  # Rigidity > 50%
            pitch_jitter=0.1098,
            fakeprint_score=0.427,  # Fakeprint không đạt ngưỡng cũ 0.70
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=False,
            is_instrumental=False,
            modern_ai_vocal_suspect=True,
        )
        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title or track.raw_query,
            source_url="https://youtube.com/watch?v=U3vUv7X0Mdc",
            input_type=track.input_type,
            raw_query=track.raw_query,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.94,
            suspected_engine="None",
            reasons=["Human vocal performance with Auto-tune"],
            detailed_explanation="Human studio production",
            acoustic_metrics=metrics,
        )

        final_result = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={"distributor": "Believe", "title": "Người Tốt Nhất Trên Đời"},
            metrics=metrics,
        )

        # PHẢI ĐƯỢC NÂNG CẤP LÊN AI!
        self.assertTrue(
            final_result.is_ai, "Phải nâng cấp thành AI do Rolloff < 10kHz và Rigidity cao"
        )
        self.assertIn(final_result.verdict_type, [VerdictType.HYBRID_AI, VerdictType.AI_GENERATED])

    def test_safeguard_upgrades_nguoi_tot_nhat_tren_doi_remix(self) -> None:
        """Bản Remix của Người Tốt Nhất Trên Đời dù có beat EDM vẫn bị bắt vì giọng AI có Rolloff < 11.5kHz và Fakeprint."""
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=11395.4,
            spectral_rolloff_85_hz=7890.4,
            pitch_rigidity_ratio=0.473,
            pitch_jitter_score=0.1024,
            harmonic_ratio=0.829,
            fakeprint_score=0.424,
            stem_bleed_score=0.326,
            ultrasonic_cutoff_detected=False,
            modern_ai_vocal_suspect=True,
        )
        track = TrackItem(
            id="trk_remix",
            track_title="Người Tốt Nhất Trên Đời (DJ Trang Moon Remix)",
            raw_query="Người Tốt Nhất Trên Đời (DJ Trang Moon Remix)",
            input_type=InputType.SONG_NAME,
            distributor="Believe",
        )
        gemini_result = DetectionResult(
            track_id="trk_remix",
            track_title="Người Tốt Nhất Trên Đời (DJ Trang Moon Remix)",
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.92,
            suspected_engine="None",
            reasons=["Vocal track separate from EDM beat"],
            detailed_explanation="Remix production",
            acoustic_metrics=metrics,
        )

        final_result = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={
                "distributor": "Believe",
                "title": "Người Tốt Nhất Trên Đời (DJ Trang Moon Remix)",
            },
            metrics=metrics,
        )

        self.assertTrue(
            final_result.is_ai,
            "Bản Remix có vocal AI bị bóp trần tần số và dính fakeprint phải bắt thành AI",
        )
        self.assertIn(final_result.verdict_type, [VerdictType.HYBRID_AI, VerdictType.AI_GENERATED])

    def test_incom_heuristics_does_not_falsely_flag_real_artist(self) -> None:
        """Heuristics INCOM không được vu oan cho ca khúc pop có ca sĩ độc lập xác thực (như 'Hồi Tưởng - Dytins Nguyễn')."""
        mock_fp = MagicMock()
        mock_storage = MagicMock()
        mock_storage.get_temp_dir.return_value = MagicMock()
        analyzer = SignalAnalyzerService(fakeprint_detector=mock_fp, storage_repo=mock_storage)
        metadata_incom_real_artist = {
            "distributor": "INCOM",
            "title": "Hồi Tưởng",
            "artist": "Dytins Nguyễn",
            "target_artist": "Dytins Nguyễn",
            "target_title": "Hồi Tưởng",
        }
        res = analyzer.check_heuristics(metadata_incom_real_artist)
        # Không được phán cứng AI 100% nếu là ca sĩ độc lập cá nhân
        self.assertIsNone(
            res,
            "Bài hát có ca sĩ riêng biệt phân phối qua INCOM phải để audio forensic kiểm tra, không được phán cứng AI",
        )

    def test_gemini_prompt_contains_modern_ai_and_instrumental_directives(self) -> None:
        """Prompt Gemini phải chứa chỉ thị nghiêm khắc về hơi thở giả lập của AI và chỉ thị nhạc không lời."""
        classifier = GeminiClassifierService(api_keys=["dummy_key"])
        metrics_ai_vocal = AcousticMetrics(
            duration_sec=30.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=9500.0,
            spectral_rolloff_85_hz=6000.0,
            stereo_correlation=0.9,
            harmonic_ratio=0.88,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_rigidity_ratio=0.58,
            pitch_jitter=0.08,
            fakeprint_score=0.25,
            fakeprint_detected=False,
            is_instrumental=False,
            modern_ai_vocal_suspect=True,
        )
        prompt_vocal = classifier._build_forensic_prompt(
            track_id="test_01", metrics=metrics_ai_vocal, metadata={}
        )
        self.assertIn("MODERN AI VOCAL", prompt_vocal)
        self.assertIn("synthetic breath", prompt_vocal.lower())

        metrics_instrumental = AcousticMetrics(
            duration_sec=30.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=16000.0,
            spectral_rolloff_85_hz=12000.0,
            stereo_correlation=0.9,
            harmonic_ratio=0.60,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            pitch_stability_label="Natural",
            pitch_rigidity_ratio=0.20,
            pitch_jitter=0.15,
            fakeprint_score=0.10,
            fakeprint_detected=False,
            is_instrumental=True,
            modern_ai_vocal_suspect=False,
        )
        prompt_inst = classifier._build_forensic_prompt(
            track_id="test_02", metrics=metrics_instrumental, metadata={}
        )
        self.assertIn("INSTRUMENTAL", prompt_inst)

    def test_safeguard_protects_dinh_tung_huy_from_false_positive(self) -> None:
        """Bảo vệ ca khúc của ca sĩ thật Đinh Tùng Huy ('Áo Chật Đem Lau') không bị vu oan AI dù có Autotune và nén YouTube."""
        track = TrackItem(
            id="trk_dinh_tung_huy",
            track_title="Áo Chật Đem Lau",
            artist="Đinh Tùng Huy",
            raw_query="Áo Chật Đem Lau Đinh Tùng Huy",
            input_type=InputType.SONG_NAME,
            distributor="Believe",
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=8730.0,
            spectral_rolloff_85_hz=4402.4,
            pitch_rigidity_ratio=0.626,
            harmonic_ratio=0.882,
            fakeprint_score=0.0,
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=True,
            stem_bleed_score=0.18,
            compression_source="Opus/AAC (YouTube/Web Streaming Lossy)",
            has_brickwall_cutoff=True,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_jitter=0.08,
            is_instrumental=False,
        )
        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.96,
            suspected_engine="Suno",
            reasons=["High vocal pitch rigidity and high-frequency rolloff below 9kHz"],
            detailed_explanation="Forensic check",
            acoustic_metrics=metrics,
        )

        final_res = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={"artist": "Đinh Tùng Huy", "distributor": "Believe"},
            metrics=metrics,
        )

        self.assertFalse(
            final_res.is_ai,
            "Ca sĩ thực tế Đinh Tùng Huy với Fakeprint 0.0 phải được Safeguard bảo vệ là PROCESSED_HUMAN",
        )
        self.assertEqual(final_res.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_safeguard_protects_minh_sang_from_false_positive(self) -> None:
        """Bảo vệ ca khúc ballad truyền thống của ca sĩ Minh Sang ('Ngàn Năm Hồn Việt') không bị vu oan AI."""
        track = TrackItem(
            id="trk_minh_sang",
            track_title="Ngàn Năm Hồn Việt",
            artist="Minh Sang",
            raw_query="Ngàn Năm Hồn Việt Minh Sang",
            input_type=InputType.SONG_NAME,
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=10375.2,
            spectral_rolloff_85_hz=6013.7,
            pitch_rigidity_ratio=0.563,
            harmonic_ratio=0.77,
            fakeprint_score=0.406,
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=False,
            stem_bleed_score=0.22,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=False,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_jitter=0.09,
            is_instrumental=False,
        )
        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.96,
            suspected_engine="Suno",
            reasons=["Rolloff below 12kHz"],
            detailed_explanation="Forensic check",
            acoustic_metrics=metrics,
        )

        final_res = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={"artist": "Minh Sang"},
            metrics=metrics,
        )

        self.assertFalse(final_res.is_ai)
        self.assertEqual(final_res.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_safeguard_protects_tlong_remix_from_false_positive(self) -> None:
        """Bảo vệ bản EDM Remix của nghệ sĩ TLong ('Xin Đừng Thương Hại Anh') không bị phán HYBRID_AI oan."""
        track = TrackItem(
            id="trk_tlong_remix",
            track_title="Xin Đừng Thương Hại Anh (Tuấn Siêu x Tón Remix)",
            artist="Ducth Music, TLong",
            raw_query="Xin Đừng Thương Hại Anh (Tuấn Siêu x Tón Remix) Ducth Music, TLong",
            input_type=InputType.SONG_NAME,
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=8820.5,
            spectral_rolloff_85_hz=5500.0,
            pitch_rigidity_ratio=0.582,
            harmonic_ratio=0.747,
            fakeprint_score=0.0,
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=True,
            stem_bleed_score=0.20,
            compression_source="Opus/AAC",
            has_brickwall_cutoff=True,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_jitter=0.07,
            is_instrumental=False,
        )
        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=True,
            verdict_type=VerdictType.HYBRID_AI,
            verdict_label="HYBRID / AI-ASSISTED (Bán nhân tạo)",
            confidence_score=0.95,
            suspected_engine="Suno/Beat",
            reasons=["[HYBRID AI]: Thành phần Vocal là AI | Thành phần Beat/Hòa âm là AI sinh tự động"],
            detailed_explanation="Forensic check",
            acoustic_metrics=metrics,
        )

        final_res = self.detection_service._apply_cross_validation_safeguard(
            result=gemini_result,
            track=track,
            metadata={"artist": "Ducth Music, TLong", "title": "Xin Đừng Thương Hại Anh (Tuấn Siêu x Tón Remix)"},
            metrics=metrics,
        )

        self.assertFalse(final_res.is_ai)
        self.assertEqual(final_res.verdict_type, VerdictType.PROCESSED_HUMAN)

    def test_verified_artist_token_matching_protects_vu_without_false_matching_anh_vu(self) -> None:
        """Kiểm tra logic tách token nghệ sĩ: 'Vũ' và 'Vũ.' là nghệ sĩ thật, nhưng 'Anh Vũ' (dự án AI) tuyệt đối không được coi là nghệ sĩ thật."""
        from backend.services.signal_analyzer_service import is_verified_human_artist

        self.assertTrue(is_verified_human_artist("Vũ", "Bước Qua Mùa Cô Đơn"))
        self.assertTrue(is_verified_human_artist("Vũ.", "Lạ Lùng"))
        self.assertTrue(is_verified_human_artist("Vũ x Đen", "Một Triệu Khả Năng"))
        self.assertTrue(is_verified_human_artist("Hoàng Khang, Mihle", "Vết Thương"))

        # Các dự án AI tuyệt đối trả về False
        self.assertFalse(is_verified_human_artist("Anh Vũ", "Mùa Mưa Đã Qua"))
        self.assertFalse(is_verified_human_artist("Anh Vũ", "Ly Cà Phê Đắng"))
        self.assertFalse(is_verified_human_artist("Trịnh Nguyên Bình", "Ước Vọng Trăng Tròn"))
        self.assertFalse(is_verified_human_artist("Huyền Tranng", "Người Tốt Nhất Trên Đời"))
        self.assertFalse(is_verified_human_artist("Lý Nguyên Tiêu", "Chẳng Cần Ai Thương Nữa"))

    def test_rule_1c_does_not_downgrade_batch_or_catalog_ai_even_with_low_fakeprint(self) -> None:
        """Rule 1c không được hạ cấp các bài hát đã có chỉ báo phát hành hàng loạt ('bài số 82/100') hoặc thuộc catalog AI."""
        track = TrackItem(
            id="trk_batch_ai",
            track_title="Em Và Thu - HuyH (Bài số 82/100)",
            artist="HuyH",
            raw_query="Em Và Thu HuyH",
            input_type=InputType.SONG_NAME,
        )
        metrics = AcousticMetrics(
            duration_sec=45.0,
            sample_rate=44100,
            spectral_rolloff_95_hz=8480.4,
            spectral_rolloff_85_hz=5200.0,
            pitch_rigidity_ratio=0.613,
            harmonic_ratio=0.949,
            fakeprint_score=0.21,  # Thấp < 0.25
            fakeprint_detected=False,
            ultrasonic_cutoff_detected=True,
            stem_bleed_score=0.027,  # Rất sạch < 0.25
            compression_source="Opus/AAC",
            has_brickwall_cutoff=True,
            pitch_stability_label="Studio Auto-Tuned",
            pitch_jitter=0.07,
            is_instrumental=False,
        )
        batch_ai_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.98,
            suspected_engine="AI Project/Batch",
            reasons=["Phát hiện định dạng phát hành hàng loạt đặc trưng của dự án nhạc AI: 'bài số 82/100'."],
            detailed_explanation="Heuristic detection",
            acoustic_metrics=metrics,
        )

        final_res = self.detection_service._apply_cross_validation_safeguard(
            result=batch_ai_result,
            track=track,
            metadata={"title": track.track_title, "artist": "HuyH"},
            metrics=metrics,
        )

        self.assertTrue(
            final_res.is_ai,
            "Bài hát thuộc dự án phát hành hàng loạt 'bài số 82/100' không được bị Rule 1c hạ cấp về Human!",
        )
        self.assertEqual(final_res.verdict_type, VerdictType.AI_GENERATED)


if __name__ == "__main__":
    unittest.main()


"""Unit tests for Dual Output Columns (Model AI vs Consensus) and Refined Safeguards.

Follows TDD RED-GREEN-REFACTOR and strict PEP 8 / Ruff compliance.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import openpyxl

from backend.core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    TrackItem,
    VerdictType,
)
from backend.repositories.file_report_repository import FileReportRepository
from backend.services.detection_service import DetectionService
from backend.services.report_service import ReportService
from backend.services.signal_analyzer_service import SignalAnalyzerService


class TestDualOutputAndSafeguards(unittest.TestCase):
    def setUp(self):
        self.mock_resolver = MagicMock()
        self.mock_audio_engine = MagicMock()
        self.mock_signal_analyzer = MagicMock(spec=SignalAnalyzerService)
        self.mock_classifier = MagicMock()
        self.mock_storage = MagicMock()
        self.mock_storage.find_result_by_query_or_url.return_value = None

        self.detection_service = DetectionService(
            audio_engine=self.mock_audio_engine,
            signal_analyzer=self.mock_signal_analyzer,
            classifier=self.mock_classifier,
            resolver=self.mock_resolver,
            track_repo=self.mock_storage,
        )

        self.real_signal_analyzer = SignalAnalyzerService(
            fakeprint_detector=MagicMock(),
            storage_repo=MagicMock(),
        )

    # -------------------------------------------------------------
    # 1. ENTITY ATTRIBUTES TEST
    # -------------------------------------------------------------
    def test_detection_result_has_dual_output_attributes(self):
        """DetectionResult entity must have ai_model_* and consensus_* attributes."""
        res = DetectionResult(
            track_id="t1",
            track_title="Test Track",
            is_ai=True,
            verdict_type=VerdictType.AI_GENERATED,
            verdict_label="AI-GENERATED (Nhạc AI tạo)",
            confidence_score=0.92,
            suspected_engine="Suno v3.5",
            reasons=["High pitch rigidity", "Lossy compression"],
            detailed_explanation="AI-generated track detected.",
            ai_model_is_ai=True,
            ai_model_verdict="AI_GENERATED",
            consensus_is_ai=True,
            consensus_verdict="AI_GENERATED",
        )
        self.assertTrue(hasattr(res, "ai_model_is_ai"))
        self.assertTrue(hasattr(res, "ai_model_verdict"))
        self.assertTrue(hasattr(res, "consensus_is_ai"))
        self.assertTrue(hasattr(res, "consensus_verdict"))
        self.assertTrue(res.ai_model_is_ai)
        self.assertEqual(res.ai_model_verdict, "AI_GENERATED")
        self.assertTrue(res.consensus_is_ai)
        self.assertEqual(res.consensus_verdict, "AI_GENERATED")

    # -------------------------------------------------------------
    # 2. HEURISTICS: NO HARD VERDICT FOR AI DISTRIBUTORS
    # -------------------------------------------------------------
    def test_distributor_heuristics_removed_for_pure_ai_distributors(self):
        """Distributors like INCOM and TTE Fast Considerate must NOT trigger early AI verdicts."""
        for dist in ["INCOM", "TTE Fast Considerate", "incom"]:
            meta = {
                "title": "Bản nhạc thử nghiệm",
                "artist": "Ca sĩ thử nghiệm",
                "distributor": dist,
            }
            res = self.real_signal_analyzer.check_heuristics(meta)
            self.assertIsNone(
                res,
                f"Distributor '{dist}' must NOT trigger early Heuristics AI verdict; "
                "it must be delegated to audio forensics + AI model!",
            )

    # -------------------------------------------------------------
    # 3. ACCENT-INSENSITIVE HUMAN ARTIST MATCHING
    # -------------------------------------------------------------
    def test_accent_insensitive_and_collaborator_artist_matching(self):
        """Verified human artists must be matched with or without Vietnamese accents."""
        test_cases = [
            ("Sara Lưu & JayKii", True),
            ("sara luu", True),
            ("Giang Hồng Ngọc", True),
            ("giang hong ngoc", True),
            ("Tuấn Phi", True),
            ("tuan phi", True),
            ("Âu Hoàng", True),
            ("au hoang", True),
            ("Arson, Khánh Hòa", True),
            ("arson, khanh hoa", True),
            ("Như Ý", True),
            ("nhu y", True),
            ("Toharmony", True),
            ("toharmony", True),
            ("Unknown AI Project 123", False),
        ]
        for artist_name, expected in test_cases:
            matched = self.real_signal_analyzer.is_verified_human_artist(artist_name)
            self.assertEqual(
                matched,
                expected,
                f"Artist '{artist_name}' expected {expected} for verified human match, got {matched}",
            )

    # -------------------------------------------------------------
    # 4. SAFEGUARD ANTI-FN: BIOLOGICAL EVIDENCE PROTECTION & HIGH BAR
    # -------------------------------------------------------------
    def test_safeguard_anti_fn_respects_human_biological_evidence(self):
        """Safeguard Anti-FN must NOT override when AI model found biological human resonance."""
        track = TrackItem(
            id="track_human_bio",
            raw_query="Bài hát người thật",
            input_type=InputType.SONG_NAME,
            track_title="Bài hát người thật",
            artist="Ca Sĩ Tự Do",
            distributor="Hãng Tự Do",
        )

        metrics = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            spectral_rolloff_95_hz=8500.0,  # Lossy cutoff < 10000 Hz
            spectral_rolloff_85_hz=7200.0,
            stereo_correlation=0.92,
            harmonic_ratio=0.91,  # Pop ballad harmonic ratio >= 0.88
            fakeprint_score=0.32,  # Low fakeprint from lossy YouTube
            fakeprint_detected=False,
            compression_source="Lossy Streaming (44.1kHz / 160kbps MP3)",
            has_brickwall_cutoff=False,
            pitch_jitter=0.015,
            pitch_rigidity_ratio=0.58,  # Studio Auto-Tune >= 0.50
            pitch_anomaly_detected=False,
            pitch_stability_label="Studio Auto-Tuned (Natural Vocal Tract)",
            vocal_formant_variance=310.0,
            is_studio_autotune=True,
            stem_bleed_score=0.12,  # Clean bleed < 0.35
            stem_bleed_detected=False,
        )

        gemini_result = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.88,
            suspected_engine="None",
            reasons=[
                "Giọng hát có cấu trúc thanh quản sinh học tự nhiên",
                "Hiệu ứng phòng thu Auto-Tune làm tăng độ cứng cao độ",
            ],
            detailed_explanation="Bản thu người thật với Auto-Tune phòng thu.",
        )

        self.mock_resolver.resolve_track.return_value = track
        self.mock_audio_engine.process_track_audio.return_value = ("dummy.wav", {})
        self.mock_signal_analyzer.analyze_audio.return_value = (metrics, Path("dummy.png"))
        self.mock_signal_analyzer.check_heuristics.return_value = None
        self.mock_classifier.classify_track.return_value = gemini_result

        final_res = self.detection_service.detect_single(track)

        # Must preserve Human verdict and populate dual columns
        self.assertFalse(
            final_res.is_ai,
            "Human biological resonance must NOT be overridden to AI by Anti-FN safeguard!",
        )
        self.assertEqual(final_res.verdict_type, VerdictType.PROCESSED_HUMAN)
        self.assertEqual(final_res.ai_model_verdict, "PROCESSED_HUMAN")
        self.assertFalse(final_res.ai_model_is_ai)
        self.assertEqual(final_res.consensus_verdict, "PROCESSED_HUMAN")
        self.assertFalse(final_res.consensus_is_ai)

    def test_safeguard_anti_fn_requires_dual_evidence_high_bar(self):
        """Safeguard Anti-FN overriding Human to AI requires fakeprint >= 0.50 AND stem_bleed >= 0.35."""
        track = TrackItem(
            id="track_high_bar",
            raw_query="Bài hát kiểm tra bar",
            input_type=InputType.SONG_NAME,
            track_title="Bài hát kiểm tra bar",
            artist="Nghệ Sĩ Mới",
        )

        # Case A: Fakeprint high (0.60) but Stem Bleed low (0.15) -> Do NOT override
        metrics_a = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            spectral_rolloff_95_hz=14000.0,
            spectral_rolloff_85_hz=11000.0,
            stereo_correlation=0.92,
            harmonic_ratio=0.82,
            fakeprint_score=0.60,
            fakeprint_detected=True,
            compression_source="Standard Digital Audio",
            has_brickwall_cutoff=False,
            pitch_jitter=0.015,
            pitch_rigidity_ratio=0.52,
            pitch_anomaly_detected=False,
            pitch_stability_label="Stable",
            vocal_formant_variance=310.0,
            is_studio_autotune=False,
            stem_bleed_score=0.15,
            stem_bleed_detected=False,
        )

        gemini_result_a = DetectionResult(
            track_id=track.id,
            track_title=track.track_title,
            is_ai=False,
            verdict_type=VerdictType.PROCESSED_HUMAN,
            verdict_label="PROCESSED HUMAN (Người làm - Kỹ thuật số)",
            confidence_score=0.85,
            suspected_engine="None",
            reasons=["Giọng hát người"],
            detailed_explanation="Bản thu người.",
        )

        self.mock_resolver.resolve_track.return_value = track
        self.mock_audio_engine.process_track_audio.return_value = ("dummy.wav", {})
        self.mock_signal_analyzer.analyze_audio.return_value = (metrics_a, Path("dummy.png"))
        self.mock_signal_analyzer.check_heuristics.return_value = None
        self.mock_classifier.classify_track.return_value = gemini_result_a

        res_a = self.detection_service.detect_single(track)
        self.assertFalse(
            res_a.is_ai,
            "Fakeprint alone without high stem bleed should NOT override to AI!",
        )

        # Case B: Both Fakeprint >= 0.50 (0.65) AND Stem Bleed >= 0.35 (0.40) -> Override to HYBRID_AI
        metrics_b = AcousticMetrics(
            sample_rate=44100,
            duration_sec=30.0,
            spectral_rolloff_95_hz=14000.0,
            spectral_rolloff_85_hz=11000.0,
            stereo_correlation=0.92,
            harmonic_ratio=0.82,
            fakeprint_score=0.65,
            fakeprint_detected=True,
            compression_source="Standard Digital Audio",
            has_brickwall_cutoff=False,
            pitch_jitter=0.015,
            pitch_rigidity_ratio=0.52,
            pitch_anomaly_detected=False,
            pitch_stability_label="Stable",
            vocal_formant_variance=310.0,
            is_studio_autotune=False,
            stem_bleed_score=0.40,
            stem_bleed_detected=True,
        )
        self.mock_signal_analyzer.analyze_audio.return_value = (metrics_b, Path("dummy.png"))
        res_b = self.detection_service.detect_single(track)
        self.assertTrue(res_b.is_ai)
        self.assertEqual(res_b.verdict_type, VerdictType.HYBRID_AI)
        self.assertEqual(res_b.ai_model_verdict, "PROCESSED_HUMAN")
        self.assertFalse(res_b.ai_model_is_ai)
        self.assertEqual(res_b.consensus_verdict, "HYBRID_AI")
        self.assertTrue(res_b.consensus_is_ai)

    # -------------------------------------------------------------
    # 5. DUAL OUTPUT COLUMNS IN REPORT SERVICE (EXCEL EXPORT)
    # -------------------------------------------------------------
    def test_report_service_exports_dual_verdict_columns(self):
        """Excel report must contain both 'Đánh giá Model AI' and 'Kết luận Tổng hợp' columns."""
        repo = FileReportRepository(output_dir=Path("data/temp"))
        report_service = ReportService(report_repo=repo)

        sample_results = [
            DetectionResult(
                track_id="t1",
                track_title="Song A",
                is_ai=True,
                verdict_type=VerdictType.AI_GENERATED,
                verdict_label="AI-GENERATED",
                confidence_score=0.95,
                suspected_engine="Suno v3.5",
                reasons=["AI voice"],
                detailed_explanation="AI track",
                ai_model_is_ai=True,
                ai_model_verdict="AI_GENERATED",
                consensus_is_ai=True,
                consensus_verdict="AI_GENERATED",
            ),
            DetectionResult(
                track_id="t2",
                track_title="Song B",
                is_ai=False,
                verdict_type=VerdictType.PROCESSED_HUMAN,
                verdict_label="PROCESSED_HUMAN",
                confidence_score=0.88,
                suspected_engine="None",
                reasons=["Human voice"],
                detailed_explanation="Human track",
                ai_model_is_ai=False,
                ai_model_verdict="PROCESSED_HUMAN",
                consensus_is_ai=False,
                consensus_verdict="PROCESSED_HUMAN",
            ),
        ]

        export_path = report_service.export_report(sample_results, format="xlsx")
        self.assertIsNotNone(export_path)

        wb = openpyxl.load_workbook(export_path)
        ws = wb.active
        headers = [cell.value for cell in ws[1]]

        self.assertIn(
            "Đánh giá Model AI",
            headers,
            "Excel report must contain 'Đánh giá Model AI' header column!",
        )
    # -------------------------------------------------------------
    # 6. BENCHMARK EVALUATOR DATASET FALLBACK & DUAL COLUMNS
    # -------------------------------------------------------------
    def test_benchmark_load_testcase_dataset_fallback_to_data_dir(self):
        """Benchmark loader must fallback to searching in data/ directory if not found in root."""
        from benchmark_evaluator import load_testcase_dataset

        # Call with relative filename 'testcase.xlsx' (which is in data/testcase.xlsx, not ./)
        ds = load_testcase_dataset(file_path="testcase.xlsx")
        self.assertGreater(len(ds), 0, "Dataset loaded from data/ must not be empty!")
        first_item = ds[0]
        self.assertIn("title", first_item)
        self.assertIn("gt_is_ai", first_item)

    def test_benchmark_audit_excel_contains_dual_columns(self):
        """Benchmark audit export must contain 'ai_model_verdict' and 'consensus_verdict'."""
        from benchmark_evaluator import export_audit_excel

        sample_metrics = {
            "total_samples": 1,
            "evaluated_successfully": 1,
            "errors": 0,
            "TP": 1,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "specificity": 1.0,
            "f1_score": 1.0,
        }
        sample_results = [
            {
                "media_id": "M1",
                "song_title": "Test Track",
                "artist": "Test Artist",
                "query": "Test Track",
                "ground_truth": "AI",
                "ai_model_verdict": "AI_GENERATED",
                "consensus_verdict": "AI_GENERATED",
                "prediction": "AI",
                "is_correct": "ĐÚNG",
            }
        ]

        out_path = export_audit_excel(sample_metrics, sample_results)
        self.assertTrue(Path(out_path).exists())

        wb = openpyxl.load_workbook(out_path)
        sheet = wb["Audit Details"]
        headers = [cell.value for cell in sheet[1]]

        self.assertIn("ai_model_verdict", headers)
        self.assertIn("consensus_verdict", headers)
        Path(out_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()


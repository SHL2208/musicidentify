"""Bộ kiểm thử tính toàn vẹn nguồn tải âm thanh (Audio Source Integrity) và chính sách Broad AI (Hybrid AI = AI Yes).

Kiểm thử:
1. Cơ chế phòng vệ Zing MP3 VIP: Khi bị khóa VIP và không có bản thu chính chủ trên YouTube,
   hệ thống PHẢI báo lỗi AudioProcessingException (tuyệt đối không tải bừa bản cover người thật).
2. Khi YouTube có bản thu chính chủ khớp chính xác nghệ sĩ (score >= 0.65), cho phép fallback an toàn.
3. Bộ phân giải Resolver Service: Từ chối các ứng viên unverified điểm thấp (< 0.65) và raise TrackNotFoundException.
4. Chất lượng âm thanh Gemini: Lệnh nén âm thanh phải sử dụng 160kbps Stereo (-b:a 160k -ac 2).
5. Chính sách Broad AI: Phân loại 'HYBRID_AI' bắt buộc phải gán is_ai = True.

Tuân thủ nghiêm ngặt chuẩn PEP 8 và Ruff.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.core.exceptions import AudioProcessingException, TrackNotFoundException
from backend.core.models.entities import InputType, TrackItem, VerdictType
from backend.services.audio_engine_service import AudioEngineService
from backend.services.gemini_classifier_service import GeminiClassifierService
from backend.services.resolver_service import InputResolverService


class TestSourceIntegrityAndBroadAIPolicy(unittest.TestCase):
    """Kiểm thử tính toàn vẹn nguồn âm thanh và chính sách Broad AI."""

    def setUp(self) -> None:
        self.mock_storage = MagicMock()
        self.mock_storage.get_temp_dir.return_value = Path("d:/test/musicidentify/data/temp")
        self.audio_engine = AudioEngineService(storage_repo=self.mock_storage, ffmpeg_path="ffmpeg")
        self.resolver = InputResolverService()
        self.classifier = GeminiClassifierService(api_keys=["test_key_fake_123"])

    @patch("backend.core.container.get_container")
    @patch.object(AudioEngineService, "_download_audio")
    def test_zingmp3_vip_without_verified_youtube_raises_audio_processing_exception(
        self,
        mock_download: MagicMock,
        mock_get_container: MagicMock,
    ) -> None:
        """Nếu Zing MP3 bị khóa VIP và YouTube không có bản thu khớp chính chủ, PHẢI báo lỗi."""
        # Giả lập tải Zing MP3 ném lỗi tài khoản VIP
        mock_download.side_effect = Exception("The song is only for VIP accounts.")

        # Giả lập YouTube search không tìm thấy bản thu chính chủ khớp nghệ sĩ
        mock_resolver = MagicMock()
        mock_resolver.search_youtube.return_value = (
            "https://www.youtube.com/watch?v=fake_cover",
            "Thần Thoại (Cover)",
            240.0,
            False,
            0.30,  # Điểm khớp thấp, không khớp nghệ sĩ Trương Đông Triết
        )
        mock_container = MagicMock()
        mock_container.resolver = mock_resolver
        mock_get_container.return_value = mock_container

        track = TrackItem(
            id="t_vip_01",
            raw_query="Thần Thoại Trương Đông Triết",
            input_type=InputType.URL,
            resolved_url="https://zingmp3.vn/bai-hat/eQhESPSNB7ZJ.html",
            track_title="Thần Thoại",
            artist="Trương Đông Triết",
            platform="zingmp3",
        )

        with self.assertRaises(AudioProcessingException) as ctx:
            self.audio_engine.process_track_audio(track)

        self.assertIn("bị khóa VIP", str(ctx.exception))
        self.assertIn("không tìm thấy bản phát hành chính chủ trên YouTube", str(ctx.exception))

    @patch("backend.services.audio_engine_service.AudioEngineService._slice_multisegment_sample")
    @patch("backend.core.container.get_container")
    @patch.object(AudioEngineService, "_download_audio")
    def test_zingmp3_vip_with_verified_youtube_succeeds(
        self,
        mock_download: MagicMock,
        mock_get_container: MagicMock,
        mock_slice: MagicMock,
    ) -> None:
        """Nếu Zing MP3 bị khóa VIP nhưng YouTube có bản thu chính chủ khớp nghệ sĩ (score >= 0.65), fallback thành công."""
        # Lần 1 (Zing) lỗi VIP, lần 2 (YouTube fallback) tải thành công
        fake_audio_path = Path("d:/test/musicidentify/data/temp/t_vip_02_raw.m4a")
        fake_sample_path = Path("d:/test/musicidentify/data/temp/t_vip_02_sample.wav")
        mock_download.side_effect = [
            Exception("The song is only for VIP accounts."),
            (fake_audio_path, {"title": "Thần Thoại - Trương Đông Triết Official"}),
        ]
        mock_slice.return_value = fake_sample_path

        mock_resolver = MagicMock()
        mock_resolver.search_youtube.return_value = (
            "https://www.youtube.com/watch?v=truong_dong_triet_official",
            "Thần Thoại - Trương Đông Triết",
            240.0,
            True,  # Khớp chính xác
            0.85,  # Điểm khớp cao >= 0.65
        )
        mock_container = MagicMock()
        mock_container.resolver = mock_resolver
        mock_get_container.return_value = mock_container

        track = TrackItem(
            id="t_vip_02",
            raw_query="Thần Thoại Trương Đông Triết",
            input_type=InputType.URL,
            resolved_url="https://zingmp3.vn/bai-hat/eQhESPSNB7ZJ.html",
            track_title="Thần Thoại",
            artist="Trương Đông Triết",
            platform="zingmp3",
        )

        sample_path, metadata = self.audio_engine.process_track_audio(track)
        self.assertEqual(sample_path, fake_sample_path)
        self.assertTrue(metadata.get("is_fallback_audio"))

    @patch("backend.services.resolver_service.InputResolverService.search_spotify")
    @patch("backend.services.resolver_service.InputResolverService._fetch_zing_metadata")
    @patch("backend.services.resolver_service.InputResolverService.search_youtube")
    def test_resolver_rejects_unverified_low_score_candidates(
        self,
        mock_search_yt: MagicMock,
        mock_fetch_zing: MagicMock,
        mock_search_spotify: MagicMock,
    ) -> None:
        """Bộ phân giải PHẢI từ chối ứng viên YouTube điểm thấp (< 0.65) và raise TrackNotFoundException."""
        # Giả lập YouTube chỉ tìm thấy video cover người khác với score 0.40
        mock_search_yt.return_value = (
            "https://www.youtube.com/watch?v=other_cover",
            "Người Hứa Trước Lại Là Người Quên (Cover)",
            250.0,
            False,
            0.40,
        )
        mock_fetch_zing.return_value = {}  # Zing không có bài
        mock_search_spotify.return_value = {}  # Spotify không có bài

        query = "NGƯỜI HỨA TRƯỚC LẠI LÀ NGƯỜI QUÊN HanZen"

        with self.assertRaises(TrackNotFoundException) as ctx:
            self.resolver.resolve_track(query, artist="HanZen")

        self.assertIn("Không tìm thấy bản thu chính chủ", str(ctx.exception))

    @patch("subprocess.run")
    @patch.object(GeminiClassifierService, "_get_client")
    def test_gemini_compression_uses_160k_stereo(
        self,
        mock_get_client: MagicMock,
        mock_subprocess: MagicMock,
    ) -> None:
        """Kiểm tra lệnh nén ffmpeg phải dùng 160kbps Stereo (-b:a 160k -ac 2)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"verdict_type": "PROCESSED_HUMAN", "is_ai": false, "confidence_score": 0.9, "suspected_engine": "None", "reasons": ["Human vocals"], "detailed_explanation": "Test"}'
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        mock_subprocess.return_value.return_code = 0

        # Giả lập file wav tồn tại
        fake_wav = Path("d:/test/musicidentify/data/temp/test_sample.wav")
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_bytes", return_value=b"fake_audio_bytes"):
                self.classifier.classify_track(
                    track_id="t_comp",
                    track_title="Test Compression",
                    audio_path=fake_wav,
                )

        # Kiểm tra lệnh ffmpeg được gọi
        called = False
        for call_args in mock_subprocess.call_args_list:
            cmd = call_args[0][0]
            if "-b:a" in cmd and "160k" in cmd and "-ac" in cmd and "2" in cmd:
                called = True
                break

        self.assertTrue(called, "Lệnh ffmpeg phải chứa cờ '-b:a 160k' và '-ac 2' (Stereo)")

    @patch.object(GeminiClassifierService, "_get_client")
    def test_hybrid_ai_policy_enforces_is_ai_true(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Kiểm tra chính sách Broad AI: Khi verdict_type là HYBRID_AI thì is_ai bắt buộc là True."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '{"verdict_type": "HYBRID_AI", "is_ai": true, "confidence_score": 0.95, '
            '"suspected_engine": "Suno", "reasons": ["AI beat with human vocals"], '
            '"detailed_explanation": "Vocal is human but beat is neural diffusion"}'
        )
        mock_client.models.generate_content.return_value = mock_response
        mock_get_client.return_value = mock_client

        res = self.classifier.classify_track(
            track_id="t_hybrid",
            track_title="Goat Xuất Trận 2Can",
        )

        self.assertTrue(res.is_ai, "Chính sách Broad AI: HYBRID_AI bắt buộc phải có is_ai = True")
        self.assertEqual(res.verdict_type, VerdictType.HYBRID_AI)
        self.assertIn("HYBRID", res.verdict_label)

    @patch("backend.services.audio_engine_service.AudioEngineService._slice_multisegment_sample")
    @patch("backend.core.container.get_container")
    @patch.object(AudioEngineService, "_download_audio")
    def test_audio_engine_falls_back_to_youtube_on_limited_mp3_download_failure(
        self,
        mock_download: MagicMock,
        mock_get_container: MagicMock,
        mock_slice: MagicMock,
    ) -> None:
        """Kiểm tra: Khi URL MP3 bị giới hạn/lỗi (403/429/stream error), AudioEngine tự động tìm và tải từ YouTube."""
        fake_audio_path = Path("d:/test/musicidentify/data/temp/t_fallback_raw.m4a")
        fake_sample_path = Path("d:/test/musicidentify/data/temp/t_fallback_sample.wav")
        # Lần 1 (MP3 stream) lỗi 403 Forbidden, lần 2 (YouTube fallback) tải thành công
        mock_download.side_effect = [
            Exception("HTTP Error 403: Forbidden - Rate limit / IP restricted"),
            (
                fake_audio_path,
                {
                    "title": "Hoa Nở Không Màu - Hoài Lâm (Official MV)",
                    "webpage_url": "https://www.youtube.com/watch?v=fallback_yt_123",
                },
            ),
        ]
        mock_slice.return_value = fake_sample_path

        mock_resolver = MagicMock()
        mock_resolver.search_youtube.return_value = (
            "https://www.youtube.com/watch?v=fallback_yt_123",
            "Hoa Nở Không Màu - Hoài Lâm",
            250.0,
            True,
            0.80,
        )
        mock_container = MagicMock()
        mock_container.resolver = mock_resolver
        mock_get_container.return_value = mock_container

        track = TrackItem(
            id="t_mp3_limit",
            raw_query="Hoa Nở Không Màu Hoài Lâm",
            input_type=InputType.SONG_NAME,
            resolved_url="https://mp3.zing.vn/bai-hat/hoa-no-khong-mau.html",
            track_title="Hoa Nở Không Màu",
            artist="Hoài Lâm",
            platform="zingmp3",
        )

        sample_path, metadata = self.audio_engine.process_track_audio(track)
        self.assertEqual(sample_path, fake_sample_path)
        self.assertTrue(metadata.get("is_fallback_audio"))
        self.assertEqual(metadata.get("fallback_platform"), "youtube")
        self.assertEqual(track.resolved_url, "https://www.youtube.com/watch?v=fallback_yt_123")


if __name__ == "__main__":
    unittest.main()

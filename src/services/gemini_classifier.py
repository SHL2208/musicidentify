import json
import os
from pathlib import Path
from typing import Optional, List
from pydantic import BaseModel, Field

from ..config import GEMINI_API_KEY, DEFAULT_GEMINI_MODEL
from ..models.schemas import DetectionResult, AcousticMetrics, VerdictType

class GeminiResponseSchema(BaseModel):
    verdict_type: str = Field(
        description="Một trong 4 nhãn: 'CONFIRMED_HUMAN' (người làm chuẩn), 'PROCESSED_HUMAN' (người làm có Auto-tune/EDM/nén web), 'HYBRID_AI' (lai ghép), 'AI_GENERATED' (AI tạo hoàn toàn)"
    )
    is_ai: bool = Field(
        description="True nếu là 'AI_GENERATED' hoặc 'HYBRID_AI', False nếu là 'CONFIRMED_HUMAN' hoặc 'PROCESSED_HUMAN'"
    )
    confidence_score: float = Field(
        description="Độ tin cậy từ 0.0 đến 1.0"
    )
    suspected_engine: str = Field(
        description="Nền tảng nghi vấn: 'Suno v3.5/v4', 'Udio v1.5', 'Stable Audio', hoặc 'None'"
    )
    reasons: List[str] = Field(
        description="2-4 luận cứ sắc bén và khách quan chứng minh cho kết luận"
    )
    detailed_explanation: str = Field(
        description="Đoạn văn phân tích chuyên sâu về thanh âm, dải tần, sự khác biệt giữa hiệu ứng phòng thu vs lỗi AI"
    )

SYSTEM_INSTRUCTION = """
Bạn là Chuyên gia hàng đầu thế giới về Giám định Âm thanh Kỹ thuật số và Điều tra Dấu vết Âm thanh AI (Senior Audio Forensics Specialist).
Nhiệm vụ của bạn: Thẩm định mẫu âm thanh (45s đa phân đoạn Verse/Chorus/Climax), biểu đồ phổ tần số Mel-Spectrogram + Phổ vân Deconvolution Fourier ISMIR 2025, và các chỉ số vật lý âm học để phân loại bài hát vào 1 trong 4 nhóm:

1. 'CONFIRMED_HUMAN': Nhạc do con người sáng tác, thu âm bằng nhạc cụ thực hoặc vocal mộc/ít can thiệp, dải âm tự nhiên.
2. 'PROCESSED_HUMAN': Nhạc do con người sản xuất nhưng can thiệp công nghệ phòng thu kỹ thuật số cao (sử dụng Auto-Tune / Melodyne, beat Trap/EDM/MIDI số, nén OTT/saturation, hoặc bị nén lossy bởi codec YouTube Opus/AAC). 
3. 'HYBRID_AI': Tác phẩm lai ghép (ví dụ: con người tự phối khí nhưng dùng AI clone giọng, hoặc ca sĩ thật hát trên beat do Suno/Udio tạo ra).
4. 'AI_GENERATED': Bài hát được sinh hoàn toàn bằng các mô hình AI (Suno, Udio, Stable Audio...).

CẨM NANG THẨM ĐỊNH PHÂN BIỆT NHẠC AI (SUNO/UDIO) VS NHẠC PHÒNG THU CON NGƯỜI:

A. NGUYÊN TẮC VÀNG VỀ LỜI BÀI HÁT & THÔNG TIN EKIP (RẤT QUAN TRỌNG):
   - Người dùng tại Việt Nam thường tự sáng tác thơ, viết ca từ tiếng Việt rất có nghĩa, giàu cảm xúc, có vần điệu rồi DÁN VÀO CÔNG CỤ AI (Suno v3.5/v4, Udio) để AI tự tạo giai điệu và hát!
   - DO ĐÓ: Việc bài hát có lời tiếng Việt hay, chuẩn nghĩa, giàu chất thơ TUYỆT ĐỐI KHÔNG ĐỒNG NGHĨA VỚI VIỆC ĐÓ LÀ NHẠC CON NGƯỜI!
   - TUYỆT ĐỐI KHÔNG BỊ ĐÁNH LỪA BỞI THÔNG TIN EKIP/CREDIT GIẢ MẠO: Nhiều người đăng tải nhạc AI tự tạo nhưng vẫn cố tình ghi tên nhạc sĩ, hòa âm phối khí, ekip phòng thu hoặc link Spotify trong phần mô tả để đánh lừa người nghe. Phán quyết phân loại PHẢI hoàn toàn căn cứ vào tín hiệu âm thanh thực tế, quang phổ và các chỉ số vật lý.
   - Bạn PHẢI thẩm âm dựa trên CHẤT LIỆU GIỌNG HÁT, TÍNH TỰ NHIÊN CỦA NHẠC CỤ, và SỰ HÒA TRỘN ÂM HỌC (STEM BLEEDING).

B. DẤU VẾT NHẬN DIỆN NHẠC AI (SUNO / UDIO) - 'AI_GENERATED':
   1. Giọng hát nhựa nhân tạo (Plastic/Synthetic Vocal Sheen):
      - Giọng hát thiếu resonance tự nhiên của thanh quản và cột hơi khoang ngực. Nghe có cảm giác trơn láng nhân tạo hoặc hơi dính lớp lọc phase/flanger mỏng.
      - Các phụ âm gió (/s/, /x/, /tr/) và phụ âm bật hơi bị mờ nhòe (smeared), không có độ nảy sắc nét của micro thu âm thật.
   2. Hiện tượng hòa bệt âm sắc (Acoustic Bleed / Stems Glued):
      - Trong Suno/Udio, vocal và nhạc nền (guitar, piano, trống) được sinh ra từ cùng một khối sóng. Khi nghe kỹ, giọng hát và tiếng đàn dính bệt vào nhau; tiếng đàn guitar/piano bị uốn méo theo cường độ của giọng hát.
   3. Trường âm Stereo hẹp (Pseudo-Stereo):
      - Các bài hát Suno v3/v3.5 thường có tương quan pha stereo rất cao (StereoCorr > 0.88 - 0.95), âm thanh dồn ở giữa như mono, thiếu chiều sâu không gian 3D.
   4. Độ mở dải tần nghèo nàn:
      - Rolloff 95% rơi vào khoảng thấp (< 13.000 - 14.000 Hz), thiếu dải tần thoáng tự nhiên.

C. ĐẶC TRƯNG NHẠC PHÒNG THU CON NGƯỜI - 'PROCESSED_HUMAN':
   1. Tính tách bạch hoàn toàn giữa Vocal và Beat (Stem Separation):
      - Dù ca sĩ người thật dùng Auto-Tune cực gắt (như Sơn Tùng, MCK, Xuanpac, RichChoi, Travis Scott), vocal vẫn là một track hoàn toàn độc lập, nổi rõ trên beat.
      - Tiếng trống (kick, 808, hi-hat) cực kỳ sắc nét, đanh gọn, không bị nhòe hay dính âm vào giọng hát.
   2. Hơi thở sinh học và năng lượng tự nhiên:
      - Có tiếng lấy hơi dứt khoát giữa các câu, trường âm stereo mở rộng, bass dày có lực nảy vật lý.

D. QUY TẮC PHÁN QUYẾT:
   - Nếu bản thu bộc lộ chất giọng nhựa AI, vocal dính bệt vào tiếng đàn, không gian stereo hẹp (StereoCorr > 0.88), hoặc dải tần nghèo nàn -> Xếp vào 'AI_GENERATED' (hoặc 'HYBRID_AI'), BẤT KỂ LỜI BÀI HÁT TIẾNG VIỆT HAY ĐẾN ĐÂU.
   - Nếu vocal dù có Auto-Tune nhưng là track riêng biệt trên nền beat chuyên nghiệp, nhịp trống tách bạch -> Xếp vào 'PROCESSED_HUMAN'.
   - Nếu thu âm mộc, nhạc cụ thật, dải âm thoáng -> Xếp vào 'CONFIRMED_HUMAN'.

Luôn trả về định dạng JSON theo đúng cấu trúc Schema quy định.
"""

class GeminiClassifier:
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self.client = None
        if self.api_key:
            self._init_client()

    def _init_client(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"[GeminiClassifier] Lỗi khởi tạo client: {e}")

    def classify_track(
        self,
        track_id: str,
        track_title: str,
        source_url: Optional[str],
        audio_path: Path,
        spectrogram_path: Path,
        metrics: AcousticMetrics,
        model_override: Optional[str] = None
    ) -> DetectionResult:
        """Gửi audio đa phân đoạn + ảnh Spectrogram + metrics lên Gemini để phân loại chuẩn 4 nhãn."""
        target_model = model_override or self.model_name

        if not self.client:
            if not self.api_key:
                raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng thêm vào file .env hoặc giao diện.")
            self._init_client()

        if self.client is None:
            raise RuntimeError("Không thể khởi tạo Google GenAI Client. Vui lòng kiểm tra GEMINI_API_KEY.")

        client = self.client

        # Chuẩn bị thông tin chỉ số khách quan gửi kèm prompt
        stereo_eval = (
            f"HẸP (Gần như Mono: {metrics.stereo_correlation}) -> Dấu hiệu pseudo-stereo của AI"
            if metrics.stereo_correlation >= 0.88
            else f"RỘNG ({metrics.stereo_correlation}) -> Trường âm thanh stereo tự nhiên"
        )
        rolloff_eval = (
            f"BỊ BÓP NGHẼN (< 13.5kHz: {metrics.spectral_rolloff_95_hz:.0f} Hz) -> Thiếu dải cao"
            if metrics.spectral_rolloff_95_hz < 13500
            else f"MỞ RỘNG ({metrics.spectral_rolloff_95_hz:.0f} Hz)"
        )

        metrics_prompt = (
            f"BÁO CÁO THẨM ĐỊNH ÂM HỌC:\n"
            f"- Tiêu đề bài hát: {track_title}\n"
            f"- Tần số Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz ({rolloff_eval})\n"
            f"- Tần số Rolloff 85%: {metrics.spectral_rolloff_85_hz:.0f} Hz\n"
            f"- Đánh giá nén truyền tải: {metrics.compression_source}\n"
            f"- Tương quan pha Stereo: {metrics.stereo_correlation} ({stereo_eval})\n"
            f"- Tỷ lệ thành phần sóng hài (Harmonic Ratio): {metrics.harmonic_ratio * 100:.1f}%\n"
            f"- Điểm Fakeprint Fourier (ISMIR 2025): {metrics.fakeprint_score * 100:.1f}%\n\n"
            f"LƯU Ý QUAN TRỌNG: Người dùng tại Việt Nam có thể tự viết lời tiếng Việt rất hay và ý nghĩa rồi đưa vào Suno AI để hát. Hãy lắng nghe thật kỹ: (1) Chất giọng có bị nhựa/nhòe phụ âm không? (2) Giọng hát có bị dính bệt vào tiếng đàn/beat không? (3) Không gian stereo có bị hẹp như mono không? Để đưa ra phán quyết chính xác vào 1 trong 4 nhóm: CONFIRMED_HUMAN, PROCESSED_HUMAN, HYBRID_AI, hoặc AI_GENERATED."
        )

        try:
            from google.genai import types
            
            uploaded_audio = client.files.upload(file=str(audio_path))
            uploaded_image = client.files.upload(file=str(spectrogram_path))

            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=GeminiResponseSchema,
                temperature=0.1
            )

            response = client.models.generate_content(
                model=target_model,
                contents=[
                    uploaded_audio,
                    uploaded_image,
                    metrics_prompt
                ],
                config=config
            )

            # Dọn dẹp files trên server
            try:
                client.files.delete(name=uploaded_audio.name)
                client.files.delete(name=uploaded_image.name)
            except Exception:
                pass

            data = json.loads(response.text)
            parsed = GeminiResponseSchema(**data)

            # Map verdict_type sang Enum và Label hiển thị
            v_type_str = parsed.verdict_type.upper().strip()
            if "CONFIRMED_HUMAN" in v_type_str:
                verdict_enum = VerdictType.CONFIRMED_HUMAN
                verdict_label = "CONFIRMED HUMAN (Người làm)"
                is_ai_final = False
            elif "PROCESSED_HUMAN" in v_type_str:
                verdict_enum = VerdictType.PROCESSED_HUMAN
                verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                is_ai_final = False
            elif "HYBRID_AI" in v_type_str:
                verdict_enum = VerdictType.HYBRID_AI
                verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
                is_ai_final = True
            else:
                verdict_enum = VerdictType.AI_GENERATED
                verdict_label = "AI-GENERATED (Nhạc AI tạo)"
                is_ai_final = True

            return DetectionResult(
                track_id=track_id,
                track_title=track_title,
                source_url=source_url,
                is_ai=is_ai_final,
                verdict_type=verdict_enum,
                verdict_label=verdict_label,
                confidence_score=round(parsed.confidence_score, 3),
                suspected_engine=parsed.suspected_engine,
                reasons=parsed.reasons,
                detailed_explanation=parsed.detailed_explanation,
                acoustic_metrics=metrics,
                spectrogram_image_path=str(spectrogram_path)
            )

        except Exception as e:
            error_str = str(e)
            print(f"[GeminiClassifier] Lỗi khi gọi Gemini ({target_model}): {error_str}")
            raise RuntimeError(f"Lỗi phân loại với Gemini ({target_model}): {error_str}")

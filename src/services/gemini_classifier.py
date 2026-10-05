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
Nhiệm vụ của bạn: Thẩm định đoạn trích âm thanh (30s), biểu đồ phổ tần số Mel-Spectrogram + Phổ vân Deconvolution Fourier ISMIR 2025, và các chỉ số vật lý âm học để phân loại bài hát vào 1 trong 4 nhóm:

1. 'CONFIRMED_HUMAN': Nhạc do con người sáng tác, thu âm bằng nhạc cụ thực hoặc vocal mộc/ít can thiệp, dải âm tự nhiên.
2. 'PROCESSED_HUMAN': Nhạc do con người sản xuất nhưng can thiệp công nghệ phòng thu cao (sử dụng Auto-Tune / Melodyne, beat Trap/EDM/MIDI số, nén OTT, hoặc bị nén lossy bởi codec YouTube Opus/AAC). 
3. 'HYBRID_AI': Tác phẩm lai ghép (ví dụ: con người tự phối khí nhưng dùng AI clone giọng, hoặc ca sĩ thật hát trên beat do Suno/Udio tạo ra).
4. 'AI_GENERATED': Bài hát được sinh hoàn toàn bằng các mô hình AI (Suno, Udio, Stable Audio...).

QUY TẮC PHÁP LÝ & KỸ THUẬT QUAN TRỌNG (TRÁNH PHÁN SAI CHO NGHỆ SĨ THẬT):
1. NGUYÊN TẮC SUY ĐOÁN CON NGƯỜI (Presumption of Human):
   - Mặc định mọi tác phẩm là do CON NGƯỜI làm ra, trừ khi có bằng chứng vật lý Deconvolution không thể chối cãi.
   - Nhạc Pop, Hip-hop, Trap, EDM hiện đại thường dùng Auto-Tune rất gắt (như Travis Scott, Sơn Tùng, Black Eyed Peas) và beat synth kỹ thuật số. ĐÂY LÀ CHỦ ĐÍCH NGHỆ THUẬT, KHÔNG PHẢI LỖI CỦA AI. Phải xếp vào 'PROCESSED_HUMAN', TUYỆT ĐỐI KHÔNG quy kết là 'AI_GENERATED'!
2. VỀ DẢI TẦN SỐ VÀ NÉN YOUTUBE:
   - File âm thanh tải từ YouTube thường bị codec Opus/AAC nén và cắt bớt dải tần >16-18kHz. Đây là hiện tượng bình thường của việc truyền tải web, KHÔNG PHẢI là dấu vết của AI.
3. VỀ CHỈ SỐ FAKEPRINT (DEEZER ISMIR 2025):
   - Nếu Điểm Fakeprint < 50% và không phát hiện vân răng lược deconvolution: Gần như chắc chắn bài hát KHÔNG dùng neural vocoder của Suno/Udio.
   - Nếu Điểm Fakeprint >= 60% kết hợp với giọng hát có lỗi lượng tử hóa (quantum artifacts, phô chênh vi mô phi tự nhiên): Mới đủ căn cứ nghi vấn 'AI_GENERATED'.

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
        """Gửi audio 30s + ảnh Spectrogram + metrics lên Gemini để phân loại chuẩn 4 nhãn."""
        target_model = model_override or self.model_name

        if not self.client:
            if not self.api_key:
                raise ValueError("Chưa cấu hình GEMINI_API_KEY. Vui lòng thêm vào file .env hoặc giao diện.")
            self._init_client()

        if self.client is None:
            raise RuntimeError("Không thể khởi tạo Google GenAI Client. Vui lòng kiểm tra GEMINI_API_KEY.")

        client = self.client

        # Chuẩn bị thông tin chỉ số khách quan gửi kèm prompt
        metrics_prompt = (
            f"BÁO CÁO THẨM ĐỊNH ÂM HỌC:\n"
            f"- Tiêu đề bài hát: {track_title}\n"
            f"- Tần số Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz (Độ mở dải tần)\n"
            f"- Tần số Rolloff 85%: {metrics.spectral_rolloff_85_hz:.0f} Hz\n"
            f"- Đánh giá nén truyền tải: {metrics.compression_source}\n"
            f"- Điểm Fakeprint Deconvolution Fourier (ISMIR 2025): {metrics.fakeprint_score * 100:.1f}%\n"
            f"- Kết luận Fakeprint: {'PHÁT HIỆN VÂN DECONVOLUTION CỦA NEURAL VOCODER' if metrics.fakeprint_detected else 'KHÔNG PHÁT HIỆN DẤU VẾT AI (CHUẨN TỰ NHIÊN)'}\n"
            f"- Tương quan pha Stereo: {metrics.stereo_correlation}\n"
            f"- Tỷ lệ thành phần sóng hài (Harmonic Ratio): {metrics.harmonic_ratio * 100:.1f}%\n\n"
            f"Yêu cầu: Lắng nghe đoạn audio, đối chiếu ảnh phổ Mel-Spectrogram + Phổ vân Deconvolution ở panel dưới, kết hợp các chỉ số trên để đưa ra phán quyết khách quan vào 1 trong 4 nhóm: CONFIRMED_HUMAN, PROCESSED_HUMAN, HYBRID_AI, hoặc AI_GENERATED."
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

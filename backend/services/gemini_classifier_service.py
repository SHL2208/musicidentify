import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.config import GEMINI_API_KEY, GEMINI_API_KEYS, DEFAULT_GEMINI_MODEL
from ..core.models.entities import DetectionResult, AcousticMetrics, VerdictType
from ..core.interfaces.services.classifier_interface import IAIClassifier
from ..core.exceptions import ClassificationException

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

1. 'AI_GENERATED': Bài hát được sinh ra bằng các mô hình AI (Suno v3/v3.5/v4, Udio v1/v1.5, Stable Audio...). Toàn bộ vocal và beat nhạc nền được sinh trực tiếp từ AI.
2. 'HYBRID_AI': Tác phẩm lai ghép (ví dụ: con người tự hòa âm phối khí chuyên nghiệp trong DAW nhưng dùng AI clone giọng, hoặc ca sĩ thật hát trên beat do Suno/Udio tạo ra).
3. 'PROCESSED_HUMAN': Nhạc do con người sản xuất trong phòng thu chuyên nghiệp, ca sĩ người thật bằng xương bằng thịt hát qua micro thu âm thật, nhưng can thiệp công nghệ phòng thu kỹ thuật số (sử dụng Auto-Tune / Melodyne, beat Trap/EDM/MIDI số, nén OTT/saturation, hoặc bị nén lossy bởi codec YouTube Opus/AAC).
4. 'CONFIRMED_HUMAN': Nhạc do con người sáng tác và thu âm mộc bằng nhạc cụ thực hoặc vocal mộc/ít can thiệp, dải âm thoáng tự nhiên.

CẨM NANG THẨM ĐỊNH PHÂN BIỆT NHẠC AI (SUNO/UDIO) VS NHẠC CON NGƯỜI:

A. NGUYÊN TẮC VÀNG VỀ LỜI BÀI HÁT & THÔNG TIN EKIP (RẤT QUAN TRỌNG):
   - Người dùng tại Việt Nam thường tự viết lời thơ tiếng Việt rất hay, giàu cảm xúc, có vần điệu rồi DÁN VÀO CÔNG CỤ AI (Suno v3.5/v4, Udio) để AI tự tạo giai điệu và hát!
   - DO ĐÓ: Việc bài hát có lời tiếng Việt hay, chuẩn nghĩa, giàu chất thơ TUYỆT ĐỐI KHÔNG ĐỒNG NGHĨA VỚI VIỆC ĐÓ LÀ NHẠC CON NGƯỜI!
   - TUYỆT ĐỐI KHÔNG BỊ ĐÁNH LỪA BỞI THÔNG TIN EKIP/CREDIT GIẢ MẠO: Người đăng nhạc AI thường cố tình tự đặt tên nhạc sĩ, ca sĩ, ekip phòng thu hoặc link Spotify/Apple Music trong phần mô tả để ngụy tạo. Nhiều kênh tạo dự án AI hàng loạt đánh số như "(Bài số 83/100)". Phán quyết phân loại PHẢI hoàn toàn căn cứ vào tín hiệu âm thanh thực tế, quang phổ và chất liệu giọng hát!
   - BẤT KỂ LỜI HAY ĐẾN ĐÂU, NẾU ÂM THANH DO SUNO/UDIO TẠO THÌ BẮT BUỘC PHẢI XẾP VÀO 'AI_GENERATED'!

B. DẤU VẾT NHẬN DIỆN NHẠC AI (SUNO / UDIO) - 'AI_GENERATED':
   1. Giọng hát nhựa nhân tạo (Plastic / Synthetic Vocal Sheen):
      - Giọng hát thiếu resonance tự nhiên của thanh quản và khoang ngực người. Nghe có cảm giác trơn láng nhân tạo hoặc hơi dính lớp lọc phase/flanger kim loại mỏng.
      - Phát âm tiếng Việt: Các thanh điệu (sắc, huyền, hỏi, ngã, nặng) có hiện tượng trượt tần số cơ học (robotic pitch glide), rung giả đều tăm tắp. Các phụ âm đầu (/s/, /x/, /tr/, /ch/, /p/, /t/) và âm đuôi bị mờ nhòe (smeared), dính bệt vào sóng âm nền thay vì nảy sắc nét của micro thu âm thật.
      - Thiếu hơi thở sinh học tự nhiên giữa các câu hát (human breath intake).
   2. Hiện tượng hòa bệt âm sắc (Acoustic Bleed / Stem Entanglement):
      - Trong Suno/Udio, vocal và nhạc nền (guitar, piano, beat) được sinh ra từ cùng một khối sóng latent diffusion. Khi nghe kỹ, giọng hát và tiếng đàn dính bệt vào nhau; tiếng đàn guitar/piano bị uốn méo theo cường độ của giọng hát.
   3. Bộ gõ và Dải tần số cao (Cymbals/Hi-Hats Fizz):
      - Tiếng hi-hats, cymbals trong Suno phát ra tiếng xì xào dạng nhiễu hạt tổng hợp (white-noise fizz/fuzziness), thiếu độ đanh cơ học của lá kim loại thật trong phòng thu.
   4. Cấu trúc lặp cơ học và độ mở dải tần:
      - Vòng hợp âm lặp cơ học, đoạn outro fade-out cụt đột ngột. Tần số cao thường có vết vân răng lược chu kỳ của tầng Deconvolution.

C. PHÂN BIỆT VỚI NHẠC PHÒNG THU CON NGƯỜI ('PROCESSED_HUMAN'):
   - Dù ca sĩ người thật dùng Auto-Tune cực gắt (như Sơn Tùng, MCK, Suboi, Travis Scott), vocal vẫn là một track hoàn toàn độc lập, nổi rõ trên beat, từng phụ âm tiếng Việt (/t/, /k/, /s/, /ch/) cực kỳ sắc nét.
   - Tiếng trống (kick, 808, hi-hat) cực kỳ sắc nét, đanh gọn, không bị nhòe hay dính âm vào giọng hát.
   - Luôn có hơi thở sinh học và năng lượng âm học tự nhiên của người thật.
   - TUYỆT ĐỐI KHÔNG xếp nhạc Suno/Udio vào 'PROCESSED_HUMAN' hay 'CONFIRMED_HUMAN'!

Luôn trả về định dạng JSON theo đúng cấu trúc Schema quy định.
"""

class GeminiClassifierService(IAIClassifier):
    """
    Hiện thực phân loại giám định âm nhạc AI bằng Gemini API.
    Hỗ trợ luân chuyển đa chìa khóa (Failover 1 > 2 > 3) và tuân thủ IAIClassifier.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model_name: Optional[str] = None
    ):
        if api_keys:
            self.api_keys: List[str] = [k.strip() for k in api_keys if k and k.strip()]
        elif api_key:
            clean_key = api_key.strip()
            self.api_keys = [clean_key] + [k for k in GEMINI_API_KEYS if k != clean_key]
        else:
            self.api_keys = list(GEMINI_API_KEYS)

        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self._clients: dict = {}

    def _get_client(self, key: str):
        if not key:
            return None
        if key in self._clients:
            return self._clients[key]
        try:
            from google import genai
            client = genai.Client(api_key=key)
            self._clients[key] = client
            return client
        except Exception as e:
            print(f"[GeminiClassifierService] Lỗi khởi tạo GenAI Client: {e}")
            return None

    def classify_track(
        self,
        track_id: str,
        track_title: str,
        source_url: Optional[str] = None,
        audio_path: Optional[Path] = None,
        spectrogram_path: Optional[Path] = None,
        metrics: Optional[AcousticMetrics] = None,
        model_override: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        model = model_override or self.model_name

        metrics_text = "Không có thông số âm học."
        if metrics:
            fp_note = "PHÁT HIỆN DẤU VÂN AI" if metrics.fakeprint_detected else "Bình thường"
            metrics_text = f"""
- Tần số lấy mẫu: {metrics.sample_rate} Hz
- Thời lượng mẫu thử (Đa phân đoạn ghép): {metrics.duration_sec:.1f}s
- Độ mở dải tần Spectral Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz
- Spectral Rolloff 85%: {metrics.spectral_rolloff_85_hz:.0f} Hz
- Tương quan pha Stereo L/R: {metrics.stereo_correlation:.2f}
- Tỷ lệ sóng hài Harmonic / Percussive: {metrics.harmonic_ratio:.2f}
- Nguồn nén truyền tải dự kiến: {metrics.compression_source}
- Vết ngắt dải tần số (Cutoff): {'Có' if metrics.has_brickwall_cutoff else 'Không'}
- Điểm vết vân Fourier Deconvolution (ISMIR 2025): {metrics.fakeprint_score * 100:.1f}% ({fp_note})
"""

        metadata_text = "Không có metadata từ nền tảng phát."
        if metadata:
            metadata_text = f"""
- Tiêu đề gốc: {metadata.get('title', 'N/A')}
- Kênh/Uploader: {metadata.get('uploader', metadata.get('channel', 'N/A'))}
- Ngày đăng: {metadata.get('upload_date', 'N/A')}
- Lượt xem: {metadata.get('view_count', 'N/A')}
- Mô tả tóm tắt: {(metadata.get('description') or '')[:500]}
"""

        prompt = f"""
HÃY GIÁM ĐỊNH MẪU ÂM THANH BÀI HÁT SAU ĐÂY:

🎵 Tên bài hát: {track_title}
🔗 Nguồn / URL: {source_url or 'N/A'}

📊 THÔNG TIN BÀI HÁT & METADATA:
{metadata_text}

🔬 THÔNG SỐ VẬT LÝ ÂM HỌC & FORENSICS:
{metrics_text}

HƯỚNG DẪN THẨM ÂM & RA KẾT LUẬN:
1. Lắng nghe kỹ đoạn âm thanh (chú ý chất giọng vocal, độ hòa bệt với tiếng đàn/beat, tiếng xì xào của cymbals/hi-hats).
2. Quan sát kỹ ảnh phổ đính kèm:
   - Tầng 1: Mel-Spectrogram xem dải cao trên 15kHz có bị cut không.
   - Tầng 2: Fourier Fakeprint xem có các đỉnh răng lược chu kỳ cách đều không.
3. Ra quyết định phân loại vào 1 trong 4 nhóm:
   - 'CONFIRMED_HUMAN': Nhạc người làm mộc mạc, dải tần tự nhiên, giọng ấm tự nhiên.
   - 'PROCESSED_HUMAN': Nhạc người làm có Auto-Tune, EDM, nén YouTube, vocal tách rời beat.
   - 'HYBRID_AI': Ca sĩ người hát trên beat Suno, hoặc beat người làm vocal AI clone.
   - 'AI_GENERATED': Nhạc do Suno/Udio tạo ra (vocal nhựa, hòa bệt âm sắc, dính noise cymbals).
"""

        candidate_keys = list(self.api_keys)
        if not candidate_keys and self.api_key:
            candidate_keys = [self.api_key]

        if not candidate_keys:
            raise ClassificationException(
                "Không tìm thấy Gemini API Key nào trong cấu hình môi trường (.env)."
            )

        last_error = None
        for key_idx, current_key in enumerate(candidate_keys, 1):
            client = self._get_client(current_key)
            if not client:
                continue

            try:
                from google.genai import types

                contents: List[Any] = [prompt]

                if spectrogram_path and Path(spectrogram_path).exists():
                    try:
                        spec_file = client.files.upload(file=str(spectrogram_path))
                        contents.append(spec_file)
                    except Exception as upload_err:
                        print(f"[GeminiClassifierService] Không thể upload ảnh phổ: {upload_err}")

                if audio_path and Path(audio_path).exists():
                    try:
                        audio_file = client.files.upload(file=str(audio_path))
                        contents.append(audio_file)
                    except Exception as upload_err:
                        print(f"[GeminiClassifierService] Không thể upload audio sample: {upload_err}")

                config = types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=GeminiResponseSchema,
                    temperature=0.1
                )

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config
                )

                if key_idx > 1:
                    print(f"[GeminiClassifierService] [SUCCESS] Đã tự động chuyển đổi thành công sang Key dự phòng #{key_idx}!")

                raw_json = response.text
                data = json.loads(raw_json)

                v_type_str = data.get("verdict_type", "CONFIRMED_HUMAN")
                try:
                    v_type = VerdictType(v_type_str)
                except ValueError:
                    v_type = VerdictType.AI_GENERATED if data.get("is_ai") else VerdictType.CONFIRMED_HUMAN

                if v_type == VerdictType.CONFIRMED_HUMAN:
                    verdict_label = "CONFIRMED HUMAN (Người làm)"
                elif v_type == VerdictType.PROCESSED_HUMAN:
                    verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                elif v_type == VerdictType.HYBRID_AI:
                    verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
                else:
                    verdict_label = "AI-GENERATED (Nhạc AI tạo)"

                return DetectionResult(
                    track_id=track_id,
                    track_title=track_title,
                    source_url=source_url,
                    is_ai=data.get("is_ai", False),
                    verdict_type=v_type,
                    verdict_label=verdict_label,
                    confidence_score=data.get("confidence_score", 0.5),
                    suspected_engine=data.get("suspected_engine", "None"),
                    reasons=data.get("reasons", []),
                    detailed_explanation=data.get("detailed_explanation", ""),
                    acoustic_metrics=metrics,
                    spectrogram_image_path=str(spectrogram_path) if spectrogram_path else None
                )

            except Exception as e:
                err_str = str(e)
                print(f"[GeminiClassifierService] [WARNING] Key #{key_idx} (***) gặp lỗi khi gọi ({model}): {err_str}")
                last_error = e
                if key_idx < len(candidate_keys):
                    print(f"[GeminiClassifierService] [FAILOVER] Đang tự động chuyển sang Key dự phòng kế tiếp #{key_idx + 1}/{len(candidate_keys)} (thứ tự 1>2>3)...")
                    continue
                else:
                    print(f"[GeminiClassifierService] [ERROR] Đã thử tất cả {len(candidate_keys)} API keys nhưng đều thất bại.")

        raise ClassificationException(
            f"Tất cả {len(candidate_keys)} Gemini API Keys đều thất bại. Lỗi cuối cùng: {last_error}"
        )

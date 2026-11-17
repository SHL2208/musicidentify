import json
import logging
import re
import subprocess
import threading
import time
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*automatic function calling.*")

from pydantic import BaseModel, Field

from ..core.config import (
    ATTACH_SPECTROGRAM_TO_LLM,
    DEFAULT_GEMINI_MODEL,
    ENABLE_CASCADED_EVALUATION,
    FLASH_LITE_CHORUS_DURATION_SEC,
    FLASH_LITE_CHORUS_START_SEC,
    GEMINI_API_KEYS,
    GEMINI_FALLBACK_MODELS,
    get_ffmpeg_binary,
)
from ..core.exceptions import ClassificationException
from ..core.interfaces.services.classifier_interface import IAIClassifier
from ..core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    InputType,
    VerdictType,
    WorkType,
)


def is_derivative_or_remix_context(metadata: dict[str, Any] | None) -> bool:
    """
    Xác định bài hát có thuộc nhóm thể loại phái sinh (Remix, Slowed, Lo-Fi, EDM) hay không.
    LƯU Ý QUAN TRỌNG: Tuyệt đối không quét từ khóa trong trường 'description' vì hầu hết các video
    YouTube đều có điều khoản bản quyền boilerplate ('vui lòng không reup hoặc remix...', 'nếu bạn muốn cover, remix...'),
    dẫn đến nhận diện nhầm các ca khúc Ballad/Pop thông thường thành bản Remix phái sinh.
    """
    if not metadata:
        return False
    song_type = str(metadata.get("song_type") or "").strip().lower()
    title_text = f"{metadata.get('title', '')} {metadata.get('target_title', '')} {metadata.get('raw_query', '')}".strip().lower()

    pattern = r"\b(remix|remake|slowed|speed\s*up|sped\s*up|lofi|lo-fi|edm|club\s*mix|drill|vinahouse|bass\s*boosted|nightcore|liên\s*khúc|lien\s*khuc|mix|remastered|remaster|deep\s*house|latino|latin\s*dance|dance\s*ver(?:sion)?)\b"
    if re.search(pattern, title_text, re.IGNORECASE):
        return True
    if song_type and re.search(pattern, song_type, re.IGNORECASE):
        return True

    # Chỉ kiểm tra tags khi tag là nhãn thể loại đơn lẻ khớp chính xác
    genre_tags = {
        "remix",
        "vinahouse",
        "edm",
        "slowed",
        "speed up",
        "sped up",
        "lofi",
        "lo-fi",
        "club mix",
        "drill",
        "nightcore",
        "liên khúc",
        "lien khuc",
        "mix",
        "remastered",
        "deep house",
        "latino",
        "latin dance",
    }
    tags = [str(t).strip().lower() for t in metadata.get("tags") or [] if t]
    return any(t in genre_tags for t in tags)


def should_require_deep_audio_inspection(
    metrics: AcousticMetrics | None, metadata: dict[str, Any] | None
) -> bool:
    """
    Xác định xem bài hát có cần thẩm âm chuyên sâu bằng Audio sample 45s và Ảnh phổ hay không.
    BẢO VỆ TUYỆT ĐỐI KHÔNG BỎ SÓT AI THẾ HỆ MỚI (Suno v3.5/v4, Udio, AI Ballad/Bolero):
    Bất kỳ bài nào rơi vào một trong các điều kiện nghi vấn sẽ BẮT BUỘC gửi Audio và Ảnh phổ lên Gemini.
    """
    if not metrics:
        return True

    # 1. Cờ nghi vấn giọng hát AI thế hệ mới (Modern AI Vocal Suspect)
    if getattr(metrics, "modern_ai_vocal_suspect", False):
        return True

    # 2. Ngữ cảnh phái sinh (Remix, Cover, Remake, Lo-Fi, Slowed) hoặc từ khóa AI / Voice Clone
    if is_derivative_or_remix_context(metadata):
        return True
    if metadata:
        title_context = f"{metadata.get('title', '')} {metadata.get('target_title', '')} {metadata.get('raw_query', '')}".lower()
        if re.search(
            r"\b(cover|ai|voice\s*clone|suno|udio|deepfake)\b", title_context, re.IGNORECASE
        ):
            return True

    # 3. Chỉ số cao độ nằm trong dải chuyển tiếp / nghi vấn phẳng (Latent Rigidity)
    rigidity = float(getattr(metrics, "pitch_rigidity_ratio", 0.0) or 0.0)
    jitter = float(getattr(metrics, "pitch_jitter", 0.0) or 0.0)
    if 0.28 <= rigidity <= 0.55 or (0.0 < jitter < 0.022):
        return True

    # 4. Chỉ số rò rỉ âm học (Stem Bleed) trên ngưỡng nghi vấn
    stem_bleed = float(getattr(metrics, "stem_bleed_score", 0.0) or 0.0)
    if stem_bleed >= 0.28:
        return True

    # 5. Dải suy giảm tần số cao nằm ở vùng xám (11.5kHz - 17.5kHz)
    rolloff_95 = float(getattr(metrics, "spectral_rolloff_95_hz", 20000.0) or 20000.0)
    if 11500.0 <= rolloff_95 <= 17500.0:
        return True

    # 6. Dấu vân Fakeprint nằm ở vùng ranh giới
    fakeprint = float(getattr(metrics, "fakeprint_score", 0.0) or 0.0)
    if 0.32 <= fakeprint <= 0.65:
        return True

    # 7. Nhạc ra mắt gần đây (>= 2023) có chỉ số mấp mé ranh giới
    if metadata:
        target_date = str(metadata.get("target_release_date") or metadata.get("release_date") or "")
        if target_date and len(target_date) >= 4:
            try:
                if int(target_date[:4]) >= 2023:
                    if rigidity > 0.24 or stem_bleed > 0.22 or rolloff_95 < 18500.0:
                        return True
            except ValueError:
                pass

    # Nếu vượt qua tất cả các bộ lọc nghi vấn trên: Bài hát có số liệu vật lý cực kỳ rõ ràng
    return False


class GeminiResponseSchemaVi(BaseModel):
    vocal_ai_score: int = Field(
        default=1,
        description="Thang điểm nhân tạo của giọng hát (1-5): 1=Giọng người mộc 100%, 2=Giọng người có Auto-Tune/Melodyne phòng thu thương mại, 3=Vùng xám/nghi vấn, 4=Khả năng cao là AI vocal clone (RVC/Diff-SVC) hoặc Suno/Udio synthetic voice, 5=Chắc chắn là giọng tạo sinh AI.",
    )
    beat_ai_score: int = Field(
        default=1,
        description="Thang điểm nhân tạo của beat/hòa âm (1-5): 1=Nhạc cụ mộc hoặc DAW multitrack người làm, 2=Beat EDM/Trap/MIDI DAW thương mại chuẩn phòng thu, 3=Vùng xám bết dính nhẹ, 4=Khả năng cao là AI generative backing loop, 5=Chắc chắn là hòa âm tạo sinh nguyên khối từ AI (Suno/Udio single-pass diffusion).",
    )
    verdict_type: str = Field(
        description="One of 5 labels: 'CONFIRMED_HUMAN' (100% human organic acoustic), 'PROCESSED_HUMAN' (100% human artist & 100% human beat with modern studio production/Auto-tune), 'HYBRID_AI' (ANY AI participation: AI beat + human voice, or AI voice + human beat, or AI remix), 'AI_GENERATED' (fully AI generated via Suno/Udio), 'UNCERTAIN' (inconclusive evidence, severe noise/distortion, or conflicting indicators)"
    )
    is_ai: bool = Field(
        description="MANDATORY: True if 'AI_GENERATED' or 'HYBRID_AI' (ANY AI involvement in beat, vocal, or composition). False ONLY if 100% human-created ('CONFIRMED_HUMAN', 'PROCESSED_HUMAN'), or 'UNCERTAIN'."
    )
    confidence_score: float = Field(description="Confidence score ranging from 0.0 to 1.0")
    suspected_engine: str = Field(
        description="Suspected platform/model: 'Suno', 'Udio', 'Stable Audio', or 'None'"
    )
    reasons: list[str] = Field(
        description="2-4 sharp, objective bullet points justifying the verdict written strictly in VIETNAMESE (Tiếng Việt). MANDATORY FOR HYBRID_AI: The FIRST reason MUST strictly begin with: '[HYBRID AI]: Thành phần Vocal là [Người thật/AI] | Thành phần Beat/Hòa âm là [AI sinh tự động/Người làm] - [Giải thích chi tiết dấu vết kỹ thuật]'"
    )
    detailed_explanation: str = Field(
        description="In-depth technical forensic analysis covering vocal timbre, frequency spectrum, and studio effects vs AI artifacts, written strictly in VIETNAMESE (Tiếng Việt)"
    )
    work_type: str = Field(
        default="UNCERTAIN",
        description="Work classification: 'ORIGINAL' (official original release), 'COVER' (performed by someone else or an AI voice clone), 'REMIX' (remixed/re-arranged preserving original vocal stem), or 'UNCERTAIN' (inconclusive)",
    )
    is_cover: bool = Field(
        default=False,
        description="True if this is a cover version performed by another singer or an AI voice clone",
    )
    original_artist: str = Field(
        default="",
        description="Original artist/creator of the song if known, or empty",
    )
    original_song_title: str = Field(
        default="",
        description="Original title of the song if this is a cover/remix, or empty",
    )
    cover_performer: str = Field(
        default="",
        description="The actual performer singing in this recording",
    )


class GeminiResponseSchemaEn(BaseModel):
    vocal_ai_score: int = Field(
        default=1,
        description="Vocal AI artifact score (1-5): 1=100% organic human voice, 2=Human singer with commercial DAW Auto-Tune/Melodyne, 3=Ambiguous/borderline, 4=Probable AI vocal clone (RVC/Diff-SVC) or Suno/Udio synthetic vocal, 5=Definitive AI synthetic vocal.",
    )
    beat_ai_score: int = Field(
        default=1,
        description="Beat/Arrangement AI artifact score (1-5): 1=Acoustic or human DAW multitrack, 2=Commercial digital EDM/Trap loop or human MIDI arrangement, 3=Ambiguous/borderline bleed, 4=Probable generative AI backing track, 5=Definitive end-to-end AI generative arrangement (Suno/Udio single-pass diffusion).",
    )
    verdict_type: str = Field(
        description="One of 5 labels: 'CONFIRMED_HUMAN' (100% human organic acoustic), 'PROCESSED_HUMAN' (100% human artist & 100% human beat with modern studio production/Auto-tune), 'HYBRID_AI' (ANY AI participation: AI beat + human voice, or AI voice + human beat, or AI remix), 'AI_GENERATED' (fully AI generated via Suno/Udio), 'UNCERTAIN' (inconclusive evidence, severe noise/distortion, or conflicting indicators)"
    )
    is_ai: bool = Field(
        description="MANDATORY: True if 'AI_GENERATED' or 'HYBRID_AI' (ANY AI involvement in beat, vocal, or composition). False ONLY if 100% human-created ('CONFIRMED_HUMAN', 'PROCESSED_HUMAN'), or 'UNCERTAIN'."
    )
    confidence_score: float = Field(description="Confidence score ranging from 0.0 to 1.0")
    suspected_engine: str = Field(
        description="Suspected platform/model: 'Suno', 'Udio', 'Stable Audio', or 'None'"
    )
    reasons: list[str] = Field(
        description="2-4 sharp, objective bullet points justifying the verdict written strictly in ENGLISH. MANDATORY FOR HYBRID_AI: The FIRST reason MUST strictly begin with: '[HYBRID AI]: Vocal component is [Human/AI] | Beat/Arrangement component is [AI-generated/Human-made] - [Detailed technical forensic evidence]'"
    )
    detailed_explanation: str = Field(
        description="In-depth technical forensic analysis covering vocal timbre, frequency spectrum, and studio effects vs AI artifacts, written strictly in ENGLISH"
    )
    work_type: str = Field(
        default="UNCERTAIN",
        description="Work classification: 'ORIGINAL' (official original release), 'COVER' (performed by someone else or an AI voice clone), 'REMIX' (remixed/re-arranged preserving original vocal stem), or 'UNCERTAIN' (inconclusive)",
    )
    is_cover: bool = Field(
        default=False,
        description="True if this is a cover version performed by another singer or an AI voice clone",
    )
    original_artist: str = Field(
        default="",
        description="Original artist/creator of the song if known, or empty",
    )
    original_song_title: str = Field(
        default="",
        description="Original title of the song if this is a cover/remix, or empty",
    )
    cover_performer: str = Field(
        default="",
        description="The actual performer singing in this recording",
    )


# Backward-compatible alias
GeminiResponseSchema = GeminiResponseSchemaVi


SYSTEM_INSTRUCTION = """
You are a world-leading Digital Audio Forensics Specialist and Senior Music Producer.
Your mission: Conduct a component-wise forensic audio inspection of the track (multi-segment excerpt: Intro, Verse, Chorus + 3-tier Mel-Spectrogram).
Evaluate the synthetic AI likelihood of BOTH components on a 1 to 5 scale:

1. VOCAL AI ARTIFACT SCORE (`vocal_ai_score` from 1 to 5):
- 1 = Organic biological human vocal (natural micro-vibrato, authentic vocal tract formants, acoustic room decay, natural breath friction).
- 2 = Authentic human vocal with commercial studio pitch quantization (Auto-Tune / Melodyne). Preserves authentic vocal tract anatomy, dynamic phrasing, and biological breathing dynamics.
- 3 = Ambiguous / borderline vocal (heavy processing or lossy streaming artifacts masking biological dynamics).
- 4 = Probable Generative AI vocal (Suno v3.5/v4 or Diff-SVC/RVC clone: synthetic breath mechanics, flat formant trajectory across different registers, phase smearing with beat).
- 5 = Definitive Generative AI synthetic vocal (neural vocoder wateriness, metallic sheen, continuous latent diffusion smearing).

2. BEAT / ARRANGEMENT AI ARTIFACT SCORE (`beat_ai_score` from 1 to 5):
- 1 = Organic acoustic instrumentation or live recording (clean multitrack separation, genuine room acoustic decay).
- 2 = Authentic commercial studio DAW electronic production (EDM/Trap/Pop/Lo-fi beat made by human producers, MIDI synths, sidechain compression against 808s, multitrack DAW stems).
- 3 = Ambiguous loop (phase smearing or heavy lossy codec artifacts masking multitrack boundaries).
- 4 = Probable Generative AI backing track (algorithmic loop from Suno/Udio, timbral bleed between instruments, lacking multitrack independence).
- 5 = Definitive Generative AI backing track (monolithic end-to-end single-pass latent diffusion, intermodulation stem bleed between percussion and synth pads).

FEW-SHOT FORENSIC SCORING BENCHMARKS (REFERENCE EXAMPLES):
- Case 1 (Acoustic Human Pop/Ballad): Real guitar and live acoustic human vocals.
  -> vocal_ai_score: 1, beat_ai_score: 1, verdict_type: "CONFIRMED_HUMAN", is_ai: false.
- Case 2 (Commercial Human Rap/Pop with Heavy Auto-Tune): Auto-Tuned human vocal over commercial FL Studio/Ableton 808 Trap beat.
  -> vocal_ai_score: 2, beat_ai_score: 2, verdict_type: "PROCESSED_HUMAN", is_ai: false.
- Case 3 (Modern Suno v3.5/v4 Pop Ballad or Bolero): Sweet simulated vibrato and breath intakes, but flat formant trajectory across registers; Tier 2 spectrogram reveals vocal harmonics entangled with backing guitars/pads (stem bleed).
  -> vocal_ai_score: 4, beat_ai_score: 5, verdict_type: "AI_GENERATED", is_ai: true.
- Case 4 (AI Voice Clone / AI Cover over Human Beat): AI voice clone (Diff-SVC/RVC) singing over a standard karaoke/EDM DAW backing beat.
  -> vocal_ai_score: 5, beat_ai_score: 2, verdict_type: "HYBRID_AI", is_ai: true.
- Case 5 (Human Vocal on Algorithmic AI Beat): Authentic human singer performing over an algorithmic Suno/Udio Lo-fi beat.
  -> vocal_ai_score: 2, beat_ai_score: 4, verdict_type: "HYBRID_AI", is_ai: true.

DECISION MATRIX:
- If vocal_ai_score >= 4 and beat_ai_score >= 4 -> 'AI_GENERATED' (is_ai: true).
- If (vocal_ai_score >= 4 and beat_ai_score <= 2) or (vocal_ai_score <= 2 and beat_ai_score >= 4) -> 'HYBRID_AI' (is_ai: true).
- If vocal_ai_score <= 2 and beat_ai_score <= 2:
  * If both are 1 -> 'CONFIRMED_HUMAN' (is_ai: false).
  * Else -> 'PROCESSED_HUMAN' (is_ai: false).
- If both are 3 -> 'UNCERTAIN' (is_ai: false).

MANDATORY GUIDELINES:
- Double-Blind Protocol: Track metadata and artist names are masked to prevent anchoring bias. Base your scores strictly on acoustic physics and auditory evidence.
- Distinguish Studio Auto-Tune from AI: Commercial Auto-Tune quantizes pitch but retains human vocal tract formants (vocal_ai_score: 2). Suno v3.5/v4 synthesizes breath intakes and vibrato, but has flat formant sheen and stem bleed (vocal_ai_score: 4 or 5).
- Web Streaming Codec Reality: Audio from YouTube/Spotify uses Opus/AAC compression. High frequencies above 16-18kHz are naturally attenuated. Do NOT classify a beat as AI purely due to lossy MP3/Opus compression.
- Work Authenticity: Evaluate whether recording is 'ORIGINAL', 'COVER', 'REMIX', or 'UNCERTAIN'.
- Return valid JSON strictly adhering to the schema.
"""


class GeminiClassifierService(IAIClassifier):
    """
    Hiện thực phân loại giám định âm nhạc AI bằng Gemini API.
    Hỗ trợ luân chuyển đa chìa khóa (Failover 1 > 2 > 3), Sticky Key và tuân thủ IAIClassifier.
    """

    _active_key_index: int = 0
    _key_cooldowns: dict[Any, float] = {}
    _key_lock = threading.Lock()

    def __init__(
        self,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        model_name: str | None = None,
        fallback_models: list[str] | None = None,
    ):
        if api_keys:
            self.api_keys: list[str] = [k.strip() for k in api_keys if k and k.strip()]
        elif api_key:
            clean_key = api_key.strip()
            self.api_keys = [clean_key] + [k for k in GEMINI_API_KEYS if k != clean_key]
        else:
            self.api_keys = list(GEMINI_API_KEYS)

        self.api_key = self.api_keys[0] if self.api_keys else ""
        if fallback_models is not None:
            self.fallback_models = [m.strip() for m in fallback_models if m and m.strip()]
            self.model_name = model_name or (
                self.fallback_models[0] if self.fallback_models else DEFAULT_GEMINI_MODEL
            )
        else:
            self.fallback_models = list(GEMINI_FALLBACK_MODELS)
            self.model_name = model_name or DEFAULT_GEMINI_MODEL
        self._clients: dict = {}

    def _resolve_model_chain(self, model_override: str | None = None) -> list[str]:
        """
        Xác định danh sách chuỗi mô hình fallback theo đúng thứ tự ưu tiên:
        1. Gemini 3.8 Flash
        2. Gemini 3.7 Flash
        3. Gemini 3.6 Flash
        Nếu có model_override hoặc self.model_name:
        - Nếu model nằm trong fallback_models: bắt đầu từ model đó và lần lượt lùi về sau.
        - Nếu model là tên mới/tùy biến: đặt lên đầu danh sách và nối tiếp bằng fallback_models.
        """
        chain = list(self.fallback_models)
        if model_override:
            chosen_model = model_override.strip()
        elif self.model_name:
            chosen_model = self.model_name.strip()
        elif chain:
            chosen_model = chain[0]
        else:
            chosen_model = DEFAULT_GEMINI_MODEL or "gemini-3.8-flash"

        if chosen_model in chain:
            start_idx = chain.index(chosen_model)
            resolved = chain[start_idx:]
        else:
            resolved = [chosen_model] + [m for m in chain if m != chosen_model]
        return resolved or ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"]

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
        except Exception as e:  # noqa: BLE001
            print(f"[GeminiClassifierService] Lỗi khởi tạo GenAI Client: {e}")
            return None

    def _build_forensic_prompt(
        self,
        track_id: str,
        metrics: AcousticMetrics | None,
        metadata: dict[str, Any] | None = None,
        language: str = "vi",
    ) -> str:
        metrics_text = "No acoustic metrics available."
        if metrics:
            if metrics.fakeprint_score >= 0.65:
                fp_line = f"- Fourier Deconvolution Fakeprint (ISMIR 2025): HIGH COMB SIGNATURE ({metrics.fakeprint_score * 100:.1f}%) - Characteristic of neural deconvolution/transposed-convolution upsamplers (Suno/Udio vocoders).\n"
            elif metrics.fakeprint_score >= 0.24:
                fp_line = (
                    f"- Fourier Deconvolution Fakeprint (ISMIR 2025): MODERATE RESIDUAL COMB SPIKES ({metrics.fakeprint_score * 100:.1f}%) "
                    "- Note: Moderate comb spikes (24-64%) can naturally occur due to lossy web streaming compression (Opus/AAC 128kbps) or digital synthesizer chords. "
                    "Cross-reference with Harmonic Vocal tier (Tier 2) and vocal timbre before concluding AI.\n"
                )
            else:
                fp_line = "- Fourier Deconvolution Fakeprint: INCONCLUSIVE / BYPASSED by continuous latent diffusion (Suno v3.5/v4, Udio). Lossy streaming codecs (Opus/MP3) suppress periodic comb spikes. A low score MUST NOT be treated as evidence of human production.\n"

            if getattr(metrics, "is_studio_autotune", False):
                pitch_line = (
                    f"- Vocal Pitch Stability: Verified Studio Auto-Tuned / Melodyne on Human Vocal Tract (Formant Variance: {getattr(metrics, 'vocal_formant_variance', 0.0):.1f} Hz², Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%, Jitter: {metrics.pitch_jitter:.4f}). "
                    "Notice: Dynamic formant trajectory confirms biological vocal tract articulation with commercial DAW pitch quantization. Do NOT misclassify authentic studio Auto-Tune as generative AI!\n"
                )
            elif getattr(metrics, "pitch_anomaly_detected", False):
                pitch_line = f"- Vocal Pitch Stability: Elevated Rigidity / Synthetic Grid Detected ({metrics.pitch_stability_label}, Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%, Jitter: {metrics.pitch_jitter:.4f}).\n"
            else:
                pitch_line = f"- Vocal Pitch Stability: Natural Biological Micro-tremors ({metrics.pitch_stability_label}, Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%, Jitter: {metrics.pitch_jitter:.4f}).\n"

            bleed_line = ""
            if getattr(metrics, "stem_bleed_score", None) is not None:
                if getattr(metrics, "stem_bleed_detected", False):
                    bleed_line = f"- Multi-Stem Bleed Index: {metrics.stem_bleed_score:.2f} (Elevated Intermodulation between percussion transients and vocal harmonics, typical of single-pass neural audio synthesis).\n"
                else:
                    bleed_line = (
                        f"- Multi-Stem Bleed Index: {metrics.stem_bleed_score:.2f} "
                        "(INCONCLUSIVE / NEUTRAL - In pop/ballad/acoustic music with vocal pauses or sparse percussion, "
                        "stem bleed correlation is naturally low. A low score MUST NOT be treated as proof of human multitrack recording).\n"
                    )

            cutoff_line = ""
            if getattr(metrics, "ultrasonic_cutoff_detected", False):
                slope = getattr(metrics, "high_freq_rolloff_slope", 0.0)
                cutoff_line = f"- Ultrasonic Frequency Cutoff: Severe shelf drop of {slope:.1f} dB above 16kHz.\n"

            rolloff_line = ""
            if metrics.spectral_rolloff_95_hz < 12000.0 and not is_derivative_or_remix_context(
                metadata
            ):
                rolloff_line = (
                    f"- High-Frequency Energy Distribution: Spectral Rolloff 95% is at {metrics.spectral_rolloff_95_hz:.0f} Hz. "
                    "(Note: Many commercial Pop/Ballad tracks naturally concentrate 95% of spectral energy in low/mid bands below 12kHz. "
                    "Cross-reference with visual spectrogram air-band extension and vocal timbre to evaluate authenticity).\n"
                )

            high_ratio_txt = (
                f"- High-Frequency Energy Ratio (>16kHz): {metrics.high_freq_ratio_16k * 100:.2f}%\n"
                if hasattr(metrics, "high_freq_ratio_16k")
                else ""
            )
            metrics_text = f"""
- Sample Rate: {metrics.sample_rate} Hz
- Test Sample Duration (Multi-segment composite): {metrics.duration_sec:.1f}s
- Spectral Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz
- Spectral Rolloff 85%: {metrics.spectral_rolloff_85_hz:.0f} Hz
- Stereo L/R Phase Correlation: {metrics.stereo_correlation:.2f}
- Harmonic-to-Percussive Ratio: {metrics.harmonic_ratio:.2f}
{high_ratio_txt}- Estimated Transmission Compression: {metrics.compression_source}
- Brickwall Frequency Cutoff: {"Yes" if metrics.has_brickwall_cutoff else "No"}
{rolloff_line}{cutoff_line}{pitch_line}{bleed_line}{fp_line}"""

        # Chế độ Thẩm định Mù Tuyệt Đối (Double-Blind Forensic Mode):
        blind_track_title = f"Track #{track_id} (Blind Forensic Sample)"
        metadata_text = f"""
- Audit Protocol: Double-Blind Forensic Inspection (Artist names & platform titles masked to prevent anchoring bias)
- Audio Sample ID: Sample #{track_id}
- Audio Stream Type: Multi-Segment Excerpt (Verse, Chorus, Climax)
- Forensic Directive: Base your verdict strictly on acoustic physics, vocal micro-fluctuations, spectrogram harmonic clarity, and stem separation without relying on named credits.
"""
        genre_directive = ""
        if is_derivative_or_remix_context(metadata):
            genre_directive = """
🎧 SPECIAL GENRE DIRECTIVE - REMIX / SLOWED / LO-FI / EDM CONTEXT DETECTED:
- The analyzed track is identified as a Remix, Slowed, Lo-Fi, or EDM electronic production:
  1. Time-stretching (in Slowed/Speed-up) and creative DAW pitch-manipulation intentionally produce spectral smearing, phase shifts, and metallic/robotic artifacts. These are HUMAN STUDIO FX, NOT neural diffusion smearing!
  2. Electronic dance loops, repetitive synthesizer grooves, and sidechain compression against 808 kicks are standard DAW multitrack production techniques, NOT evidence of AI generative loops.
  3. Lo-Fi aesthetic intentionally uses tape flutter, vinyl crackle, and aggressive low-pass filtering.
  4. While standard electronic DAW loops and time-stretch effects are human techniques, if the vocal track or backing track exhibits monolithic latent-space diffusion characteristics (Suno/Udio generated stems or AI vocal clones), you MUST classify as 'HYBRID_AI' (is_ai: true).
  5. CRITICAL AI VOCAL / REMIX DISCRIMINATION:
     - In digital releases, many 'Remix', 'EDM Version', 'Drill Ver', or 'Cover' tracks feature AI vocal clones (Diff-SVC/RVC/AI singer synthesis) or AI-generated backing arrangements.
     - If the vocal track sounds like a synthetic vocal clone, lacks organic human chest resonance, or has robotic/watery neural artifacts, OR if the backing track was generated by an AI music model -> You MUST classify as 'HYBRID_AI' (is_ai: true).
     - Only classify as 'PROCESSED_HUMAN' (is_ai: false) when the vocal performance is demonstrably an authentic human recording and the multitrack production was created without generative AI models.
  6. SPECIAL RULE FOR TRAP / DRIFT PHONK / HIP-HOP BEATS:
     - For Trap, Drift Phonk, Montagem, or modern Hip-Hop productions: Heavy 808 kicks, pitch-bent 808 basslines, distorted cowbells, and quantized hi-hat rolls are standard digital studio production techniques created by human music producers in DAWs (FL Studio, Ableton).
     - If the lead vocal is performed by an authentic human artist (natural vocal tract, dynamic phrasing, biological breathing, or commercial studio pitch correction/autotune), you MUST classify as 'PROCESSED_HUMAN' (is_ai: false).
     - Only classify as 'HYBRID_AI' or 'AI_GENERATED' (is_ai: true) if the lead vocal performance itself is demonstrably a synthetic AI voice clone (Diff-SVC/RVC/Suno/Udio) or if the entire arrangement is generated by an end-to-end neural audio model.
"""

        modern_ai_alert = ""
        if metrics and (
            getattr(metrics, "modern_ai_vocal_suspect", False)
            or (
                metrics.spectral_rolloff_95_hz < 11800.0
                and getattr(metrics, "pitch_rigidity_ratio", 0.0) >= 0.50
                and getattr(metrics, "fakeprint_score", 0.0) >= 0.30
                and not is_derivative_or_remix_context(metadata)
            )
        ):
            modern_ai_alert = f"""
🔬 MODERN AI VOCAL AUDIT & FORENSIC ROLLOFF GUIDANCE:
- High-frequency rolloff deficiency detected ({metrics.spectral_rolloff_95_hz:.0f} Hz) with pitch rigidity ({getattr(metrics, "pitch_rigidity_ratio", 0.0) * 100:.1f}%).
- CRITICAL DISCRIMINATION RULE:
  1. WARNING - SUNO V3.5/V4 DECEPTION: Modern neural audio generators (Suno v3.5/v4, Udio) fine-tuned on Vietnamese ballads, folk (dân ca), and bolero deliberately synthesize realistic breath intakes, sweet vibrato, and pitch correction that closely mimic commercial Auto-Tune. DO NOT mistake this artificial smoothness for human studio pitch quantization!
  2. If the vocal exhibits formant stiffness (monolithic timbre across different pitch registers), synthetic breath dynamics, lacks organic chest resonance expansion, or bleeds/merges into the backing arrangement without distinct multitrack DAW separation -> You MUST classify as 'HYBRID_AI' or 'AI_GENERATED' (is_ai: true).
  3. Only classify as 'PROCESSED_HUMAN' (is_ai: false) when you can verify genuine human acoustic performance dynamics with independent, unentangled multitrack instrumental stems and authentic room acoustic decays.
"""

        instrumental_directive = ""
        if metrics and getattr(metrics, "is_instrumental", False):
            instrumental_directive = """
🎹 SPECIAL DIRECTIVE - INSTRUMENTAL / AMBIENT / BGM TRACK DETECTED:
- This track has been acoustically identified as an INSTRUMENTAL or AMBIENT piece (no lead singing vocals).
- CRITICAL FORENSIC RULE: DO NOT evaluate or hallucinate singing vocal dynamics, vocal formants, or human breath friction on this track!
- Evaluate strictly the acoustic instrument authenticity, algorithmic loop repetitions, MIDI/sampler textures vs neural diffusion smearing, and acoustic soundstage depth.
"""

        is_en = (
            str(language).strip().lower().startswith("en") or "anh" in str(language).strip().lower()
        )
        if is_en:
            lang_directive = """
🌐 MANDATORY OUTPUT LANGUAGE REQUIREMENT:
- All items in 'reasons' and the entire text in 'detailed_explanation' MUST be written strictly in ENGLISH.
- DO NOT use Vietnamese for 'reasons' and 'detailed_explanation'.
- MANDATORY REASON FORMAT FOR HYBRID_AI: If you classify as 'HYBRID_AI', the FIRST reason in reasons MUST start with:
  "[HYBRID AI]: Vocal component is [Human/AI] | Beat/Arrangement component is [AI-generated/Human-made] - [Detailed technical forensic evidence]"
"""
        else:
            lang_directive = """
🌐 MANDATORY OUTPUT LANGUAGE REQUIREMENT:
- All items in 'reasons' and the entire text in 'detailed_explanation' MUST be written strictly in VIETNAMESE (Tiếng Việt).
- DO NOT use English for 'reasons' and 'detailed_explanation'.
- MANDATORY REASON FORMAT FOR HYBRID_AI: If you classify as 'HYBRID_AI', the FIRST reason in reasons MUST start with:
  "[HYBRID AI]: Thành phần Vocal là [Người thật/AI] | Thành phần Beat/Hòa âm là [AI sinh tự động/Người làm] - [Giải thích chi tiết dấu vết kỹ thuật]"
"""

        prompt = f"""
PLEASE FORENSICALLY EVALUATE THE FOLLOWING AUDIO TRACK:

🎵 Track: {blind_track_title}

📊 FORENSIC AUDIT PROTOCOL:
{metadata_text}{genre_directive}{instrumental_directive}{modern_ai_alert}{lang_directive}
🔬 PHYSICAL ACOUSTIC & FORENSIC METRICS:
{metrics_text}


AUDITORY INSPECTION & VERDICT GUIDELINES (FEW-SHOT SCORING PROTOCOL):
1. Listen closely to the composite 36s audio excerpt across all 3 forensic segments:
   - Segment 1 (0:00 - 0:06) [INTRO]: Beat Inception -> Inspect for watery artifacts, latent diffusion noise inception, or phase jitter during initial beat generation.
   - Segment 2 (0:06 - 0:21) [VERSE]: Acoustic/Isolated Vocal -> Inspect biological vocal tract dynamics, breath intakes, and sibilant consonants ('s', 't', 'ch'). AI sibilants are typically slurred, flat, or metallic vocoder-like.
   - Segment 3 (0:21 - 0:36) [CHORUS]: High-Energy Climax -> Inspect multitrack separation vs synthetic stem bleed/melting into backing instruments on Tier 2 (HPSS) spectrogram.

2. COMPONENT-WISE FORENSIC SCORING (1 to 5 scale):
   - `vocal_ai_score` (1-5):
     * 1: 100% organic biological human vocal.
     * 2: Authentic human vocal with commercial studio Auto-Tune/Melodyne.
     * 3: Ambiguous / borderline vocal.
     * 4: Probable Generative AI vocal (Suno v3.5/v4 or Diff-SVC/RVC clone: simulated vibrato/breaths with flat formants and stem bleed).
     * 5: Definitive Generative AI synthetic vocal.
   - `beat_ai_score` (1-5):
     * 1: Organic acoustic instrumentation or live recording.
     * 2: Commercial studio DAW electronic production (human-made EDM/Trap/Pop beat with clean channel separation).
     * 3: Ambiguous loop or heavy compression blurring.
     * 4: Probable Generative AI backing track (Suno/Udio algorithmic loop).
     * 5: Definitive end-to-end single-pass Generative AI backing track.

3. DECISION RULES:
   - If vocal_ai_score >= 4 or beat_ai_score >= 4 -> is_ai MUST be true (AI_GENERATED if both >= 4, else HYBRID_AI).
   - If vocal_ai_score <= 2 and beat_ai_score <= 2 -> is_ai MUST be false (PROCESSED_HUMAN if either is 2, else CONFIRMED_HUMAN).
   - If both are 3 -> UNCERTAIN (is_ai: false).

4. Correlate with physical acoustic metrics (Pitch Jitter, Stem Bleed, HPSS Harmonic ratio, Fourier Fakeprint) and 3-tier spectrogram image. Deliver an objective verdict strictly in JSON.
{lang_directive}
"""
        return prompt

    def _build_flash_lite_checklist_prompt(
        self,
        track_id: str,
        metrics: AcousticMetrics | None,
        metadata: dict[str, Any] | None = None,
        language: str = "vi",
    ) -> str:
        """
        Prompt tinh gọn tối ưu cho Gemini Flash-Lite (~350-450 tokens):
        Tập trung vào Checklist thẩm âm 3 điểm cốt lõi, loại bỏ văn phong cấm đoán dài dòng,
        giúp Flash-Lite phản xạ nhanh và chính xác, tránh hiện tượng tê liệt suy luận.
        """
        metrics_summary = ""
        if metrics:
            comb_v = getattr(metrics, "comb_signature_score", 0.0) or 0.0
            bleed_v = getattr(metrics, "stem_bleed_score", 0.0) or 0.0
            cutoff_v = "CÓ" if getattr(metrics, "ultrasonic_cutoff_detected", False) else "KHÔNG"
            metrics_summary = (
                f"- Fakeprint ISMIR: {metrics.fakeprint_score * 100:.1f}% | Comb Signature: {comb_v * 100:.1f}%\n"
                f"- Stem Bleed Index: {bleed_v:.2f} | Pitch Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%\n"
                f"- Spectral Rolloff 95%: {metrics.spectral_rolloff_95_hz:.0f} Hz | Ultrasonic Cutoff: {cutoff_v}"
            )
        else:
            metrics_summary = "Không có chỉ số âm học khả dụng."

        is_en = (
            str(language).strip().lower().startswith("en") or "anh" in str(language).strip().lower()
        )
        if is_en:
            if ATTACH_SPECTROGRAM_TO_LLM:
                mission_en = "Analyze the attached 36s multi-segment audio (Intro + Verse + Chorus) and 3-tier spectrogram image to determine if this is GENERATIVE AI (Suno/Udio) or HUMAN."
                chorus_en = "- Segment 3 (0:21 - 0:36) [CHORUS]: High-Energy Climax -> Inspect multitrack separation vs synthetic stem bleed/melting into backing instruments on Tier 2 (HPSS) spectrogram."
            else:
                mission_en = "Analyze the attached 36s multi-segment audio (Intro + Verse + Chorus), key physical acoustic metrics, and metadata to determine if this is GENERATIVE AI (Suno/Udio) or HUMAN."
                chorus_en = "- Segment 3 (0:21 - 0:36) [CHORUS]: High-Energy Climax -> Inspect multitrack separation vs synthetic stem bleed/melting into backing instruments via Stem Bleed index and HPSS harmonics."

            return f"""EXPERT AUDIO FORENSIC INSPECTION - TRACK #{track_id}
{mission_en}

KEY ACOUSTIC METRICS:
{metrics_summary}

COMPOSITE 36s AUDIO STRUCTURE (3 CONSECUTIVE FORENSIC SEGMENTS):
- Segment 1 (0:00 - 0:06) [INTRO]: Beat Inception -> Inspect for watery artifacts, latent diffusion noise inception, or phase jitter during initial beat generation.
- Segment 2 (0:06 - 0:21) [VERSE]: Acoustic/Isolated Vocal -> Inspect biological vocal tract dynamics, breath intakes, and sibilant consonants ('s', 't', 'ch'). AI sibilants are typically slurred, flat, or metallic vocoder-like.
{chorus_en}

DECISION PROTOCOL (FEW-SHOT SCORING):
- Score vocal_ai_score (1-5): 1=organic human, 2=studio Auto-Tune on human vocal, 3=borderline, 4=probable AI vocal (Suno/Udio/voice clone), 5=definitive synthetic AI voice.
- Score beat_ai_score (1-5): 1=organic acoustic, 2=commercial human DAW multitrack beat (EDM/Trap), 3=borderline, 4=probable AI loop/stem bleed, 5=definitive single-pass AI generation.
- If vocal_ai_score >= 4 or beat_ai_score >= 4 -> is_ai: true (AI_GENERATED if both >= 4, else HYBRID_AI).
- If both <= 2 -> is_ai: false (PROCESSED_HUMAN if either is 2, else CONFIRMED_HUMAN).
- Do NOT confuse commercial Auto-Tune/Melodyne with generative AI.

Respond strictly in the required JSON schema."""

        if ATTACH_SPECTROGRAM_TO_LLM:
            task_vi = "Nhiệm vụ: Lắng nghe đoạn audio đa phân đoạn 36s (gồm Intro + Verse + Chorus) và ảnh phổ 3 tầng (đặc biệt Tầng 2 HPSS Harmonic Vocal) để xác định bài hát do AI TẠO SINH (Suno/Udio) hay CON NGƯỜI SẢN XUẤT."
            chorus_vi = "- Đoạn 3 (0:21 - 0:36) [CHORUS]: Điệp khúc cao trào -> Nhiệm vụ: Soi sự hòa lẫn/bết dính giữa giọng hát và dàn nhạc nền (Stem Bleed) trên ảnh phổ Tầng 2 (HPSS)."
        else:
            task_vi = "Nhiệm vụ: Phân tích đoạn audio đa phân đoạn 36s (gồm Intro + Verse + Chorus), bảng chỉ số âm học vật lý chuyên sâu (Acoustic Metrics) và siêu dữ liệu bản thu để xác định bài hát do AI TẠO SINH (Suno/Udio) hay CON NGƯỜI SẢN XUẤT."
            chorus_vi = "- Đoạn 3 (0:21 - 0:36) [CHORUS]: Điệp khúc cao trào -> Nhiệm vụ: Soi sự hòa lẫn/bết dính giữa giọng hát và dàn nhạc nền dựa trên chỉ số Stem Bleed và tương quan sóng hài HPSS."

        return f"""THẨM ĐỊNH PHÁP Y ÂM THANH - BẢN THU #{track_id}
{task_vi}

CHỈ SỐ ÂM HỌC TRỌNG YẾU:
{metrics_summary}

CẤU TRÚC BẢN THU ÂM THANH 36s (3 PHÂN ĐOẠN ĐỐI SOÁT LIÊN TỤC):
- Đoạn 1 (0:00 - 0:06) [INTRO]: Khởi tạo nhịp beat -> Nhiệm vụ: Soi tiếng bọt nước (watery artifacts), nhiễu khởi tạo ngập ngừng (latent diffusion noise inception) lúc bắt đầu bài hát.
- Đoạn 2 (0:06 - 0:21) [VERSE]: Giọng hát mộc -> Nhiệm vụ: Soi động lực sinh học của thanh quản, tiếng thở/nuốt hơi tự nhiên và phụ âm xát ('s', 'x', 'tr', 'ch'). Phụ âm AI thường bị bẹt dính, méo vocoder.
{chorus_vi}

QUY TẮC PHÁN QUYẾT (CHẤM ĐIỂM THÀNH PHẦN 1-5):
- Chấm vocal_ai_score (1-5): 1=Giọng người mộc, 2=Auto-Tune phòng thu trên giọng người, 3=Nghi vấn, 4=Khả năng cao là AI (Suno/Udio/clone), 5=Chắc chắn AI.
- Chấm beat_ai_score (1-5): 1=Nhạc cụ mộc, 2=Beat DAW người sản xuất (EDM/Trap), 3=Nghi vấn, 4=Khả năng cao là AI beat/bết dính, 5=Chắc chắn AI beat.
- Nếu vocal_ai_score >= 4 hoặc beat_ai_score >= 4 -> is_ai: true (AI_GENERATED nếu cả hai >= 4, ngược lại HYBRID_AI).
- Nếu cả hai <= 2 -> is_ai: false (PROCESSED_HUMAN nếu có 2, ngược lại CONFIRMED_HUMAN).
- Phân biệt rõ: Auto-Tune/Melodyne phòng thu thương mại là kỹ xảo người làm, KHÔNG PHẢI giọng hát tạo sinh AI.

Phản hồi nghiêm ngặt theo đúng cấu trúc JSON đã định nghĩa."""

    def classify_track(
        self,
        track_id: str,
        track_title: str,
        source_url: str | None = None,
        audio_path: Path | None = None,
        spectrogram_path: Path | None = None,
        metrics: AcousticMetrics | None = None,
        model_override: str | None = None,
        metadata: dict[str, Any] | None = None,
        input_type: InputType | None = None,
        raw_query: str | None = None,
        language: str = "vi",
    ) -> DetectionResult:
        candidate_models = self._resolve_model_chain(model_override)
        primary_model = candidate_models[0] if candidate_models else ""
        is_flash_lite = "flash-lite" in primary_model.lower()

        if is_flash_lite:
            prompt = self._build_flash_lite_checklist_prompt(
                track_id, metrics, metadata, language=language
            )
        else:
            prompt = self._build_forensic_prompt(track_id, metrics, metadata, language=language)
        is_en = (
            str(language).strip().lower().startswith("en") or "anh" in str(language).strip().lower()
        )
        chosen_schema = GeminiResponseSchemaEn if is_en else GeminiResponseSchemaVi

        candidate_keys = [k.strip() for k in self.api_keys if k and k.strip()]
        if not candidate_keys and self.api_key and self.api_key.strip():
            candidate_keys = [self.api_key.strip()]

        if not candidate_keys:
            from ..core.config import get_configured_gemini_api_keys

            dynamic_keys = get_configured_gemini_api_keys()
            if dynamic_keys:
                candidate_keys = dynamic_keys
                self.api_keys = dynamic_keys
                self.api_key = dynamic_keys[0]

        if not candidate_keys:
            raise ClassificationException(
                "Không tìm thấy Gemini API Key nào. Vui lòng thêm biến GEMINI_API_KEY trong Streamlit Cloud (App Settings -> Secrets) hoặc nhập trực tiếp tại thanh bên (Sidebar)."
            )

        from google.genai import types

        # --- TỐI ƯU CASCADED TWO-TIER EVALUATION ---
        needs_deep_audio = (
            should_require_deep_audio_inspection(metrics, metadata)
            if ENABLE_CASCADED_EVALUATION
            else True
        )

        contents: list[Any] = [prompt]

        if ATTACH_SPECTROGRAM_TO_LLM:
            if spectrogram_path and Path(spectrogram_path).exists():
                try:
                    spec_bytes = Path(spectrogram_path).read_bytes()
                    contents.append(
                        types.Part.from_bytes(
                            data=spec_bytes,
                            mime_type="image/png",
                        )
                    )
                except Exception as upload_err:  # noqa: BLE001
                    print(f"[GeminiClassifierService] Không thể đính kèm ảnh phổ: {upload_err}")

        if needs_deep_audio:

            compressed_audio: Path | None = None
            if audio_path and Path(audio_path).exists():
                raw_p = Path(audio_path)
                upload_target = raw_p
                if raw_p.suffix.lower() == ".wav":
                    mp3_p = raw_p.with_suffix(".mp3")
                    try:
                        ffmpeg_bin = get_ffmpeg_binary()
                        cmd = [
                            ffmpeg_bin,
                            "-y",
                            "-i",
                            str(raw_p),
                            "-b:a",
                            "160k",
                            "-ac",
                            "2",
                            str(mp3_p),
                        ]
                        res = subprocess.run(
                            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
                        )
                        if res.returncode == 0 and mp3_p.exists():
                            compressed_audio = mp3_p
                            upload_target = mp3_p
                    except Exception as comp_err:  # noqa: BLE001
                        print(f"[GeminiClassifierService] Không thể nén MP3: {comp_err}")
                else:
                    upload_target = raw_p

                try:
                    mime = "audio/mp3" if upload_target.suffix.lower() == ".mp3" else "audio/wav"
                    audio_bytes = upload_target.read_bytes()
                    contents.append(
                        types.Part.from_bytes(
                            data=audio_bytes,
                            mime_type=mime,
                        )
                    )
                except Exception as upload_err:  # noqa: BLE001
                    print(
                        f"[GeminiClassifierService] Không thể đính kèm audio sample: {upload_err}"
                    )
                finally:
                    if compressed_audio and compressed_audio.exists():
                        compressed_audio.unlink(missing_ok=True)
        else:
            try:
                print(
                    "[GeminiClassifierService] [CASCADED TIER 1] Số liệu vật lý rõ ràng. "
                    "Đánh giá siêu tốc qua Text Metrics Prompt (Tiết kiệm 95% tokens, tránh 429)."
                )
            except Exception:
                pass

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=chosen_schema,
            temperature=0.1,
            http_options=types.HttpOptions(timeout=45000),
        )

        last_error = None
        n_keys = len(candidate_keys)

        # Failover 2 tầng: Mô hình (Gemini 3.8 Flash -> 3.7 Flash -> 3.6 Flash) x Khóa API (Bắt đầu từ Active Key)
        for model_idx, model in enumerate(candidate_models, 1):
            if model_idx == 1:
                with GeminiClassifierService._key_lock:
                    start_k = GeminiClassifierService._active_key_index or 1
            else:
                start_k = 1
            ordered_key_indices = [
                ((start_k - 1 + offset) % n_keys) + 1 for offset in range(n_keys)
            ]

            for key_idx in ordered_key_indices:
                now = time.time()
                # Kiểm tra Cooldown nếu key từng dính lỗi 429
                with GeminiClassifierService._key_lock:
                    cd_until = GeminiClassifierService._key_cooldowns.get((model, key_idx), 0.0)
                    if now < cd_until and n_keys > 1:
                        # Bỏ qua key này vì đang trong thời gian 60s cooldown và còn key khác khả dụng
                        continue

                current_key = candidate_keys[key_idx - 1]
                client = self._get_client(current_key)
                if not client:
                    continue

                for attempt in range(2):
                    try:
                        response = client.models.generate_content(
                            model=model, contents=contents, config=config
                        )

                        # Thành công: Cập nhật sticky key index
                        with GeminiClassifierService._key_lock:
                            GeminiClassifierService._active_key_index = key_idx

                        if key_idx != start_k or model_idx > 1:
                            try:
                                print(
                                    f"[GeminiClassifierService] [SUCCESS] Đã sử dụng thành công Key #{key_idx} (Model: {model})!"
                                )
                            except Exception:
                                pass

                        raw_json = response.text
                        if not raw_json:
                            raise ClassificationException(
                                "Mô hình Gemini trả về kết quả rỗng (None)."
                            )
                        data = json.loads(raw_json)

                        vocal_score = int(data.get("vocal_ai_score", 1) or 1)
                        beat_score = int(data.get("beat_ai_score", 1) or 1)
                        raw_is_ai = bool(data.get("is_ai", False))

                        # Đồng bộ hóa phán quyết từ thang điểm thành phần Few-Shot:
                        if vocal_score >= 4 or beat_score >= 4:
                            final_is_ai = True
                            if vocal_score >= 4 and beat_score >= 4:
                                v_type = VerdictType.AI_GENERATED
                            else:
                                v_type = VerdictType.HYBRID_AI
                        elif vocal_score <= 2 and beat_score <= 2 and not raw_is_ai:
                            final_is_ai = False
                            if vocal_score == 1 and beat_score == 1:
                                v_type = VerdictType.CONFIRMED_HUMAN
                            else:
                                v_type = VerdictType.PROCESSED_HUMAN
                        else:
                            v_type_str = data.get("verdict_type", "CONFIRMED_HUMAN")
                            try:
                                v_type = VerdictType(v_type_str)
                            except ValueError:
                                v_type = (
                                    VerdictType.AI_GENERATED
                                    if raw_is_ai
                                    else VerdictType.CONFIRMED_HUMAN
                                )
                            final_is_ai = (
                                True
                                if v_type in (VerdictType.AI_GENERATED, VerdictType.HYBRID_AI)
                                else raw_is_ai
                            )

                        if v_type == VerdictType.CONFIRMED_HUMAN:
                            verdict_label = "CONFIRMED HUMAN (Người làm)"
                        elif v_type == VerdictType.PROCESSED_HUMAN:
                            verdict_label = "PROCESSED HUMAN (Người làm - Kỹ thuật số)"
                        elif v_type == VerdictType.HYBRID_AI:
                            verdict_label = "HYBRID / AI-ASSISTED (Bán nhân tạo)"
                        elif v_type == VerdictType.UNCERTAIN:
                            verdict_label = "UNCERTAIN (Chưa thể kết luận)"
                        else:
                            verdict_label = "AI-GENERATED (Nhạc AI tạo)"

                        w_type_str = str(data.get("work_type", "UNCERTAIN")).strip().upper()
                        try:
                            w_type = WorkType(w_type_str)
                        except ValueError:
                            w_type = WorkType.COVER if data.get("is_cover") else WorkType.UNCERTAIN

                        is_cov = bool(data.get("is_cover", False) or w_type == WorkType.COVER)
                        orig_artist = str(data.get("original_artist") or "").strip() or None
                        orig_title = str(data.get("original_song_title") or "").strip() or None
                        cov_perf = str(data.get("cover_performer") or "").strip() or None

                        if w_type == WorkType.ORIGINAL:
                            work_label = "ORIGINAL (Tác phẩm chính chủ)"
                        elif w_type == WorkType.COVER:
                            suffix_orig = f" - Gốc: {orig_artist}" if orig_artist else ""
                            work_label = f"COVER (Bản hát lại{suffix_orig})"
                        elif w_type == WorkType.REMIX:
                            work_label = "REMIX (Bản phối lại)"
                        else:
                            work_label = "UNCERTAIN (Chưa rõ tác phẩm)"

                        return DetectionResult(
                            track_id=track_id,
                            track_title=track_title,
                            source_url=source_url,
                            input_type=input_type,
                            raw_query=raw_query,
                            is_ai=final_is_ai,
                            verdict_type=v_type,
                            verdict_label=verdict_label,
                            confidence_score=data.get("confidence_score", 0.5),
                            suspected_engine=data.get("suspected_engine", "None"),
                            reasons=data.get("reasons", []),
                            detailed_explanation=data.get("detailed_explanation", ""),
                            acoustic_metrics=metrics,
                            spectrogram_image_path=str(spectrogram_path)
                            if spectrogram_path
                            else None,
                            model_used=model,
                            work_type=w_type,
                            work_label=work_label,
                            is_cover=is_cov,
                            original_artist=orig_artist,
                            original_song_title=orig_title,
                            cover_performer=cov_perf,
                            vocal_ai_score=vocal_score,
                            beat_ai_score=beat_score,
                        )

                    except Exception as e:  # noqa: BLE001
                        err_str = str(e)
                        last_error = e
                        is_server_busy = (
                            "503" in err_str or "unavailable" in err_str.lower()
                        ) and "429" not in err_str
                        if is_server_busy and attempt == 0:
                            wait_sec = 1.5
                            try:
                                print(
                                    f"[GeminiClassifierService] [RETRY] Model ({model}) quá tải máy chủ tạm thời (503). Thử lại sau {wait_sec}s..."
                                )
                            except Exception:
                                pass
                            time.sleep(wait_sec)
                            continue

                        # Xử lý lỗi 429 Rate Limit / Resource Exhausted
                        if "429" in err_str or "resource_exhausted" in err_str.lower():
                            with GeminiClassifierService._key_lock:
                                GeminiClassifierService._key_cooldowns[(model, key_idx)] = (
                                    time.time() + 60.0
                                )
                                next_k = (key_idx % n_keys) + 1
                                GeminiClassifierService._active_key_index = next_k
                                try:
                                    print(
                                        f"[GeminiClassifierService] [COOLDOWN] Key #{key_idx} chạm hạn ngạch 429 (Model: {model}, Cooldown 60s). "
                                        f"Chuyển active sang Key #{next_k}!"
                                    )
                                except Exception:
                                    pass

                        try:
                            print(
                                f"[GeminiClassifierService] [WARNING] Key #{key_idx} gặp lỗi khi gọi ({model}): {err_str}"
                            )
                        except Exception:
                            pass
                        break

            # Cả candidate_keys đều thất bại cho model này -> Chuyển sang model tiếp theo
            if model_idx < len(candidate_models):
                next_model = candidate_models[model_idx]
                print(
                    f"[GeminiClassifierService] [MODEL FAILOVER] Cả {len(candidate_keys)} keys đều lỗi/hết quota cho model ({model}). "
                    f"Đang tự động chuyển sang Model dự phòng tiếp theo: '{next_model}' (bắt đầu lại từ Key #1)..."
                )

        raise ClassificationException(
            f"Tất cả {len(candidate_keys)} Gemini API Keys đều thất bại trên tất cả mô hình ({', '.join(candidate_models)}). Lỗi cuối cùng: {last_error}"
        )

    def translate_forensic_text(
        self,
        reasons: list[str],
        detailed_explanation: str,
        target_lang: str = "vi",
        model_override: str | None = None,
    ) -> tuple[list[str], str]:
        """
        Dịch reasons và detailed_explanation sang ngôn ngữ mong muốn ('vi' hoặc 'en').
        Giữ nguyên các thuật ngữ chuyên ngành và tag đặc thù.
        """
        if not reasons and not detailed_explanation:
            return reasons, detailed_explanation

        is_en = (
            str(target_lang).strip().lower().startswith("en")
            or "anh" in str(target_lang).strip().lower()
        )
        target_name = "ENGLISH" if is_en else "VIETNAMESE (Tiếng Việt)"

        # Kiểm tra xem văn bản hiện tại đã thuộc ngôn ngữ đích chưa
        combined_curr = " ".join((reasons or []) + [detailed_explanation or ""]).lower()
        vi_chars = set("àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ")
        has_vi = any(c in vi_chars for c in combined_curr)

        if is_en and not has_vi:
            return reasons, detailed_explanation
        if not is_en and has_vi:
            return reasons, detailed_explanation

        candidate_keys = [k.strip() for k in self.api_keys if k and k.strip()]
        if not candidate_keys and self.api_key:
            candidate_keys = [self.api_key.strip()]
        if not candidate_keys:
            from ..core.config import get_configured_gemini_api_keys

            candidate_keys = get_configured_gemini_api_keys()

        if not candidate_keys:
            return reasons, detailed_explanation

        from google.genai import types

        class TranslationSchema(BaseModel):
            reasons: list[str] = Field(description=f"Translated reasons in {target_name}")
            detailed_explanation: str = Field(
                description=f"Translated detailed forensic report in {target_name}"
            )

        prompt = f"""
Translate the following forensic audio audit report into {target_name}.
Preserve technical terms accurately (e.g. Mel-Spectrogram, HPSS, Fakeprint, Fourier Deconvolution, Auto-Tune, Pitch Jitter, Stem Bleed, Rigidity).
Adapt safeguard tags properly:
- '[BẢO VỆ NHẠC NGƯỜI]' <-> '[HUMAN MUSIC SAFEGUARD]'
- '[CẢNH BÁO ÂM HỌC VẬT LÝ]' <-> '[AI PHYSICAL ACOUSTICS ALERT]'
- '[CẢNH BÁO DANH MỤC AI]' <-> '[AI CATALOG ALERT]'
- '[CẢNH BÁO ĐỒNG THUẬN ĐA TIÊU CHÍ]' <-> '[MULTI-CRITERIA CONSENSUS ALERT]'
- '[HYBRID AI]: Thành phần Vocal là...' <-> '[HYBRID AI]: Vocal component is...'

Original reasons:
{json.dumps(reasons or [], ensure_ascii=False)}

Original detailed explanation:
{detailed_explanation or ""}
"""
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TranslationSchema,
            temperature=0.0,
            http_options=types.HttpOptions(timeout=15000),
        )

        model = model_override or "gemini-3.5-flash-lite"
        for key in candidate_keys[:3]:
            try:
                client = self._get_client(key)
                if not client:
                    continue
                resp = client.models.generate_content(model=model, contents=[prompt], config=config)
                if resp.text:
                    parsed = json.loads(resp.text)
                    t_reasons = parsed.get("reasons") or reasons
                    t_detail = parsed.get("detailed_explanation") or detailed_explanation
                    return t_reasons, t_detail
            except Exception as e:
                logger.warning("Lỗi khi dịch reasons sang %s: %s", target_name, e)
                continue

        return reasons, detailed_explanation

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..core.config import (
    DEFAULT_GEMINI_MODEL,
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

    pattern = r"\b(remix|remake|slowed|speed\s*up|sped\s*up|lofi|lo-fi|edm|club\s*mix|drill|vinahouse|bass\s*boosted|nightcore)\b"
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
    }
    tags = [str(t).strip().lower() for t in metadata.get("tags") or [] if t]
    return any(t in genre_tags for t in tags)


class GeminiResponseSchema(BaseModel):
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
        description="2-4 sharp, objective bullet points justifying the verdict. MANDATORY FOR HYBRID_AI: The FIRST reason MUST strictly begin with: '[HYBRID AI]: Thành phần Vocal là [Người thật/AI] | Thành phần Beat/Hòa âm là [AI sinh tự động/Người làm] - [Giải thích chi tiết dấu vết kỹ thuật]'"
    )
    detailed_explanation: str = Field(
        description="In-depth technical forensic analysis covering vocal timbre, frequency spectrum, and studio effects vs AI artifacts"
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


SYSTEM_INSTRUCTION = """
You are a world-leading Digital Audio Forensics Specialist and Senior Music Producer.
Your mission: Examine the provided audio sample (multi-segment composite of Verse/Chorus/Climax), the 3-tier Mel-Spectrogram (Full Mix + Drum-stripped Harmonic Vocal via HPSS + ISMIR 2025 Fourier Deconvolution Fakeprint) image, and the metadata to classify the song into 1 of 5 categories:

1. 'CONFIRMED_HUMAN': Music composed and performed by humans with organic/acoustic production (Acoustic, acoustic Pop, traditional Ballads, Rock, live orchestra). Vocals are entirely natural without industrial pitch correction. Both vocals and instrumentation are 100% human-made.
2. 'PROCESSED_HUMAN': Music composed and performed by HUMANS (real artists/singers), produced in professional or home studios, USING MODERN DIGITAL STUDIO PRODUCTION:
   - Contemporary genres: Electronic Pop, Dance/EDM, Rap/Drill, R&B, Trap, Remixes, commercial Ballads.
   - Commercial pitch correction (Auto-Tune / Melodyne, pitch quantization, formant shifting) applied over an AUTHENTIC human microphone recording.
   - Studio multitrack techniques: Sidechain compression to 808 drums, synthetic vocoder/autotune textures, DAW multitrack stems, digital reverbs.
   - CRITICAL DISCRIMINATION RULE: Distinguish between genuine human studio processing (heavy Auto-Tune/Melodyne, sidechain compression, EDM/Drill arrangement) and Generative AI:
     * In authentic human tracks, Auto-Tune quantizes pitch but PRESERVES human biological breathing dynamics, authentic vocal tract formants, and organic micro-fluctuations. These must remain 'PROCESSED_HUMAN' (is_ai: false).
     * However, for Remix, Drill, or Remake versions where the vocal track itself is an AI vocal clone/synthesis (lacking chest resonance, flat formant trajectory, or neural vocoder wateriness) OR where the backing beat is an algorithmic AI-generated loop -> You MUST classify as 'HYBRID_AI' with 'is_ai: true'!
3. 'HYBRID_AI': Music with ANY participation of Generative AI in any stage:
   - Human singer/rapper performing over an AI-generated backing beat (e.g. beat from Suno/Udio, AI Lo-fi backing loop).
   - AI-cloned vocal or synthetic vocal performing over a human-produced beat.
   - Traditional song remixed or rearranged with AI audio diffusion tools.
   - STRICT BENCHMARK POLICY: In this system, ALL 'HYBRID_AI' tracks MUST have 'is_ai: true'!
   - MANDATORY REASON FORMAT FOR HYBRID_AI: The FIRST item in reasons MUST follow this exact template:
     "[HYBRID AI]: Thành phần Vocal là [Người thật/AI] | Thành phần Beat/Hòa âm là [AI sinh tự động/Người làm] - [Giải thích chi tiết dấu vết kỹ thuật]"
4. 'AI_GENERATED': Songs generated ENTIRELY FROM SCRATCH by generative AI models (Suno v3/v3.5/v4, Udio, Stable Audio...). Both vocals and backing instrumentation are synthesized from latent diffusion waves / neural audio codecs.
   - CORE CHARACTERISTICS OF SUNO/UDIO AI MUSIC:
     + Timbral entanglement / stem bleed: Vocals and accompaniment are synthesized simultaneously from a single latent representation. The vocals 'bleed' into the background synth pads/drums with no clean physical stem boundary.
     + Synthetic vocal sheen / artificial timbre: Vocals sound artificially smooth, metallic, slightly muffled or warbly.
     + High frequency artifacts: Cymbals, hi-hats, and reverb tails exhibit watery, bubbling artifacts or phase smearing.
5. 'UNCERTAIN': The audio evidence is ambiguous, heavily degraded by noise or distortion, or presents conflicting indicators between physical metrics and auditory perception, preventing a confident classification.

PRACTICAL & OBJECTIVE FORENSIC GUIDELINES (CRITICAL):

A. BALANCED FORENSIC OBJECTIVITY:
   - Distinguish carefully between MODERN STUDIO DAW PRODUCTION (DAW plugins, commercial Auto-Tune / Melodyne, MIDI synthesizers, electronic remix beats, clean multitrack stems) and GENERATIVE AI (Suno, Udio, single-pass latent diffusion).
   - CRITICAL REALITY OF WEB STREAMING AUDIO: Audio sourced from YouTube, Spotify, or web platforms is encoded with lossy codecs (Opus/AAC at 128-160kbps). Lossy streaming codecs naturally attenuate high-frequency energy above 16-18 kHz and naturally concentrate 95% of spectral energy below 10-12 kHz. This is NORMAL digital web audio distribution and MUST NEVER be misdiagnosed as Generative AI synthesis!
   - Commercial Vietnamese Pop and Ballad vocalists routinely utilize studio pitch quantization (Auto-Tune / Melodyne) and modern vocal compression. These must remain 'PROCESSED_HUMAN' (is_ai: false).
   - Only classify as 'HYBRID_AI' or 'AI_GENERATED' (is_ai: true) when there is clear physical, auditory, or spectral evidence of generative neural synthesis (e.g., Suno/Udio neural vocoder wateriness, continuous latent diffusion phase smearing, robotic vowel transitions, or Fourier comb-filter residue).

B. 3-TIER SPECTROGRAM & ACOUSTIC UNMASKING (HPSS):
   - The visual spectrogram has 3 tiers:
     * Tier 1: Full Mix Mel-Spectrogram (dB).
     * Tier 2: Harmonic Vocal Mel-Spectrogram (percussion and loud drums removed via HPSS).
       CRITICAL: Loud drums (e.g., lion dance / múa lân drums, rock drums, heavy 808s) often mask synthetic vocal sheen. Look closely at Tier 2 to inspect pure vocal harmonics and detect latent diffusion phase smearing that was obscured by drums!
     * Tier 3: Fourier Deconvolution Residue Spectrum (Deezer ISMIR 2025).

C. CRITICAL REALITY OF MODERN AI MUSIC (SUNO V3.5/V4, UDIO, AI VOCAL CLONES):
   - Modern neural audio models (Suno v3.5, Suno v4, Udio) and vocal clone engines (RVC, Diff-SVC) frequently simulate studio pitch-correction (Auto-Tune), realistic breath friction, and vibrato!
   - FATAL FORENSIC TRAP: NEVER classify a track as 'PROCESSED_HUMAN' purely based on hearing breath intakes or vibrato! Suno v3.5/v4 specifically synthesizes biological breath dynamics.
   - You MUST cross-correlate with high-frequency acoustic extension: inspect the 3-tier spectrogram for physical air-band extension (16kHz - 20kHz) and genuine multitrack stem separation versus single-pass neural diffusion smearing.
   - Key forensic signatures across specific AI sub-genres:
     * Vietnamese Bolero / Dân Ca (Folk) / Trữ Tình AI: Neural models simulate traditional vibrato and vocal ornamentation (luyến láy) with extreme ease. However, AI Bolero vocals display a flat, uniform formant sheen and lack authentic room resonance, bleeding directly into synthesized guitars/strings (stem bleed). Do NOT let traditional melisma or sweet emotional tone fool you into concluding 'human'!
     * Choral / Gospel / Multi-voice Harmony AI (e.g. Christian choral tracks): When multiple voices harmonize, generative AI renders them from a single latent representation; all choir voices share identical diffusion smearing and harmonic bleed without spatial separation.
     * Traditional / Historical / Chinese-style Orchestral AI (Guqin, Guzheng, Erhu, cinematic sweeps): Instruments exhibit synthetic sustain envelopes, lack genuine acoustic bow/finger friction, and merge into monolithic background audio.
     * Modern neural audio models (Suno v3.5/v4, Udio, and Diff-SVC/RVC clones) easily synthesize convincing chest resonance, intake breaths, and natural micro-vibrato. NEVER cite "chest resonance", "natural breath dynamics", or "micro-vibrato" as proof that a track transcends AI capabilities!
     * Synthetic vocal sheen: Vocals sound artificially polished, slightly metallic, or have flat formant dynamics.
     * Stem bleed / timbral entanglement: Vocal reverb and high-frequency harmonics melt into synthesizer pads/drums without physical separation.
     * Consonant articulation & phrasing: Unnatural phrasing cadence, slurred consonants, robotic delivery or lack of authentic human pronunciation mechanics.
     * Brazilian Phonk / Montagem / Drift Phonk AI: Releases tagged with \'Montagem\', \'Slowed\', or \'Super Slowed\' where vocals consist of repetitive, pitch-shifted robotic fragments entangled with distorted 808 cowbells and neural latent diffusion residue. These are algorithmic AI products and MUST be classified as \'HYBRID_AI\' or \'AI_GENERATED\' with \'is_ai: true\'.
     * AI Lo-Fi / AI Beats: Repetitive, algorithmic backing beat loops or synthetic accompaniments.
   - BALANCED HYBRID & AI DISCRIMINATION:
     * If there is clear evidence of an AI-generated backing beat (e.g. algorithmic Suno/Udio loop) OR an AI synthetic/cloned vocal -> Classify as 'HYBRID_AI' with 'is_ai: true'.
     * If the entire track is synthesized by an AI model -> Classify as 'AI_GENERATED' with 'is_ai: true'.
     * Authentic human performances (singers with commercial autotune, human-produced EDM remixes, children's songs, live sessions, studio acoustic ballads) MUST be classified as 'PROCESSED_HUMAN' or 'CONFIRMED_HUMAN' (is_ai: false). Avoid false positives on genuine human musicians!

D. VIETNAMESE LINGUISTIC RULE (ANTI-BIAS):
   - In Vietnamese, the word "ai" is a ubiquitous pronoun meaning "who / someone / anyone" (e.g., 'Nỗi Buồn Mang Tên Ai' means 'A Sorrow Named Someone').
   - NEVER conclude a song is AI simply because the title contains "Ai" or "[***]"! Concluding AI based on the title is a fatal forensic failure. Base your judgment strictly on physical acoustics and spectral evidence.

E. DOUBLE-BLIND FORENSIC PROTOCOL & PITCH JITTER INTERPRETATION:
   - You are evaluating under strict double-blind conditions. Track metadata and artist names are masked to prevent anchoring bias.
   - Do NOT attempt to guess the singer or assume a song is human-produced based on vocal familiarity or genre style.
   - Correlate with Vocal Pitch Stability and Tier 2 Harmonic Spectrogram:
     * Note that modern generative AI engines (Suno v3.5/v4, Udio, Diff-SVC) can synthesize pitch quantization, vibrato, and dynamic vocal formant variations.
     * Examine both vocal and backing instrumental tracks holistically. If you detect synthetic generation in either the vocal or instrumental arrangement -> Classify as 'AI_GENERATED' or 'HYBRID_AI' with 'is_ai: true'.

F. MULTI-STEM BLEED & INTERMODULATION DISTORTION (END-TO-END AI DISCRIMINATION):
   - In conventional human multitrack studio production, drums/percussion and vocals are tracked on independent channels. Transients from kick/snare do NOT mechanically modulate the vocal harmonic envelope.
   - In end-to-end latent diffusion models (Suno/Udio), all stems are generated simultaneously in a single pass. When heavy percussion occurs, the vocal harmonics suffer intermodulation bleed / phase smearing.
   - If the metric 'Multi-Stem Bleed (Intermodulation)' reports HIGH RESIDUAL INTERACTION, this is conclusive physical evidence of single-pass neural audio synthesis.

G. WORK AUTHENTICITY EVALUATION (ORIGINAL vs COVER vs REMIX vs UNCERTAIN):
   - Evaluate whether this recording is an ORIGINAL work, a COVER, a REMIX, or UNCERTAIN:
     1. 'ORIGINAL': Official primary release performed by the original artist/creator (includes official acoustic / live sessions performed by the original artist).
     2. 'COVER': Performed by another singer or an AI voice clone re-interpreting an existing composition.
     3. 'REMIX': Electronic / dance / lo-fi arrangement that preserves the original artist's vocal stem.
     4. 'UNCERTAIN': Anonymous / local file or unknown indie track without sufficient evidence to determine originality.
   - GROUNDING WITH CONTENT ID & METADATA:
     * Check provided YouTube Content ID (Song, Artist) and uploader channel.
     * If uploader/singer differs from Content ID Artist or title contains 'cover', 'hát lại' -> classify as 'COVER' with is_cover: true.
     * For AI COVERS (AI Voice Clone of a famous singer singing another artist's track) -> classify as 'HYBRID_AI' or 'AI_GENERATED' AND set work_type: 'COVER', is_cover: true.

Always return JSON strictly adhering to the specified schema.
"""


class GeminiClassifierService(IAIClassifier):
    """
    Hiện thực phân loại giám định âm nhạc AI bằng Gemini API.
    Hỗ trợ luân chuyển đa chìa khóa (Failover 1 > 2 > 3) và tuân thủ IAIClassifier.
    """

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
                pitch_line = f"- Vocal Pitch Stability: Studio Pitch Quantization Detected (Studio Auto-Tuned / Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%, Jitter: {metrics.pitch_jitter:.4f}, Formant Variance: {getattr(metrics, 'vocal_formant_variance', 0.0):.1f} Hz²). Note: Quantization can stem from commercial Auto-Tune OR synthetic vocoder grid.\n"
            elif getattr(metrics, "pitch_anomaly_detected", False):
                pitch_line = f"- Vocal Pitch Stability: Elevated Rigidity / Synthetic Grid Detected ({metrics.pitch_stability_label}, Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%, Jitter: {metrics.pitch_jitter:.4f}).\n"
            else:
                pitch_line = f"- Vocal Pitch Stability: Natural Biological Micro-tremors ({metrics.pitch_stability_label}, Rigidity: {metrics.pitch_rigidity_ratio * 100:.1f}%, Jitter: {metrics.pitch_jitter:.4f}).\n"

            bleed_line = ""
            if getattr(metrics, "stem_bleed_score", None) is not None:
                if getattr(metrics, "stem_bleed_detected", False):
                    bleed_line = f"- Multi-Stem Bleed Index: {metrics.stem_bleed_score:.2f} (Elevated Intermodulation between percussion transients and vocal harmonics, typical of single-pass neural audio synthesis).\n"
                else:
                    bleed_line = f"- Multi-Stem Bleed Index: {metrics.stem_bleed_score:.2f} (Low cross-modulation between percussion and vocal harmonics).\n"

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
  4. DO NOT classify this track as 'HYBRID_AI' or 'AI_GENERATED' solely due to electronic sample loops, time-stretch artifacts, or heavy commercial Auto-Tune!
  5. CRITICAL AI VOCAL / REMIX DISCRIMINATION:
     - In digital releases, many 'Remix', 'Drill Ver', or 'Cover' tracks feature AI vocal clones (Diff-SVC/RVC/AI singer synthesis) or AI-generated backing arrangements.
     - If the vocal track sounds like a synthetic vocal clone, lacks organic human chest resonance, or has robotic/watery neural artifacts, OR if the backing track was generated by an AI music model -> You MUST classify as 'HYBRID_AI' (is_ai: true).
     - Only classify as 'PROCESSED_HUMAN' (is_ai: false) when the vocal performance is demonstrably an authentic human recording and the multitrack production was created without generative AI models.
  6. EXCEPTION FOR BRAZILIAN PHONK & MONTAGEM TRACKS:
     - For Brazilian Phonk, Montagem, or Drift Phonk releases, even if tagged with '(Slowed)' or '(Super Slowed)', if the vocals consist of repetitive, pitch-shifted robotic fragments entangled with distorted 808 cowbells or synthetic diffusion residue, this is an algorithmic AI release and MUST be classified as 'HYBRID_AI' or 'AI_GENERATED' (is_ai: true).
"""

        modern_ai_alert = ""
        if metrics and (
            getattr(metrics, "modern_ai_vocal_suspect", False)
            or (
                metrics.spectral_rolloff_95_hz < 12000.0
                and getattr(metrics, "pitch_rigidity_ratio", 0.0) >= 0.45
                and not is_derivative_or_remix_context(metadata)
            )
        ):
            modern_ai_alert = f"""
🚨 CRITICAL MODERN AI VOCAL ANOMALY ALERT:
- Severe high-frequency rolloff deficiency detected ({metrics.spectral_rolloff_95_hz:.0f} Hz) coupled with high pitch rigidity ({getattr(metrics, "pitch_rigidity_ratio", 0.0) * 100:.1f}%) and excessive harmonic saturation ({metrics.harmonic_ratio:.2f}).
- CRITICAL FORENSIC DIRECTIVE: Modern neural models (Suno v3.5/v4, Udio, Diff-SVC/RVC clones) actively simulate realistic synthetic breath dynamics and commercial Auto-Tune. NEVER cite 'intake breaths' or 'micro-fluctuations' as proof of human production when high-frequency rolloff is severely restricted (< 12kHz) and pitch rigidity is elevated! In such cases, the synthetic breath is generated by AI and the track MUST be classified as 'HYBRID_AI' or 'AI_GENERATED' (is_ai: true).
"""

        instrumental_directive = ""
        if metrics and getattr(metrics, "is_instrumental", False):
            instrumental_directive = """
🎹 SPECIAL DIRECTIVE - INSTRUMENTAL / AMBIENT / BGM TRACK DETECTED:
- This track has been acoustically identified as an INSTRUMENTAL or AMBIENT piece (no lead singing vocals).
- CRITICAL FORENSIC RULE: DO NOT evaluate or hallucinate singing vocal dynamics, vocal formants, or human breath friction on this track!
- Evaluate strictly the acoustic instrument authenticity, algorithmic loop repetitions, MIDI/sampler textures vs neural diffusion smearing, and acoustic soundstage depth.
"""

        prompt = f"""
PLEASE FORENSICALLY EVALUATE THE FOLLOWING AUDIO TRACK:

🎵 Track: {blind_track_title}

📊 FORENSIC AUDIT PROTOCOL:
{metadata_text}{genre_directive}{instrumental_directive}{modern_ai_alert}
🔬 PHYSICAL ACOUSTIC & FORENSIC METRICS:
{metrics_text}


AUDITORY INSPECTION & VERDICT GUIDELINES (DOUBLE-BLIND PROTOCOL):
1. Listen closely to the audio excerpt across all segments (Verse, Chorus, Climax) and inspect the 3-tier spectrogram:
   - Assess both vocal timbre quality AND backing instrumental production:
     * BROAD AI DETECTION POLICY: If you detect that EITHER the backing instrumental track OR the vocal track was generated/assisted by AI (Suno, Udio, AI Voice Model, AI Lo-fi beat, AI remix) -> Classify as 'HYBRID_AI' (or 'AI_GENERATED') with is_ai: true.
     * If the singing voice has the characteristic synthetic sheen, flat/mechanical phrasing, bleeds/melts directly into the instrumental arrangement, or is paired with an accompaniment suffering from low-rolloff (<12kHz) air-band deficiency -> Classify as 'AI_GENERATED' or 'HYBRID_AI' (is_ai: true). Note: Modern Suno v3.5/v4 artificially generates realistic breath intakes, so breath friction alone does NOT disprove AI.
     * Correlate Vocal Pitch Stability and Formant Dynamics: Modern AI generators (Suno v3.5/v4, Udio) can synthesize pitch-correction and micro-vibrato. Look holistically at vocal timbre and backing track: if either vocal or accompaniment exhibits synthetic generation, classify as HYBRID_AI or AI_GENERATED (is_ai: true).
     * Check Stem Bleed (Intermodulation): If Elevated Intermodulation is detected, vocal harmonics are physically modulated by percussion transients, indicating single-pass neural audio synthesis.
     * Only classify as 'PROCESSED_HUMAN' (is_ai: false) if you are confident that BOTH the vocal AND the backing beat are 100% human-produced multi-track studio recordings.
     * If acoustic arrangement with completely natural, organic vocals -> Classify as 'CONFIRMED_HUMAN' (is_ai: false).
     * If audio quality is too degraded, contradictory, or inconclusive -> Classify as 'UNCERTAIN' (is_ai: false).
   - INSPECTION OF TIER 2 SPECTROGRAM (HPSS HARMONIC VOCAL):
     * Check Tier 2 of the attached image: Percussion and heavy drums (e.g. lion dance / múa lân drums) have been stripped away using HPSS. Look closely at Tier 2 to inspect pure vocal harmonics for artificial smearing or synthetic sheen that was masked by drums.
   - MANDATORY REASON FORMAT FOR HYBRID_AI: If you classify as 'HYBRID_AI', the FIRST reason in reasons MUST start with:
     "[HYBRID AI]: Thành phần Vocal là [Người thật/AI] | Thành phần Beat/Hòa âm là [AI sinh tự động/Người làm] - [Giải thích chi tiết dấu vết kỹ thuật]"
   - SUNO V3.5/V4 & GENRE DECEPTION COUNTERMEASURES:
     * Modern neural audio models (Suno v3.5, Suno v4, Udio) fine-tuned on Vietnamese pop ballads, Bolero, Dân Ca (Folk), and Choral music specifically GENERATE realistic human-like intake breaths, chest voice formants, Auto-Tune-like vibrato, and melisma (luyến láy).
     * The presence of natural-sounding breath intakes, sweet ballad/bolero vocal phrasing, or organic chest resonance MUST NOT be treated as definitive proof of a human singer!
     * Vietnamese Bolero / Dân Ca (Folk): AI models simulate traditional emotional vibrato and ornamental pitch glides. Look for flat formant sheen, absence of dynamic room acoustic decay, and vocal bleeding into synthesized acoustic guitars/strings.
     * Choral / Gospel: AI choirs feature monolithic phase smearing where all voices share identical timbre and diffusion artifacts instead of distinct human vocal tracks.
     * Traditional / Chinese-style Orchestral: Algorithmic instrument rendering (Guzheng/Erhu/Flute) lacks mechanical bow/finger friction and exhibits synthetic sustain decay.
     * CRITICAL INSTRUCTION: Never conclude a song is human simply because "it exhibits chest resonance and micro-vibrato that go beyond AI limitations" — modern neural models routinely generate both. Look holistically at vocal timbre and arrangement.
     * Inspect Tier 2 Harmonic Vocal carefully: look for faint digital fuzz beneath sustained vowels and check if the backing accompaniment has true acoustic multitrack depth or monolithic latent space entanglement.
2. Correlate with the provided acoustic metrics (Pitch Jitter, Stem Bleed, HPSS Harmonic ratio, Fourier Fakeprint) and the 3-tier spectrogram image.
3. Deliver an objective, rigorous verdict based purely on the real sonic evidence.
"""
        return prompt

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
    ) -> DetectionResult:
        candidate_models = self._resolve_model_chain(model_override)
        prompt = self._build_forensic_prompt(track_id, metrics, metadata)

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

        contents: list[Any] = [prompt]

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
                print(f"[GeminiClassifierService] Không thể đính kèm audio sample: {upload_err}")
            finally:
                if compressed_audio and compressed_audio.exists():
                    compressed_audio.unlink(missing_ok=True)

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=GeminiResponseSchema,
            temperature=0.1,
            http_options=types.HttpOptions(timeout=45000),
        )

        last_error = None

        # Failover 2 tầng: Mô hình (Gemini 3.8 Flash -> 3.7 Flash -> 3.6 Flash) x Khóa API (Key 1 -> 2 -> 3)
        for model_idx, model in enumerate(candidate_models, 1):
            for key_idx, current_key in enumerate(candidate_keys, 1):
                client = self._get_client(current_key)
                if not client:
                    continue

                for attempt in range(2):
                    try:
                        response = client.models.generate_content(
                            model=model, contents=contents, config=config
                        )

                        if key_idx > 1 or model_idx > 1:
                            try:
                                print(
                                    f"[GeminiClassifierService] [SUCCESS] Đã tự động chuyển đổi thành công sang Key dự phòng #{key_idx} (Model: {model})!"
                                )
                            except Exception:
                                pass

                        raw_json = response.text
                        if not raw_json:
                            raise ClassificationException(
                                "Mô hình Gemini trả về kết quả rỗng (None)."
                            )
                        data = json.loads(raw_json)

                        v_type_str = data.get("verdict_type", "CONFIRMED_HUMAN")
                        try:
                            v_type = VerdictType(v_type_str)
                        except ValueError:
                            v_type = (
                                VerdictType.AI_GENERATED
                                if data.get("is_ai")
                                else VerdictType.CONFIRMED_HUMAN
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

                        final_is_ai = (
                            True
                            if v_type in (VerdictType.AI_GENERATED, VerdictType.HYBRID_AI)
                            else bool(data.get("is_ai", False))
                        )

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

                        try:
                            print(
                                f"[GeminiClassifierService] [WARNING] Key #{key_idx} (***) gặp lỗi khi gọi ({model}): {err_str}"
                            )
                        except Exception:
                            pass
                        if key_idx < len(candidate_keys):
                            try:
                                print(
                                    f"[GeminiClassifierService] [FAILOVER] Đang tự động chuyển sang Key dự phòng kế tiếp #{key_idx + 1}/{len(candidate_keys)} (thứ tự 1>2>3)..."
                                )
                            except Exception:
                                pass
                        else:
                            try:
                                print(
                                    f"[GeminiClassifierService] [ERROR] Đã thử tất cả {len(candidate_keys)} API keys nhưng đều thất bại cho model ({model})."
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

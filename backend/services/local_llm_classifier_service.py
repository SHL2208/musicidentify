import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

import requests

from ..core.config import (
    ATTACH_SPECTROGRAM_TO_LLM,
    LOCAL_DEFAULT_MODEL,
    LOCAL_FALLBACK_MODELS,
    LOCAL_MODEL_API_KEY,
    LOCAL_MODEL_BASE_URL,
)
from ..core.exceptions import ClassificationException
from ..core.interfaces.services.classifier_interface import IAIClassifier
from ..core.models.entities import (
    AcousticMetrics,
    DetectionResult,
    VerdictType,
    WorkType,
)
from .gemini_classifier_service import (
    SYSTEM_INSTRUCTION,
    GeminiClassifierService,
)

logger = logging.getLogger(__name__)


class LocalLLMClassifierService(GeminiClassifierService):
    """
    Hiện thực giám định âm nhạc AI bằng Local LLM / OpenAI-compatible Proxy
    (Ví dụ: One-API, New-API, LiteLLM, Ollama, vLLM, LM Studio, v.v.)
    Kế thừa GeminiClassifierService để tái sử dụng toàn bộ hệ thống Forensic Prompt,
    Acoustic Metrics text generation, và model failover.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        fallback_models: list[str] | None = None,
        timeout: float | None = None,
    ):
        raw_url = base_url or LOCAL_MODEL_BASE_URL
        self.base_url = str(raw_url).rstrip("/")
        self.api_key = (api_key or LOCAL_MODEL_API_KEY).strip()
        self.model_name = (model_name or LOCAL_DEFAULT_MODEL).strip()
        self.fallback_models = (
            [m.strip() for m in fallback_models if m and m.strip()]
            if fallback_models is not None
            else list(LOCAL_FALLBACK_MODELS)
        )
        import os
        self.timeout = float(timeout) if timeout is not None else float(os.getenv("LOCAL_MODEL_TIMEOUT", "180"))
        self.api_keys = [self.api_key] if self.api_key else []

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
        input_type: Any | None = None,
        raw_query: str | None = None,
        language: str = "vi",
    ) -> DetectionResult:
        candidate_models = self._resolve_model_chain(model_override)
        primary_model = candidate_models[0] if candidate_models else self.model_name
        is_flash_lite = "flash-lite" in primary_model.lower() or "lite" in primary_model.lower()

        if is_flash_lite:
            prompt = self._build_flash_lite_checklist_prompt(
                track_id, metrics, metadata, language=language
            )
        else:
            prompt = self._build_forensic_prompt(track_id, metrics, metadata, language=language)

        # Chuẩn bị nội dung gửi lên OpenAI-compatible endpoint (/v1/chat/completions)
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Đính kèm ảnh phổ Spectrogram (nếu có và được bật cấu hình) dưới dạng Vision Image URL (base64)
        if ATTACH_SPECTROGRAM_TO_LLM and spectrogram_path and Path(spectrogram_path).exists():
            try:
                img_bytes = Path(spectrogram_path).read_bytes()
                b64_img = base64.b64encode(img_bytes).decode("utf-8")
                user_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_img}"},
                    }
                )
            except Exception as e:
                logger.warning(f"[LocalLLMClassifier] Không thể đính kèm ảnh phổ base64: {e}")
        elif not ATTACH_SPECTROGRAM_TO_LLM:
            logger.info(
                "[LocalLLMClassifier] Chế độ Pure Text siêu tốc (ATTACH_SPECTROGRAM_TO_LLM=False): Bỏ qua gửi ảnh phổ lên LLM."
            )

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": user_content},
        ]

        endpoint = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = None
        for raw_model in candidate_models:
            # Tự động thêm prefix 'antigravity/' nếu người dùng chỉ nhập 'gemini-3.8-flash' hoặc 'gemini-3.8-flash-high'
            if "/" not in raw_model:
                model = f"antigravity/{raw_model}"
            else:
                model = raw_model
            payload = {
                "model": model,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.1,
            }

            try:
                logger.info(
                    f"[LocalLLMClassifier] Gửi yêu cầu giám định tới: {endpoint} (Model: {model})"
                )
                response = requests.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code != 200:
                    err_msg = f"HTTP {response.status_code}: {response.text[:300]}"
                    logger.warning(
                        f"[LocalLLMClassifier] Model {model} trả về lỗi ({err_msg}). Thử fallback..."
                    )
                    last_error = err_msg
                    continue

                res_json = response.json()
                choices = res_json.get("choices", [])
                if not choices:
                    last_error = "Choices rỗng trong response"
                    continue

                raw_content = choices[0].get("message", {}).get("content", "").strip()
                if not raw_content:
                    last_error = "Nội dung phản hồi rỗng từ model"
                    continue

                # Làm sạch markdown nếu model bọc trong ```json ... ```
                clean_json_str = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.MULTILINE)
                clean_json_str = re.sub(r"\s*```$", "", clean_json_str, flags=re.MULTILINE).strip()

                try:
                    data = json.loads(clean_json_str)
                except Exception as json_err:
                    last_error = f"Lỗi parse JSON: {json_err} | Text: {clean_json_str[:200]}"
                    continue

                # Chuẩn hóa Verdict Type
                v_type_raw = str(data.get("verdict_type", "")).strip().upper()
                v_type: VerdictType
                if "PROCESSED" in v_type_raw:
                    v_type = VerdictType.PROCESSED_HUMAN
                elif "CONFIRMED" in v_type_raw or "AUTHENTIC" in v_type_raw or v_type_raw == "HUMAN":
                    v_type = VerdictType.CONFIRMED_HUMAN
                elif "HYBRID" in v_type_raw or "ASSISTED" in v_type_raw:
                    v_type = VerdictType.HYBRID_AI
                elif "AI" in v_type_raw or "GENERATED" in v_type_raw or "SYNTHETIC" in v_type_raw:
                    v_type = VerdictType.AI_GENERATED
                elif "UNCERTAIN" in v_type_raw:
                    v_type = VerdictType.UNCERTAIN
                else:
                    try:
                        v_type = VerdictType(v_type_raw)
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

                # Chuẩn hóa Work Type
                w_type_str = str(data.get("work_type", "UNCERTAIN")).strip().upper()
                try:
                    w_type = WorkType(w_type_str)
                except ValueError:
                    if "ORIGINAL" in w_type_str:
                        w_type = WorkType.ORIGINAL
                    elif "COVER" in w_type_str:
                        w_type = WorkType.COVER
                    elif "REMIX" in w_type_str:
                        w_type = WorkType.REMIX
                    else:
                        w_type = WorkType.COVER if data.get("is_cover") else WorkType.UNCERTAIN

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

                logger.info(
                    f"[LocalLLMClassifier] [SUCCESS] Giám định thành công bằng Local Model ({model})!"
                )
                return DetectionResult(
                    track_id=track_id,
                    track_title=track_title,
                    source_url=source_url,
                    input_type=input_type,
                    raw_query=raw_query,
                    is_ai=final_is_ai,
                    verdict_type=v_type,
                    verdict_label=verdict_label,
                    confidence_score=float(data.get("confidence_score", 0.5) or 0.5),
                    suspected_engine=str(data.get("suspected_engine") or "None"),
                    reasons=data.get("reasons", []) if isinstance(data.get("reasons"), list) else [str(data.get("reasons"))],
                    detailed_explanation=str(data.get("detailed_explanation", "")),
                    acoustic_metrics=metrics,
                    spectrogram_image_path=str(spectrogram_path) if spectrogram_path else None,
                    model_used=model,
                    work_type=w_type,
                    work_label=work_label,
                    is_cover=bool(data.get("is_cover", False) or w_type == WorkType.COVER),
                    original_artist=orig_artist,
                    original_song_title=orig_title,
                    cover_performer=cov_perf,
                )

            except Exception as req_err:
                last_error = str(req_err)
                logger.warning(
                    f"[LocalLLMClassifier] Lỗi khi kết nối model {model}: {req_err}. Thử fallback tiếp theo..."
                )
                continue

        raise ClassificationException(
            f"Tất cả model trên Local LLM ({self.base_url}) đều thất bại. Lỗi cuối cùng: {last_error}"
        )

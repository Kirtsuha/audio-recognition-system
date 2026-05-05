from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.audio import crop_segment, load_audio_from_bytes, pad_or_trim, validate_audio_key
from app.config import settings
from app.model import AudioEncoder
from app.reference_store import ReferenceEmbeddingStore
from app.s3_storage import S3Storage
from app.schemas import (
    FingerprintCandidate,
    FingerprintRecognitionPayload,
    RerankedCandidate,
    RerankOptions,
    RerankResponse,
)
from app.to_mel import to_mel_batch

logger = logging.getLogger("ml-reranker.engine")


@dataclass(frozen=True)
class RuntimeStatus:
    model_loaded: bool
    device: str
    model_path: str
    reference_store_loaded: bool
    reference_store_vectors: int
    reference_store_tracks: int


class RerankerEngine:
    def __init__(self, storage: S3Storage):
        self.storage = storage
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: AudioEncoder | None = None
        self.reference_store: ReferenceEmbeddingStore | None = None

        self.load_model()
        self.load_reference_store()

    def load_model(self) -> None:
        model_path = Path(settings.model_path)

        if not model_path.exists():
            logger.warning("Model checkpoint does not exist: %s", model_path)
            self.model = None
            return

        model = AudioEncoder(emb_dim=settings.emb_dim).to(self.device)
        state = torch.load(model_path, map_location=self.device)

        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]

        model.load_state_dict(state)
        model.eval()
        self.model = model

        logger.info("Loaded AudioEncoder model path=%s device=%s", model_path, self.device)

    def load_reference_store(self) -> None:
        if not settings.use_precomputed_reference_embeddings:
            logger.info("Precomputed reference embeddings disabled")
            self.reference_store = None
            return

        store = ReferenceEmbeddingStore(
            embeddings_path=settings.ref_embeddings_path,
            meta_path=settings.ref_meta_path,
            config_path=settings.ref_config_path,
        )

        try:
            store.load()
            self.reference_store = store
        except FileNotFoundError:
            logger.warning(
                "Reference embedding store missing embeddings=%s meta=%s config=%s; slow fallback will be used",
                settings.ref_embeddings_path,
                settings.ref_meta_path,
                settings.ref_config_path,
            )
            self.reference_store = None
        except Exception:
            logger.exception("Failed to load reference embedding store; slow fallback will be used")
            self.reference_store = None

    def status(self) -> RuntimeStatus:
        ref_status = self.reference_store.status() if self.reference_store is not None else {}

        return RuntimeStatus(
            model_loaded=self.model is not None,
            device=str(self.device),
            model_path=settings.model_path,
            reference_store_loaded=bool(ref_status.get("loaded", False)),
            reference_store_vectors=int(ref_status.get("vectors", 0) or 0),
            reference_store_tracks=int(ref_status.get("tracks", 0) or 0),
        )

    def ensure_ready(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"ML reranker model is not loaded. Expected checkpoint at {settings.model_path}"
            )

    def _parse_default_jitter(self) -> list[float]:
        out: list[float] = []
        for raw in settings.offset_jitter_seconds.split(","):
            raw = raw.strip()
            if not raw:
                continue
            out.append(float(raw))
        return out or [0.0]

    def _option_max_candidates(self, options: RerankOptions) -> int:
        return settings.max_candidates if options.max_candidates is None else int(options.max_candidates)

    def _option_segment_seconds(self, options: RerankOptions) -> float:
        return settings.segment_seconds if options.segment_seconds is None else float(options.segment_seconds)

    def _option_hop_seconds(self, options: RerankOptions) -> float:
        if options.fingerprint_offset_hop_seconds is not None:
            return float(options.fingerprint_offset_hop_seconds)
        return settings.fingerprint_offset_hop_seconds

    def _option_jitter(self, options: RerankOptions) -> list[float]:
        if options.offset_jitter_seconds is not None:
            return [float(x) for x in options.offset_jitter_seconds]
        return self._parse_default_jitter()

    def _option_weights(self, options: RerankOptions) -> tuple[float, float]:
        fp_weight = settings.fp_weight if options.fp_weight is None else float(options.fp_weight)
        ml_weight = settings.ml_weight if options.ml_weight is None else float(options.ml_weight)

        total = fp_weight + ml_weight
        if total <= 0:
            return 1.0, 0.0

        return fp_weight / total, ml_weight / total

    def _option_threshold(self, options: RerankOptions) -> float:
        if options.no_match_threshold is not None:
            return float(options.no_match_threshold)
        return settings.no_match_threshold

    def _candidate_offset_sec(self, candidate: FingerprintCandidate, hop_seconds: float) -> float:
        if candidate.best_offset_sec is not None:
            return max(float(candidate.best_offset_sec), 0.0)
        return max(float(candidate.best_offset) * float(hop_seconds), 0.0)

    def _fingerprint_score(self, candidate: FingerprintCandidate) -> float:
        return float(candidate.aligned_matches)

    def _similarity_to_probability(self, similarity: float) -> float:
        return max(0.0, min(1.0, (float(similarity) + 1.0) / 2.0))

    def _final_confidence(
        self,
        fingerprint_confidence: float,
        ml_probability: float,
        options: RerankOptions,
    ) -> float:
        fp_weight, ml_weight = self._option_weights(options)
        fp = max(0.0, min(1.0, float(fingerprint_confidence)))
        ml = max(0.0, min(1.0, float(ml_probability)))
        return float(fp_weight * fp + ml_weight * ml)

    def _load_audio_from_s3(self, bucket: str, key: str) -> np.ndarray:
        validate_audio_key(key)
        data = self.storage.get_bytes(bucket, key)
        return load_audio_from_bytes(data, sample_rate=settings.sr)

    @torch.inference_mode()
    def _embed_batch(self, segments: list[np.ndarray]) -> np.ndarray:
        self.ensure_ready()
        assert self.model is not None

        if not segments:
            return np.zeros((0, settings.emb_dim), dtype="float32")

        audio_batch = np.stack(segments).astype("float32")
        mel_batch = to_mel_batch(audio_batch, device=self.device, normalize=True)
        embeddings = self.model(mel_batch).detach().cpu().numpy().astype("float32")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        return (embeddings / norms).astype("float32")

    def debug_fetch_audio(self, bucket: str, key: str) -> dict:
        validate_audio_key(key)
        data = self.storage.get_bytes(bucket, key)
        audio = load_audio_from_bytes(data, sample_rate=settings.sr)
        return {
            "bucket": bucket,
            "key": key,
            "bytes": len(data),
            "duration_sec": len(audio) / float(settings.sr),
            "sample_rate": settings.sr,
        }

    def _score_candidate_precomputed(
        self,
        *,
        candidate: FingerprintCandidate,
        query_embedding: np.ndarray,
        offset_sec: float,
    ) -> tuple[float, float] | None:
        if self.reference_store is None or not self.reference_store.loaded():
            return None

        try:
            track_id = int(candidate.track_id)
        except Exception:
            return None

        result = self.reference_store.get_nearby_embeddings(
            track_id=track_id,
            offset_sec=offset_sec,
            neighbor_windows=settings.ref_neighbor_windows,
        )

        if result is None:
            return None

        ref_embeddings, windows = result

        if len(ref_embeddings) == 0:
            return None

        similarities = ref_embeddings @ query_embedding.astype("float32")
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_offset_sec = float(windows[best_idx].start_sec)

        return best_similarity, best_offset_sec

    def _score_candidates_slow_audio_fallback(
        self,
        *,
        reference_bucket: str,
        candidates: list[FingerprintCandidate],
        candidate_indices: list[int],
        query_segment: np.ndarray,
        hop_seconds: float,
        jitter_values: list[float],
        segment_seconds: float,
    ) -> dict[int, tuple[float, float]]:
        batch_query: list[np.ndarray] = []
        batch_ref: list[np.ndarray] = []
        batch_candidate_idx: list[int] = []
        batch_offset_sec: list[float] = []

        ref_cache: dict[str, np.ndarray] = {}

        for idx in candidate_indices:
            candidate = candidates[idx]

            try:
                ref_audio = ref_cache.get(candidate.s3_key)
                if ref_audio is None:
                    ref_audio = self._load_audio_from_s3(reference_bucket, candidate.s3_key)
                    ref_cache[candidate.s3_key] = ref_audio

                base_offset_sec = self._candidate_offset_sec(candidate, hop_seconds)

                for jitter in jitter_values:
                    offset_sec = max(base_offset_sec + float(jitter), 0.0)
                    ref_segment = crop_segment(
                        ref_audio,
                        start_sec=offset_sec,
                        segment_seconds=segment_seconds,
                    )

                    batch_query.append(query_segment)
                    batch_ref.append(ref_segment)
                    batch_candidate_idx.append(idx)
                    batch_offset_sec.append(offset_sec)

            except Exception:
                logger.exception(
                    "Slow fallback failed to prepare candidate track_id=%s s3_key=%s",
                    candidate.track_id,
                    candidate.s3_key,
                )

        if not batch_ref:
            return {}

        query_embeddings = self._embed_batch(batch_query)
        ref_embeddings = self._embed_batch(batch_ref)
        similarities = np.sum(query_embeddings * ref_embeddings, axis=1)

        best_by_candidate: dict[int, tuple[float, float]] = {}

        for sim, idx, offset_sec in zip(similarities, batch_candidate_idx, batch_offset_sec):
            previous = best_by_candidate.get(idx)
            if previous is None or float(sim) > previous[0]:
                best_by_candidate[idx] = (float(sim), float(offset_sec))

        return best_by_candidate

    def rerank(
        self,
        *,
        request_id: str,
        query_bucket: str,
        query_key: str,
        reference_bucket: str | None,
        fingerprint: FingerprintRecognitionPayload,
        options: RerankOptions,
    ) -> RerankResponse:
        self.ensure_ready()

        started_total = time.time()
        started_query = time.time()

        query_audio = self._load_audio_from_s3(query_bucket, query_key)

        timing_download_query = int((time.time() - started_query) * 1000)
        started_rerank = time.time()

        reference_bucket = reference_bucket or settings.reference_audio_bucket
        max_candidates = self._option_max_candidates(options)
        segment_seconds = self._option_segment_seconds(options)
        hop_seconds = self._option_hop_seconds(options)
        jitter_values = self._option_jitter(options)
        threshold = self._option_threshold(options)

        candidates = fingerprint.candidates[:max_candidates]

        if not candidates:
            return RerankResponse(
                request_id=request_id,
                matched=False,
                reason="no_fingerprint_candidates",
                best=None,
                candidates=[],
                timing_ms={
                    "total": int((time.time() - started_total) * 1000),
                    "download_query": timing_download_query,
                    "rerank": 0,
                },
            )

        target_len = int(round(segment_seconds * settings.sr))
        query_segment = pad_or_trim(query_audio, target_len=target_len)

        # Fast path: query is embedded once.
        query_embedding = self._embed_batch([query_segment])[0]

        best_by_candidate: dict[int, tuple[float, float]] = {}
        fallback_indices: list[int] = []

        for idx, candidate in enumerate(candidates):
            base_offset_sec = self._candidate_offset_sec(candidate, hop_seconds)

            best_precomputed: tuple[float, float] | None = None

            for jitter in jitter_values:
                offset_sec = max(base_offset_sec + float(jitter), 0.0)

                scored = self._score_candidate_precomputed(
                    candidate=candidate,
                    query_embedding=query_embedding,
                    offset_sec=offset_sec,
                )

                if scored is None:
                    continue

                if best_precomputed is None or scored[0] > best_precomputed[0]:
                    best_precomputed = scored

            if best_precomputed is not None:
                best_by_candidate[idx] = best_precomputed
            else:
                fallback_indices.append(idx)

        if fallback_indices and settings.ref_fallback_to_audio:
            fallback_scores = self._score_candidates_slow_audio_fallback(
                reference_bucket=reference_bucket,
                candidates=candidates,
                candidate_indices=fallback_indices,
                query_segment=query_segment,
                hop_seconds=hop_seconds,
                jitter_values=jitter_values,
                segment_seconds=segment_seconds,
            )
            best_by_candidate.update(fallback_scores)

        reranked: list[RerankedCandidate] = []

        for idx, candidate in enumerate(candidates):
            default_offset_sec = self._candidate_offset_sec(candidate, hop_seconds)

            ml_similarity, used_offset_sec = best_by_candidate.get(
                idx,
                (-1.0, default_offset_sec),
            )

            ml_probability = self._similarity_to_probability(ml_similarity)
            fp_conf = max(0.0, min(1.0, float(candidate.confidence)))
            final_conf = self._final_confidence(fp_conf, ml_probability, options)

            reranked.append(
                RerankedCandidate(
                    track_id=candidate.track_id,
                    title=candidate.title,
                    artist=candidate.artist,
                    s3_key=candidate.s3_key,
                    best_offset=candidate.best_offset,
                    best_offset_sec=float(used_offset_sec),
                    fingerprint_confidence=fp_conf,
                    fingerprint_score=self._fingerprint_score(candidate),
                    ml_similarity=float(ml_similarity),
                    ml_probability=float(ml_probability),
                    final_confidence=float(final_conf),
                    aligned_matches=int(candidate.aligned_matches),
                    total_matches=int(candidate.total_matches),
                    offset_count=int(candidate.offset_count),
                    coverage=float(candidate.coverage),
                    unique_coverage=float(candidate.unique_coverage),
                    score_gap=float(candidate.score_gap),
                )
            )

        reranked.sort(key=lambda item: item.final_confidence, reverse=True)

        best = reranked[0] if reranked else None
        matched = bool(best is not None and best.final_confidence >= threshold)
        reason = None if matched else "below_no_match_threshold"

        timing = {
            "total": int((time.time() - started_total) * 1000),
            "download_query": timing_download_query,
            "rerank": int((time.time() - started_rerank) * 1000),
        }

        logger.info(
            (
                "Rerank finished request_id=%s matched=%s best_track_id=%s "
                "final=%.4f candidates=%s fallback_candidates=%s ref_store=%s timing=%s"
            ),
            request_id,
            matched,
            best.track_id if best else None,
            best.final_confidence if best else 0.0,
            len(reranked),
            len(fallback_indices),
            self.reference_store.loaded() if self.reference_store else False,
            timing,
        )

        return RerankResponse(
            request_id=request_id,
            matched=matched,
            reason=reason,
            best=best,
            candidates=reranked,
            timing_ms=timing,
        )
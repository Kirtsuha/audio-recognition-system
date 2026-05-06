package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.hse.musicrecognition.config.RecognitionProperties;
import org.hse.musicrecognition.dto.*;
import org.hse.musicrecognition.exception.BadRequestException;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.UUID;

@Slf4j
@Service
@RequiredArgsConstructor
public class RecognitionService {

    private final FingerprintClient fingerprintClient;
    private final MlRerankerClient mlRerankerClient;
    private final QueryAudioStorageService queryAudioStorageService;
    private final HistoryService historyService;
    private final RecognitionProperties recognitionProperties;

    public RecognitionResponse recognize(
            String username,
            String filename,
            String contentType,
            String source,
            byte[] audioBytes
    ) {
        if (audioBytes == null || audioBytes.length == 0) {
            throw new BadRequestException("Audio file is empty");
        }

        String requestId = UUID.randomUUID().toString().replace("-", "");
        long startedAt = System.currentTimeMillis();

        StoredAudioRef storedAudio = queryAudioStorageService.save(
                requestId,
                filename,
                contentType,
                audioBytes
        );

        FingerprintCandidatesResponse fingerprint = fingerprintClient.retrieveCandidates(
                audioBytes,
                filename,
                recognitionProperties.getFingerprint().getTopK()
        );

        RecognitionResponse response;

        if (fingerprint != null && fingerprint.isMatched()) {
            response = buildFingerprintResponse(fingerprint);
        } else {
            response = recognizeWithReranker(requestId, storedAudio, fingerprint);
        }

        long processingTimeMs = System.currentTimeMillis() - startedAt;

        log.info(
                "Recognition finished requestId={} username={} match={} trackId={} source={} confidence={} processingTimeMs={}",
                requestId,
                username,
                response.isMatch(),
                response.getTrackId(),
                response.getSource(),
                response.getConfidence(),
                processingTimeMs
        );

        historyService.save(
                username,
                filename,
                response.getTrackId(),
                response.getTitle(),
                response.getArtist(),
                response.getConfidence(),
                response.getSource()
        );

        return response;
    }

    private RecognitionResponse buildFingerprintResponse(FingerprintCandidatesResponse fingerprint) {
        FingerprintCandidateDto best = fingerprint.bestCandidate();

        if (best == null) {
            return new RecognitionResponse(
                    false,
                    null,
                    null,
                    null,
                    fingerprint.confidence() == null ? 0.0 : fingerprint.confidence(),
                    "fingerprint-not-found"
            );
        }

        return new RecognitionResponse(
                true,
                best.trackId() == null ? null : String.valueOf(best.trackId()),
                best.title(),
                best.artist(),
                best.confidence() == null
                        ? fingerprint.confidence() == null ? 0.0 : fingerprint.confidence()
                        : best.confidence(),
                "fingerprint"
        );
    }

    private RecognitionResponse recognizeWithReranker(
            String requestId,
            StoredAudioRef storedAudio,
            FingerprintCandidatesResponse fingerprint
    ) {
        MlRerankRequest request = new MlRerankRequest(
                requestId,
                new QueryAudioRef(storedAudio.bucket(), storedAudio.key()),
                recognitionProperties.getStorage().getReferenceAudioBucket(),
                fingerprint,
                Map.of()
        );

        log.info(
                "Calling ML reranker requestId={} queryAudio={}/{} referenceBucket={} fingerprintMatched={} candidates={}",
                requestId,
                storedAudio.bucket(),
                storedAudio.key(),
                recognitionProperties.getStorage().getReferenceAudioBucket(),
                fingerprint == null ? null : fingerprint.matched(),
                fingerprint == null || fingerprint.candidates() == null ? 0 : fingerprint.candidates().size()
        );

        MlRerankResponse rerank;

        try {
            rerank = mlRerankerClient.rerank(request);
        } catch (ExternalServiceUnavailableException exc) {
            log.warn(
                    "ML reranker failed; returning fingerprint not-found fallback requestId={} queryAudio={}/{}",
                    requestId,
                    storedAudio.bucket(),
                    storedAudio.key(),
                    exc
            );

            return new RecognitionResponse(
                    false,
                    null,
                    null,
                    null,
                    resolveRerankNotFoundConfidence(null, fingerprint),
                    "ml-reranker-unavailable"
            );
        }

        if (rerank == null || !rerank.isMatched() || rerank.best() == null) {
            double confidence = resolveRerankNotFoundConfidence(rerank, fingerprint);

            return new RecognitionResponse(
                    false,
                    null,
                    null,
                    null,
                    confidence,
                    "ml-reranker-not-found"
            );
        }

        RerankedCandidateDto best = rerank.best();

        return new RecognitionResponse(
                true,
                best.trackId() == null ? null : String.valueOf(best.trackId()),
                best.title(),
                best.artist(),
                best.finalConfidence() == null ? 0.0 : best.finalConfidence(),
                "ml-reranker"
        );
    }

    private double resolveRerankNotFoundConfidence(
            MlRerankResponse rerank,
            FingerprintCandidatesResponse fingerprint
    ) {
        if (rerank != null && rerank.best() != null && rerank.best().finalConfidence() != null) {
            return rerank.best().finalConfidence();
        }

        if (fingerprint != null && fingerprint.confidence() != null) {
            return fingerprint.confidence();
        }

        return 0.0;
    }
}

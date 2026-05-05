package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.FingerprintResponse;
import org.hse.musicrecognition.dto.MlRecognitionResponse;
import org.hse.musicrecognition.dto.RecognitionResponse;
import org.hse.musicrecognition.dto.TrackMetadataResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
public class RecognitionService {

    private final FingerprintClient fingerprintClient;
    private final MlFallbackGateway mlFallbackGateway;
    private final HistoryService historyService;

    @Value("${fingerprint.confidence-threshold}")
    private double fingerprintConfidenceThreshold;

    @Value("${ml.confidence-threshold:0.70}")
    private double mlConfidenceThreshold;

    public RecognitionResponse recognize(String username, String filename, byte[] audioBytes) throws Exception {
        FingerprintResponse fingerprintResponse = fingerprintClient.recognize(audioBytes);

        RecognitionResponse response;

        if (shouldUseMlFallback(fingerprintResponse)) {
            response = recognizeWithMlFallback(audioBytes, filename);
        } else {
            response = buildFingerprintResponse(fingerprintResponse);
        }

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

    private RecognitionResponse buildFingerprintResponse(FingerprintResponse fingerprintResponse) {
        return new RecognitionResponse(
                fingerprintResponse.isMatch(),
                fingerprintResponse.getTrackId().toString(),
                fingerprintResponse.getTitle(),
                fingerprintResponse.getArtist(),
                fingerprintResponse.getConfidence(),
                "fingerprint"
        );
    }

    private RecognitionResponse recognizeWithMlFallback(byte[] audioBytes, String filename) throws Exception {
        MlRecognitionResponse mlResponse = mlFallbackGateway.recognize(audioBytes, filename);

        double confidence = mlResponse.confidence() == null ? 0.0 : mlResponse.confidence();

        boolean mlMatched = Boolean.TRUE.equals(mlResponse.matched());

        if (!mlMatched) {
            return new RecognitionResponse(
                    false,
                    mlResponse.trackId() == null ? null : String.valueOf(mlResponse.trackId()),
                    null,
                    null,
                    confidence,
                    "ml-fallback-not-found"
            );
        }

        if (mlResponse.trackId() == null) {
            return new RecognitionResponse(
                    false,
                    null,
                    null,
                    null,
                    confidence,
                    "ml-fallback-not-found"
            );
        }

        if (confidence < mlConfidenceThreshold) {
            return new RecognitionResponse(
                    false,
                    String.valueOf(mlResponse.trackId()),
                    null,
                    null,
                    confidence,
                    "ml-fallback-low-confidence"
            );
        }

        TrackMetadataResponse metadata = fingerprintClient.getTrack(mlResponse.trackId());

        return new RecognitionResponse(
                true,
                String.valueOf(metadata.trackId()),
                metadata.title(),
                metadata.artist(),
                confidence,
                "ml-fallback"
        );
    }

    private boolean shouldUseMlFallback(FingerprintResponse fingerprintResponse) {
        return fingerprintResponse == null
                || !fingerprintResponse.isMatch()
                || fingerprintResponse.getConfidence() < fingerprintConfidenceThreshold;
    }
}
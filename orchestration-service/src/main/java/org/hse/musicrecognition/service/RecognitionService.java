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

    public RecognitionResponse recognize(String username, String filename, byte[] audioBytes) throws Exception {
        FingerprintResponse fingerprintResponse = fingerprintClient.recognize(audioBytes);

        RecognitionResponse response;
        if (shouldUseMlFallback(fingerprintResponse)) {
            MlRecognitionResponse mlResponse = mlFallbackGateway.recognize(audioBytes, filename);
            if (mlResponse.trackId() == null) {
                throw new IllegalStateException("ML fallback returned empty track id");
            }
            TrackMetadataResponse metadata = fingerprintClient.getTrack(mlResponse.trackId());

            response = new RecognitionResponse(
                    true,
                    String.valueOf(metadata.trackId()),
                    metadata.title(),
                    metadata.artist(),
                    mlResponse.confidence() == null ? 0.0 : mlResponse.confidence(),
                    "ml-fallback"
            );
        } else {
            response = new RecognitionResponse(
                    fingerprintResponse.isMatch(),
                    fingerprintResponse.getTrackId(),
                    fingerprintResponse.getTitle(),
                    fingerprintResponse.getArtist(),
                    fingerprintResponse.getConfidence(),
                    "fingerprint"
            );
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

    private boolean shouldUseMlFallback(FingerprintResponse fingerprintResponse) {
        return !fingerprintResponse.isMatch()
                || fingerprintResponse.getConfidence() < fingerprintConfidenceThreshold;
    }
}

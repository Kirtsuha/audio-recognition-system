package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.FingerprintCandidatesResponse;
import org.hse.musicrecognition.dto.TrackMetadataResponse;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

@Service
@RequiredArgsConstructor
public class FingerprintClient {

    private final RestClient restClient;

    @Value("${fingerprint.service.url}")
    private String fingerprintUrl;

    public FingerprintCandidatesResponse retrieveCandidates(
            byte[] audioBytes,
            String filename,
            int topK
    ) {
        ByteArrayResource resource = new ByteArrayResource(audioBytes) {
            @Override
            public String getFilename() {
                return filename == null || filename.isBlank()
                        ? "audio.wav"
                        : filename;
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);

        try {
            return restClient.post()
                    .uri(fingerprintUrl + "/retrieve-candidates?top_k={topK}", topK)
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(body)
                    .retrieve()
                    .body(FingerprintCandidatesResponse.class);
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("Fingerprint service is unavailable", e);
        }
    }

    public TrackMetadataResponse getTrack(Integer trackId) {
        try {
            return restClient.get()
                    .uri(fingerprintUrl + "/tracks/{trackId}", trackId)
                    .retrieve()
                    .body(TrackMetadataResponse.class);
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("Fingerprint service is unavailable", e);
        }
    }
}
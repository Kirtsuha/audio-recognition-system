package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.FingerprintCandidatesResponse;
import org.hse.musicrecognition.dto.FingerprintTrackUploadResponse;
import org.hse.musicrecognition.dto.TrackMetadataResponse;
import org.hse.musicrecognition.dto.TrackSearchResponse;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.ResponseEntity;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;
import org.springframework.web.multipart.MultipartFile;

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

    public TrackSearchResponse searchTracksByTitle(String title, int limit) {
        try {
            return restClient.get()
                    .uri(
                            fingerprintUrl + "/tracks/search/by-title?title={title}&limit={limit}",
                            title,
                            limit
                    )
                    .retrieve()
                    .body(TrackSearchResponse.class);
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("Fingerprint service is unavailable", e);
        }
    }

    public ResponseEntity<byte[]> getTrackAudio(Integer trackId) {
        try {
            return restClient.get()
                    .uri(fingerprintUrl + "/tracks/{trackId}/audio", trackId)
                    .retrieve()
                    .toEntity(byte[].class);
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("Fingerprint service is unavailable", e);
        }
    }

    public FingerprintTrackUploadResponse uploadTrack(
            MultipartFile file,
            String title,
            String artist,
            String album
    ) {
        try {
            ByteArrayResource resource = new ByteArrayResource(file.getBytes()) {
                @Override
                public String getFilename() {
                    return file.getOriginalFilename() == null || file.getOriginalFilename().isBlank()
                            ? "track.mp3"
                            : file.getOriginalFilename();
                }
            };

            MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
            body.add("file", resource);
            body.add("title", title);
            body.add("artist", artist);
            if (album != null && !album.isBlank()) {
                body.add("album", album);
            }

            return restClient.post()
                    .uri(fingerprintUrl + "/s3/upload-track")
                    .contentType(MediaType.MULTIPART_FORM_DATA)
                    .body(body)
                    .retrieve()
                    .body(FingerprintTrackUploadResponse.class);
        } catch (Exception e) {
            throw new ExternalServiceUnavailableException("Fingerprint service is unavailable", e);
        }
    }

    public void indexS3(String bucket, String prefix) {
        try {
            restClient.post()
                    .uri(
                            fingerprintUrl + "/index/s3?s3_bucket={bucket}&s3_prefix={prefix}",
                            bucket,
                            prefix == null ? "" : prefix
                    )
                    .retrieve()
                    .toBodilessEntity();
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("Fingerprint service is unavailable", e);
        }
    }
}

package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.TrackMetadataResponse;
import org.hse.musicrecognition.dto.TrackSearchResponse;
import org.hse.musicrecognition.service.FingerprintClient;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Set;

@RestController
@RequestMapping("/api/tracks")
@RequiredArgsConstructor
public class TrackController {

    private final FingerprintClient fingerprintClient;

    @GetMapping("/{trackId}")
    public TrackMetadataResponse getTrack(@PathVariable Integer trackId) {
        return fingerprintClient.getTrack(trackId);
    }

    @GetMapping("/search")
    public TrackSearchResponse search(
            @RequestParam(value = "query", required = false) String query,
            @RequestParam(value = "title", required = false) String title,
            @RequestParam(value = "artist", required = false) String artist,
            @RequestParam(value = "limit", required = false, defaultValue = "20") int limit
    ) {
        String effectiveQuery = query == null || query.isBlank() ? title : query;
        return fingerprintClient.searchTracks(effectiveQuery, artist, limit);
    }

    @GetMapping("/{trackId}/audio")
    public ResponseEntity<byte[]> getTrackAudio(@PathVariable Integer trackId) {
        ResponseEntity<byte[]> response = fingerprintClient.getTrackAudio(trackId);
        byte[] body = response.getBody();

        HttpHeaders headers = sanitizeAudioHeaders(response.getHeaders());
        if (body != null) {
            headers.setContentLength(body.length);
        }

        return ResponseEntity
                .status(response.getStatusCode())
                .headers(headers)
                .body(body);
    }

    private HttpHeaders sanitizeAudioHeaders(HttpHeaders source) {
        Set<String> hopByHopHeaders = Set.of(
                HttpHeaders.CONNECTION,
                HttpHeaders.TRANSFER_ENCODING,
                "Keep-Alive",
                HttpHeaders.PROXY_AUTHENTICATE,
                HttpHeaders.PROXY_AUTHORIZATION,
                "TE",
                HttpHeaders.TRAILER,
                HttpHeaders.UPGRADE
        );

        HttpHeaders headers = new HttpHeaders();

        MediaType contentType = source.getContentType();
        headers.setContentType(contentType == null ? MediaType.APPLICATION_OCTET_STREAM : contentType);

        if (source.getContentDisposition() != null) {
            headers.setContentDisposition(source.getContentDisposition());
        }

        source.forEach((name, values) -> {
            if (
                    hopByHopHeaders.stream().noneMatch(header -> header.equalsIgnoreCase(name))
                            && !HttpHeaders.CONTENT_TYPE.equalsIgnoreCase(name)
                            && !HttpHeaders.CONTENT_LENGTH.equalsIgnoreCase(name)
                            && !HttpHeaders.CONTENT_DISPOSITION.equalsIgnoreCase(name)
            ) {
                headers.put(name, values);
            }
        });

        return headers;
    }
}

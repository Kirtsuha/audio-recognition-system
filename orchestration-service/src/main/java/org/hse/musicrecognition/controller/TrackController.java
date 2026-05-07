package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.TrackMetadataResponse;
import org.hse.musicrecognition.dto.TrackSearchResponse;
import org.hse.musicrecognition.service.FingerprintClient;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

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

        HttpHeaders headers = new HttpHeaders();
        headers.putAll(response.getHeaders());

        return ResponseEntity
                .status(response.getStatusCode())
                .headers(headers)
                .body(response.getBody());
    }
}

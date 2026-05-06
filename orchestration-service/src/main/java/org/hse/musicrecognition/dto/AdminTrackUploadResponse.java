package org.hse.musicrecognition.dto;

public record AdminTrackUploadResponse(
        Integer trackId,
        String title,
        String artist,
        String album,
        String status
) {
}

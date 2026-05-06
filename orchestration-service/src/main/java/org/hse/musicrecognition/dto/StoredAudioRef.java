package org.hse.musicrecognition.dto;

public record StoredAudioRef(
        String bucket,
        String key
) {
}
package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

public record TrackMetadataResponse(
        @JsonProperty("track_id")
        Integer trackId,
        String title,
        String artist,
        String album,
        @JsonProperty("s3_key")
        String s3Key
) {
}

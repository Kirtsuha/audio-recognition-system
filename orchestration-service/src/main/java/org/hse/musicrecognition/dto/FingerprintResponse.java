package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

@Data
public class FingerprintResponse {
    private boolean match;
    @JsonProperty("track_id")
    private String trackId;
    private String title;
    private String artist;
    @JsonProperty("s3_key")
    private String s3Key;
    private double confidence;
}


package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Data;

@Data
@JsonIgnoreProperties(ignoreUnknown = true)
public class FingerprintResponse {

    private boolean match;

    @JsonAlias({"track_id", "trackId"})
    private Integer trackId;

    private String title;

    private String artist;

    @JsonAlias({"s3_key", "s3Key"})
    private String s3Key;

    private double confidence;

    private String source;
}
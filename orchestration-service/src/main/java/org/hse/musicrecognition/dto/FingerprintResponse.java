package org.hse.musicrecognition.dto;

import lombok.Data;
import lombok.Getter;

@Data
public class FingerprintResponse {
    private boolean match;
    private String trackId;
    private String title;
    private String artist;
    private double confidence;
}


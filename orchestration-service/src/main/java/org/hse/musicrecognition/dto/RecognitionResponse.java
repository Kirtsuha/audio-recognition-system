package org.hse.musicrecognition.dto;
import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class RecognitionResponse {
    private boolean match;
    private String trackId;
    private String title;
    private String artist;
    private double confidence;
    private String source;
}


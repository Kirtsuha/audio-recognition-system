package org.hse.musicrecognition.dto;

public record MlRecognitionResponse(
        String requestId,
        Integer trackId,
        Double confidence,
        String error
) {
}

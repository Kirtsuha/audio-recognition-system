package org.hse.musicrecognition.dto;

public record MlRecognitionRequest(
        String requestId,
        String filename,
        String audioBase64
) {
}

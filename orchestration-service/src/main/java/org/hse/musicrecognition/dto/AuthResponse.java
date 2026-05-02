package org.hse.musicrecognition.dto;

public record AuthResponse(
        String accessToken,
        String tokenType
) {
}
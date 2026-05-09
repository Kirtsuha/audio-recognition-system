package org.hse.musicrecognition.dto;

public record UserResponse(
        Long id,
        String username,
        String role
) {
}
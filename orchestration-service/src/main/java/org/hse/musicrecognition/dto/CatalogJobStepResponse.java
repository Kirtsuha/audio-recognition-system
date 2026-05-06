package org.hse.musicrecognition.dto;

public record CatalogJobStepResponse(
        String name,
        String status,
        String errorMessage
) {
}

package org.hse.musicrecognition.dto;

import java.util.UUID;

public record CatalogJobStartedResponse(
        UUID jobId,
        String status
) {
}

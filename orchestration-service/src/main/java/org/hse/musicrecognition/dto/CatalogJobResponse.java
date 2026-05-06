package org.hse.musicrecognition.dto;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

public record CatalogJobResponse(
        UUID jobId,
        String type,
        String status,
        Instant startedAt,
        Instant finishedAt,
        String errorMessage,
        List<CatalogJobStepResponse> steps
) {
}

package org.hse.musicrecognition.dto;

import java.util.List;
import java.util.UUID;

public record CatalogSyncResponse(
        UUID jobId,
        String status,
        List<CatalogJobStepResponse> steps
) {
}

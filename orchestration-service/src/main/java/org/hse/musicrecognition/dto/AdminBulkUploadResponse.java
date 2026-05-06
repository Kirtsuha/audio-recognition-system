package org.hse.musicrecognition.dto;

import java.util.UUID;

public record AdminBulkUploadResponse(
        UUID jobId,
        String status
) {
}

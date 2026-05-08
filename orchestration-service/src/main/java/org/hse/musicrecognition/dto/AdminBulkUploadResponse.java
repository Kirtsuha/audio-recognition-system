package org.hse.musicrecognition.dto;

import java.util.Map;

public record AdminBulkUploadResponse(
        String status,
        Map<String, Object> summary
) {
}

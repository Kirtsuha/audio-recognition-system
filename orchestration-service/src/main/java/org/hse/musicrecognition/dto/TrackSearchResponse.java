package org.hse.musicrecognition.dto;

import java.util.List;

public record TrackSearchResponse(
        List<TrackMetadataResponse> items,
        Integer count
) {
}

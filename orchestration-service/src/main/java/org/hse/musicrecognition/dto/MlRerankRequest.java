package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Map;

public record MlRerankRequest(

        @JsonProperty("request_id")
        String requestId,

        @JsonProperty("query_audio")
        QueryAudioRef queryAudio,

        @JsonProperty("reference_bucket")
        String referenceBucket,

        FingerprintCandidatesResponse fingerprint,

        Map<String, Object> options
) {
}
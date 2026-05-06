package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public record FingerprintOffsetDto(

        Integer offset,

        @JsonProperty("aligned_matches")
        @JsonAlias({"aligned_matches", "alignedMatches"})
        Integer alignedMatches,

        @JsonProperty("offset_sec")
        @JsonAlias({"offset_sec", "offsetSec"})
        Double offsetSec
) {
}
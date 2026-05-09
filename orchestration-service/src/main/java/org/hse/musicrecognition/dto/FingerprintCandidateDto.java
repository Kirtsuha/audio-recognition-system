package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record FingerprintCandidateDto(

        @JsonProperty("track_id")
        @JsonAlias({"track_id", "trackId"})
        Integer trackId,

        @JsonProperty("best_offset")
        @JsonAlias({"best_offset", "bestOffset"})
        Integer bestOffset,

        @JsonProperty("best_offset_sec")
        @JsonAlias({"best_offset_sec", "bestOffsetSec"})
        Double bestOffsetSec,

        @JsonProperty("aligned_matches")
        @JsonAlias({"aligned_matches", "alignedMatches"})
        Integer alignedMatches,

        @JsonProperty("total_matches")
        @JsonAlias({"total_matches", "totalMatches"})
        Integer totalMatches,

        @JsonProperty("offset_count")
        @JsonAlias({"offset_count", "offsetCount"})
        Integer offsetCount,

        Double coverage,

        @JsonProperty("unique_coverage")
        @JsonAlias({"unique_coverage", "uniqueCoverage"})
        Double uniqueCoverage,

        @JsonProperty("top_offsets")
        @JsonAlias({"top_offsets", "topOffsets"})
        List<FingerprintOffsetDto> topOffsets,

        @JsonProperty("score_gap")
        @JsonAlias({"score_gap", "scoreGap"})
        Double scoreGap,

        Double confidence,

        String title,

        String artist,

        @JsonProperty("s3_key")
        @JsonAlias({"s3_key", "s3Key"})
        String s3Key
) {
}
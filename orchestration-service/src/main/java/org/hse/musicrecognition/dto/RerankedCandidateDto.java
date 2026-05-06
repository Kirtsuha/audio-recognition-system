package org.hse.musicrecognition.dto;

import com.fasterxml.jackson.annotation.JsonAlias;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public record RerankedCandidateDto(

        @JsonAlias({"track_id", "trackId"})
        Integer trackId,

        String title,

        String artist,

        @JsonAlias({"s3_key", "s3Key"})
        String s3Key,

        @JsonAlias({"best_offset", "bestOffset"})
        Double bestOffset,

        @JsonAlias({"best_offset_sec", "bestOffsetSec"})
        Double bestOffsetSec,

        @JsonAlias({"fingerprint_confidence", "fingerprintConfidence"})
        Double fingerprintConfidence,

        @JsonAlias({"fingerprint_score", "fingerprintScore"})
        Double fingerprintScore,

        @JsonAlias({"ml_similarity", "mlSimilarity"})
        Double mlSimilarity,

        @JsonAlias({"ml_probability", "mlProbability"})
        Double mlProbability,

        @JsonAlias({"final_confidence", "finalConfidence"})
        Double finalConfidence,

        @JsonAlias({"aligned_matches", "alignedMatches"})
        Integer alignedMatches,

        @JsonAlias({"total_matches", "totalMatches"})
        Integer totalMatches,

        @JsonAlias({"offset_count", "offsetCount"})
        Integer offsetCount,

        Double coverage,

        @JsonAlias({"unique_coverage", "uniqueCoverage"})
        Double uniqueCoverage,

        @JsonAlias({"score_gap", "scoreGap"})
        Double scoreGap
) {
}
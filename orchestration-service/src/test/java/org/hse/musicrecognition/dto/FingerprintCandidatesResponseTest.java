package org.hse.musicrecognition.dto;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

class FingerprintCandidatesResponseTest {

    @Test
    void bestCandidatePrefersTrackIdMatch() {
        FingerprintCandidateDto first = candidate(1, "First");
        FingerprintCandidateDto second = candidate(2, "Second");
        FingerprintCandidatesResponse response = new FingerprintCandidatesResponse(
                true,
                2,
                0.9,
                null,
                "fingerprint",
                2,
                100,
                80,
                30,
                10,
                List.of(first, second)
        );

        assertThat(response.isMatched()).isTrue();
        assertThat(response.bestCandidate()).isEqualTo(second);
    }

    @Test
    void bestCandidateFallsBackToFirstCandidate() {
        FingerprintCandidateDto first = candidate(1, "First");
        FingerprintCandidatesResponse response = new FingerprintCandidatesResponse(
                false,
                99,
                0.2,
                "low_confidence",
                "fingerprint",
                1,
                10,
                8,
                1,
                0,
                List.of(first)
        );

        assertThat(response.isMatched()).isFalse();
        assertThat(response.bestCandidate()).isEqualTo(first);
    }

    private static FingerprintCandidateDto candidate(Integer trackId, String title) {
        return new FingerprintCandidateDto(
                trackId,
                0,
                0.0,
                10,
                12,
                1,
                0.5,
                0.5,
                List.of(),
                3.0,
                0.8,
                title,
                "Artist",
                "track-" + trackId + ".wav"
        );
    }
}

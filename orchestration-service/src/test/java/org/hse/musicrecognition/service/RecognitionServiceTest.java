package org.hse.musicrecognition.service;

import org.hse.musicrecognition.config.RecognitionProperties;
import org.hse.musicrecognition.dto.FingerprintCandidateDto;
import org.hse.musicrecognition.dto.FingerprintCandidatesResponse;
import org.hse.musicrecognition.dto.MlRerankRequest;
import org.hse.musicrecognition.dto.MlRerankResponse;
import org.hse.musicrecognition.dto.RecognitionResponse;
import org.hse.musicrecognition.dto.RerankedCandidateDto;
import org.hse.musicrecognition.dto.StoredAudioRef;
import org.hse.musicrecognition.exception.BadRequestException;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyDouble;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class RecognitionServiceTest {

    private final FingerprintClient fingerprintClient = mock(FingerprintClient.class);
    private final MlRerankerClient mlRerankerClient = mock(MlRerankerClient.class);
    private final QueryAudioStorageService queryAudioStorageService = mock(QueryAudioStorageService.class);
    private final HistoryService historyService = mock(HistoryService.class);
    private final RecognitionProperties properties = new RecognitionProperties();
    private final RecognitionService service = new RecognitionService(
            fingerprintClient,
            mlRerankerClient,
            queryAudioStorageService,
            historyService,
            properties
    );

    @Test
    void rejectsEmptyAudio() {
        assertThatThrownBy(() -> service.recognize("user", "a.wav", "audio/wav", "web", new byte[0]))
                .isInstanceOf(BadRequestException.class)
                .hasMessageContaining("empty");
    }

    @Test
    void returnsFingerprintMatchWithoutReranker() {
        properties.getFingerprint().setTopK(3);
        byte[] audio = new byte[]{1, 2, 3};
        when(queryAudioStorageService.save(anyString(), anyString(), anyString(), any()))
                .thenReturn(new StoredAudioRef("queries", "debug/request/input.wav"));
        when(fingerprintClient.retrieveCandidates(audio, "query.wav", 3))
                .thenReturn(fingerprintResponse(true, 10, 0.91));

        RecognitionResponse response = service.recognize(
                "alice",
                "query.wav",
                "audio/wav",
                "web",
                audio
        );

        assertThat(response.isMatch()).isTrue();
        assertThat(response.getTrackId()).isEqualTo("10");
        assertThat(response.getTitle()).isEqualTo("Song 10");
        assertThat(response.getSource()).isEqualTo("fingerprint");
        verify(mlRerankerClient, never()).rerank(any());
        verify(historyService).save(
                "alice",
                "query.wav",
                "10",
                "Song 10",
                "Artist",
                0.91,
                "fingerprint"
        );
    }

    @Test
    void fallsBackToRerankerWhenFingerprintDoesNotMatch() {
        properties.getStorage().setReferenceAudioBucket("full-tracks");
        byte[] audio = new byte[]{1, 2, 3};
        FingerprintCandidatesResponse fingerprint = fingerprintResponse(false, 42, 0.33);
        when(queryAudioStorageService.save(anyString(), anyString(), anyString(), any()))
                .thenReturn(new StoredAudioRef("queries", "debug/request/input.wav"));
        when(fingerprintClient.retrieveCandidates(audio, "query.wav", properties.getFingerprint().getTopK()))
                .thenReturn(fingerprint);
        when(mlRerankerClient.rerank(any())).thenReturn(rerankResponse(true, 42, 0.77));

        RecognitionResponse response = service.recognize(
                "bob",
                "query.wav",
                "audio/wav",
                "web",
                audio
        );

        assertThat(response.isMatch()).isTrue();
        assertThat(response.getTrackId()).isEqualTo("42");
        assertThat(response.getSource()).isEqualTo("ml-reranker");
        assertThat(response.getConfidence()).isEqualTo(0.77);

        ArgumentCaptor<MlRerankRequest> captor = ArgumentCaptor.forClass(MlRerankRequest.class);
        verify(mlRerankerClient).rerank(captor.capture());
        assertThat(captor.getValue().queryAudio().bucket()).isEqualTo("queries");
        assertThat(captor.getValue().referenceBucket()).isEqualTo("full-tracks");
        assertThat(captor.getValue().fingerprint()).isEqualTo(fingerprint);
    }

    @Test
    void returnsNotFoundWhenRerankerUnavailable() {
        byte[] audio = new byte[]{1};
        when(queryAudioStorageService.save(anyString(), anyString(), anyString(), any()))
                .thenReturn(new StoredAudioRef("queries", "debug/request/input.wav"));
        when(fingerprintClient.retrieveCandidates(audio, "query.wav", properties.getFingerprint().getTopK()))
                .thenReturn(fingerprintResponse(false, 1, 0.25));
        when(mlRerankerClient.rerank(any()))
                .thenThrow(new ExternalServiceUnavailableException("down"));

        RecognitionResponse response = service.recognize(
                "bob",
                "query.wav",
                "audio/wav",
                "web",
                audio
        );

        assertThat(response.isMatch()).isFalse();
        assertThat(response.getSource()).isEqualTo("ml-reranker-unavailable");
        assertThat(response.getConfidence()).isEqualTo(0.25);
    }

    private static FingerprintCandidatesResponse fingerprintResponse(
            boolean matched,
            Integer trackId,
            Double confidence
    ) {
        return new FingerprintCandidatesResponse(
                matched,
                trackId,
                confidence,
                matched ? null : "low_confidence",
                "fingerprint",
                1,
                100,
                80,
                20,
                3,
                List.of(new FingerprintCandidateDto(
                        trackId,
                        0,
                        0.0,
                        20,
                        25,
                        1,
                        0.2,
                        0.25,
                        List.of(),
                        4.0,
                        confidence,
                        "Song " + trackId,
                        "Artist",
                        "track-" + trackId + ".wav"
                ))
        );
    }

    private static MlRerankResponse rerankResponse(boolean matched, Integer trackId, Double confidence) {
        RerankedCandidateDto best = new RerankedCandidateDto(
                trackId,
                "Song " + trackId,
                "Artist",
                "track-" + trackId + ".wav",
                0.0,
                0.0,
                0.4,
                20.0,
                0.9,
                0.95,
                confidence,
                20,
                25,
                1,
                0.2,
                0.25,
                4.0
        );

        return new MlRerankResponse(
                "request",
                "ml_reranker",
                matched,
                matched ? null : "below_no_match_threshold",
                best,
                List.of(best),
                Map.of("total", 10)
        );
    }
}

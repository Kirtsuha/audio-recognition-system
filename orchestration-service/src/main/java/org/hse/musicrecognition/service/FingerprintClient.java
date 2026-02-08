package org.hse.musicrecognition.service;


import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.FingerprintResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
@RequiredArgsConstructor
public class FingerprintClient {

    private final WebClient webClient;

    @Value("${fingerprint.service.url}")
    private String fingerprintUrl;

    public Mono<FingerprintResponse> recognize(byte[] audioBytes) {
        return webClient.post()
                .uri(fingerprintUrl)
                .contentType(MediaType.APPLICATION_OCTET_STREAM)
                .bodyValue(audioBytes)
                .retrieve()
                .bodyToMono(FingerprintResponse.class);
    }
}


package org.hse.musicrecognition.service;


import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.FingerprintResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

@Service
@RequiredArgsConstructor
public class FingerprintClient {

    private final WebClient webClient;

    @Value("${fingerprint.service.url}")
    private String fingerprintUrl;

    public Mono<FingerprintResponse> recognize(byte[] audioBytes) {
        MultipartBodyBuilder bodyBuilder = new MultipartBodyBuilder();
        bodyBuilder.part("file", audioBytes)
                .header(HttpHeaders.CONTENT_TYPE, MediaType.APPLICATION_OCTET_STREAM_VALUE)
                .header("Content-Disposition", "form-data; name=\"file\"; filename=\"audio.wav\"");

        return webClient.post()
                .uri(fingerprintUrl)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(bodyBuilder.build()))
                .retrieve()
                .bodyToMono(FingerprintResponse.class);
    }
}


package org.hse.musicrecognition.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.hse.musicrecognition.dto.MlRerankRequest;
import org.hse.musicrecognition.dto.MlRerankResponse;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;

@Slf4j
@Service
@RequiredArgsConstructor
public class MlRerankerClient {

    private final ObjectMapper objectMapper;

    @Value("${ml-reranker.service.url}")
    private String mlRerankerUrl;

    private final HttpClient httpClient = HttpClient.newBuilder()
            .version(HttpClient.Version.HTTP_1_1)
            .build();

    public MlRerankResponse rerank(MlRerankRequest request) {
        try {
            String requestBody = objectMapper.writeValueAsString(request);
            String url = mlRerankerUrl + "/rerank";

            log.info(
                    "Calling ML reranker url={} requestId={} queryBucket={} queryKey={} referenceBucket={} candidates={}",
                    url,
                    request.requestId(),
                    request.queryAudio() == null ? null : request.queryAudio().bucket(),
                    request.queryAudio() == null ? null : request.queryAudio().key(),
                    request.referenceBucket(),
                    request.fingerprint() == null || request.fingerprint().candidates() == null
                            ? 0
                            : request.fingerprint().candidates().size()
            );

            log.info("ML reranker request body={}", requestBody);

            HttpRequest httpRequest = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .version(HttpClient.Version.HTTP_1_1)
                    .header("Content-Type", "application/json; charset=utf-8")
                    .header("Accept", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(requestBody, StandardCharsets.UTF_8))
                    .build();

            HttpResponse<String> response = httpClient.send(
                    httpRequest,
                    HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );

            log.info(
                    "ML reranker response status={} body={}",
                    response.statusCode(),
                    response.body()
            );

            if (response.statusCode() >= 400) {
                throw new ExternalServiceUnavailableException(
                        "ML reranker rejected request: " + response.body()
                );
            }

            return objectMapper.readValue(response.body(), MlRerankResponse.class);

        } catch (ExternalServiceUnavailableException e) {
            throw e;
        } catch (Exception e) {
            log.error("ML reranker service call failed", e);

            throw new ExternalServiceUnavailableException(
                    "ML reranker service is unavailable",
                    e
            );
        }
    }
}
package org.hse.musicrecognition.service;

import org.hse.musicrecognition.config.MlProperties;
import org.hse.musicrecognition.dto.MlRecognitionRequest;
import org.hse.musicrecognition.dto.MlRecognitionResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.stereotype.Service;

import java.util.Base64;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

@Service
public class MlFallbackGateway {

    private final KafkaTemplate<String, MlRecognitionRequest> kafkaTemplate;
    private final Map<String, CompletableFuture<MlRecognitionResponse>> pendingRequests = new ConcurrentHashMap<>();
    private final String requestTopic;
    private final long timeoutMs;

    public MlFallbackGateway(
            KafkaTemplate<String, MlRecognitionRequest> kafkaTemplate, MlProperties mlProperties
    ) {
        this.kafkaTemplate = kafkaTemplate;
        this.requestTopic = mlProperties.getKafka().getRequestTopic();
        this.timeoutMs = mlProperties.getTimeoutMs();
    }

    public MlRecognitionResponse recognize(byte[] audioBytes, String filename) throws Exception {
        String requestId = UUID.randomUUID().toString();
        CompletableFuture<MlRecognitionResponse> future = new CompletableFuture<>();
        pendingRequests.put(requestId, future);

        MlRecognitionRequest request = new MlRecognitionRequest(
                requestId,
                filename,
                Base64.getEncoder().encodeToString(audioBytes)
        );

        kafkaTemplate.send(requestTopic, requestId, request);

        try {
            MlRecognitionResponse response = future.get(timeoutMs, TimeUnit.MILLISECONDS);
            if (response.error() != null && !response.error().isBlank()) {
                throw new IllegalStateException("ML fallback failed: " + response.error());
            }
            return response;
        } finally {
            pendingRequests.remove(requestId);
        }
    }

    @KafkaListener(topics = "${ml.kafka.response-topic}")
    public void receiveResponse(MlRecognitionResponse response) {
        if (response == null || response.requestId() == null) {
            return;
        }

        CompletableFuture<MlRecognitionResponse> future = pendingRequests.get(response.requestId());
        if (future != null) {
            future.complete(response);
        }
    }
}

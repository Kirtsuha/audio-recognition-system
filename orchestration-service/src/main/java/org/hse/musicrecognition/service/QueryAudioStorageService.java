package org.hse.musicrecognition.service;

import io.minio.PutObjectArgs;
import io.minio.MinioClient;
import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.config.RecognitionProperties;
import org.hse.musicrecognition.dto.StoredAudioRef;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.springframework.stereotype.Service;

import java.io.ByteArrayInputStream;
import java.util.Locale;

@Service
@RequiredArgsConstructor
public class QueryAudioStorageService {

    private final MinioClient minioClient;
    private final RecognitionProperties recognitionProperties;

    public StoredAudioRef save(
            String requestId,
            String originalFilename,
            String contentType,
            byte[] audioBytes
    ) {
        String bucket = recognitionProperties.getStorage().getQueryAudioBucket();
        String key = buildObjectKey(requestId, originalFilename);

        try {
            minioClient.putObject(
                    PutObjectArgs.builder()
                            .bucket(bucket)
                            .object(key)
                            .stream(new ByteArrayInputStream(audioBytes), audioBytes.length, -1)
                            .contentType(resolveContentType(contentType, originalFilename))
                            .build()
            );

            return new StoredAudioRef(bucket, key);
        } catch (Exception e) {
            throw new ExternalServiceUnavailableException("Failed to upload query audio to MinIO", e);
        }
    }

    private String buildObjectKey(String requestId, String originalFilename) {
        String prefix = recognitionProperties.getStorage().getQueryAudioPrefix();
        String extension = extractExtension(originalFilename);

        if (prefix == null || prefix.isBlank()) {
            return requestId + "/input" + extension;
        }

        return prefix + "/" + requestId + "/input" + extension;
    }

    private String extractExtension(String filename) {
        if (filename == null || filename.isBlank()) {
            return ".wav";
        }

        int dotIndex = filename.lastIndexOf('.');
        if (dotIndex < 0 || dotIndex == filename.length() - 1) {
            return ".wav";
        }

        String extension = filename.substring(dotIndex).toLowerCase(Locale.ROOT);

        if (extension.length() > 16) {
            return ".wav";
        }

        return extension;
    }

    private String resolveContentType(String contentType, String filename) {
        if (contentType != null && !contentType.isBlank()) {
            return contentType;
        }

        String extension = extractExtension(filename);

        return switch (extension) {
            case ".mp3" -> "audio/mpeg";
            case ".wav" -> "audio/wav";
            default -> "application/octet-stream";
        };
    }
}
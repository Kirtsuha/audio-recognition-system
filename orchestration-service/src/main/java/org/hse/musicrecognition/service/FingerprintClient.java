package org.hse.musicrecognition.service;


import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.FingerprintResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestTemplate;


@Service
@RequiredArgsConstructor
public class FingerprintClient {

    private final RestClient restClient;

    @Value("${fingerprint.service.url}")
    private String fingerprintUrl;

    public FingerprintResponse recognize(byte[] audioBytes) {

        ByteArrayResource resource = new ByteArrayResource(audioBytes) {
            @Override
            public String getFilename() {
                return "audio.wav";
            }
        };

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", resource);

        return restClient.post()
                .uri(fingerprintUrl)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(body)
                .retrieve()
                .body(FingerprintResponse.class);
    }
}


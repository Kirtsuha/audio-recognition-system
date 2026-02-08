import org.hse.musicrecognition.Main;
import org.hse.musicrecognition.config.WebClientConfig;
import org.hse.musicrecognition.controller.RecognitionController;
import org.hse.musicrecognition.dto.FingerprintResponse;
import org.hse.musicrecognition.service.FingerprintClient;
import org.junit.jupiter.api.Test;
import org.mockito.Mock;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.webflux.test.autoconfigure.WebFluxTest;
import org.springframework.boot.webtestclient.autoconfigure.AutoConfigureWebTestClient;
import org.springframework.context.annotation.Import;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.reactive.server.WebTestClient;
import org.springframework.web.reactive.function.BodyInserters;
import reactor.core.publisher.Mono;


import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.when;

@SpringBootTest(classes = Main.class)
@AutoConfigureWebTestClient
class RecognitionControllerTest {

    @Autowired
    private WebTestClient webTestClient;

    @MockitoBean
    private FingerprintClient fingerprintClient;

    @Test
    void shouldReturnRecognitionResult() {
        FingerprintResponse response = new FingerprintResponse();
        response.setMatch(true);
        response.setTitle("Song");
        response.setArtist("Artist");
        response.setConfidence(0.95);

        when(fingerprintClient.recognize(any()))
                .thenReturn(Mono.just(response));

        webTestClient.post()
                .uri("/api/recognition")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData(
                        "file",
                        new ByteArrayResource("test".getBytes()) {
                            @Override
                            public String getFilename() { return "test.wav"; }
                        }
                ))
                .exchange()
                .expectStatus().isOk()
                .expectBody()
                .jsonPath("$.match").isEqualTo(true)
                .jsonPath("$.title").isEqualTo("Song");
    }
}

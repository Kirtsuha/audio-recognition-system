package org.hse.musicrecognition.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.RecognitionResponse;
import org.hse.musicrecognition.service.FingerprintClient;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import reactor.core.publisher.Mono;

@RestController
@RequestMapping("/api/recognition")
@RequiredArgsConstructor
public class RecognitionController {

    private final FingerprintClient fingerprintClient;

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Распознать аудиофайл",
            description = "Принимает аудиофайл и возвращает информацию о треке",
            responses = {@ApiResponse(responseCode = "200", description = "Распознано успешно")})
    public Mono<RecognitionResponse> recognize(@RequestPart("file") FilePart filePart) {

        return filePart.content()
                .reduce(DataBuffer::write)
                .map(dataBuffer -> {
                    byte[] bytes = new byte[dataBuffer.readableByteCount()];
                    dataBuffer.read(bytes);
                    DataBufferUtils.release(dataBuffer);
                    return bytes;
                })
                .flatMap(fingerprintClient::recognize)
                .map(fp -> new RecognitionResponse(
                        fp.isMatch(),
                        fp.getTitle(),
                        fp.getArtist(),
                        fp.getConfidence()
                ));
    }
}


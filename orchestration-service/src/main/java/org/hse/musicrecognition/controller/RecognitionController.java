package org.hse.musicrecognition.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.RecognitionHistory;
import org.hse.musicrecognition.dto.RecognitionResponse;
import org.hse.musicrecognition.service.FingerprintClient;
import org.hse.musicrecognition.service.HistoryService;
import org.springframework.core.io.buffer.DataBuffer;
import org.springframework.core.io.buffer.DataBufferUtils;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.security.Principal;
import java.util.List;

@RestController
@RequestMapping("/api/recognition")
@RequiredArgsConstructor
public class RecognitionController {

    private final FingerprintClient fingerprintClient;
    private final HistoryService historyService;

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Распознать аудиофайл",
            description = "Принимает аудиофайл и возвращает информацию о треке",
            responses = {@ApiResponse(responseCode = "200", description = "Распознано успешно")})
    public Mono<RecognitionResponse> recognize(@RequestPart("file") FilePart filePart, Principal principal) {

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
                ))
                .flatMap(resp ->
                        historyService.save(principal.getName(), resp)
                                .thenReturn(resp)
                );
    }
}


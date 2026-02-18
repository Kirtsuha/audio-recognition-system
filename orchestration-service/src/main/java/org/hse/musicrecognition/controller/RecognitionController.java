package org.hse.musicrecognition.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.RecognitionHistory;
import org.hse.musicrecognition.dto.FingerprintResponse;
import org.hse.musicrecognition.dto.RecognitionResponse;
import org.hse.musicrecognition.service.FingerprintClient;
import org.hse.musicrecognition.service.HistoryService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.security.Principal;

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
    public RecognitionResponse recognize(@RequestPart("file") MultipartFile file,
                                         Principal principal) throws IOException {
        byte[] bytes = file.getBytes();

        FingerprintResponse fp = fingerprintClient.recognize(bytes);

        RecognitionResponse resp = new RecognitionResponse(
                fp.isMatch(),
                fp.getTitle(),
                fp.getArtist(),
                fp.getConfidence()
        );

        historyService.save(
                principal.getName(),
                file.getOriginalFilename(),
                resp.getTitle(),
                resp.getArtist(),
                resp.getConfidence()
        );

        return resp;
    }

}


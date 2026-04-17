package org.hse.musicrecognition.controller;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.dto.RecognitionResponse;
import org.hse.musicrecognition.service.RecognitionService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.security.Principal;

@RestController
@RequestMapping("/api/recognition")
@RequiredArgsConstructor
public class RecognitionController {

    private final RecognitionService recognitionService;

    @PostMapping(consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "Распознать аудиофайл",
            description = "Принимает аудиофайл и возвращает информацию о треке",
            responses = {@ApiResponse(responseCode = "200", description = "Распознано успешно")})
    public RecognitionResponse recognize(@RequestPart("file") MultipartFile file,
                                         Principal principal) throws Exception {
        byte[] bytes = file.getBytes();
        return recognitionService.recognize(principal.getName(), file.getOriginalFilename(), bytes);
    }

}


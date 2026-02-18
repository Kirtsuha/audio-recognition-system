package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.RecognitionHistory;
import org.hse.musicrecognition.service.HistoryService;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.security.Principal;
import java.util.List;

@RestController
@RequestMapping("/api/history")
@RequiredArgsConstructor
public class HistoryController {

    private final HistoryService historyService;

    @GetMapping
    public List<RecognitionHistory> history(Principal principal) {
        return historyService.get(principal.getName());
    }
}

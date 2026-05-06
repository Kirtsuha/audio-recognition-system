package org.hse.musicrecognition.controller;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.RecognitionHistory;
import org.hse.musicrecognition.service.HistoryService;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

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

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteOne(
            @PathVariable Long id,
            Principal principal,
            Authentication authentication
    ) {
        historyService.deleteOne(
                principal.getName(),
                isAdmin(authentication),
                id
        );

        return ResponseEntity.noContent().build();
    }

    @DeleteMapping
    public ResponseEntity<Void> deleteOwnHistory(Principal principal) {
        historyService.deleteOwnHistory(principal.getName());
        return ResponseEntity.noContent().build();
    }

    @DeleteMapping("/{userId}/history")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Void> deleteUserHistory(
            @PathVariable Long userId
    ) {
        historyService.deleteUserHistoryByAdmin(userId);
        return ResponseEntity.noContent().build();
    }

    private boolean isAdmin(Authentication authentication) {
        return authentication != null
                && authentication.getAuthorities()
                .stream()
                .anyMatch(authority -> "ROLE_ADMIN".equals(authority.getAuthority()));
    }
}
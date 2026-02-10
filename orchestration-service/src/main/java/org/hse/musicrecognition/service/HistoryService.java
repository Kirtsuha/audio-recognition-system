package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.RecognitionHistory;
import org.hse.musicrecognition.domain.User;
import org.hse.musicrecognition.repository.HistoryRepository;
import org.hse.musicrecognition.repository.UserRepository;
import org.springframework.stereotype.Service;

import java.time.LocalDateTime;
import java.util.List;

@Service
@RequiredArgsConstructor
public class HistoryService {

    private final HistoryRepository repo;
    private final UserRepository userRepo;

    public void save(String username, String filename, String title, String artist, double confidence) {
        User user = userRepo.findByUsername(username).orElseThrow();

        repo.save(RecognitionHistory.builder()
                .filename(filename)
                .title(title)
                .artist(artist)
                .confidence(confidence)
                .createdAt(LocalDateTime.now())
                .user(user)
                .build());
    }

    public List<RecognitionHistory> get(String username) {
        return repo.findByUserUsername(username);
    }
}

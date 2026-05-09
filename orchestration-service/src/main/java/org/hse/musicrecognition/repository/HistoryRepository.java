package org.hse.musicrecognition.repository;

import org.hse.musicrecognition.domain.RecognitionHistory;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface HistoryRepository extends JpaRepository<RecognitionHistory, Long> {

    List<RecognitionHistory> findByUserUsername(String username);

    void deleteByUserUsername(String username);

    void deleteByUserId(Long userId);
}
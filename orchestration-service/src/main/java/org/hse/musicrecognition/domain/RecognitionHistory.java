package org.hse.musicrecognition.domain;

import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.Id;
import jakarta.persistence.ManyToOne;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Entity
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class RecognitionHistory {
    @Id
    @GeneratedValue
    private Long id;

    @ManyToOne
    private User user;

    private String filename;
    private String trackId;
    private String title;
    private String artist;
    private double confidence;
    private String source;

    private LocalDateTime createdAt;

}

package org.hse.musicrecognition.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;

@Entity
@Table(name = "catalog_update_job_steps")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CatalogUpdateJobStep {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY, optional = false)
    @JoinColumn(name = "job_id", nullable = false)
    private CatalogUpdateJob job;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 96)
    private CatalogUpdateStepName name;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 64)
    private CatalogUpdateJobStatus status;

    private Instant startedAt;

    private Instant finishedAt;

    @Column(columnDefinition = "text")
    private String errorMessage;
}

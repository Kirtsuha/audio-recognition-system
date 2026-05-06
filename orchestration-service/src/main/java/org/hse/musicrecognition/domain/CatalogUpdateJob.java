package org.hse.musicrecognition.domain;

import jakarta.persistence.*;
import lombok.*;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

@Entity
@Table(name = "catalog_update_jobs")
@Getter
@Setter
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CatalogUpdateJob {

    @Id
    private UUID id;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 64)
    private CatalogUpdateJobType type;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 64)
    private CatalogUpdateJobStatus status;

    private Long createdByUserId;

    @Column(nullable = false)
    private Instant startedAt;

    private Instant finishedAt;

    @Column(columnDefinition = "text")
    private String errorMessage;

    @Builder.Default
    @OneToMany(mappedBy = "job", cascade = CascadeType.ALL, orphanRemoval = true, fetch = FetchType.EAGER)
    @OrderBy("id ASC")
    private List<CatalogUpdateJobStep> steps = new ArrayList<>();

    public void addStep(CatalogUpdateJobStep step) {
        step.setJob(this);
        steps.add(step);
    }
}

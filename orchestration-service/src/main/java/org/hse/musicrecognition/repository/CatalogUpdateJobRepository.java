package org.hse.musicrecognition.repository;

import org.hse.musicrecognition.domain.CatalogUpdateJob;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.UUID;

public interface CatalogUpdateJobRepository extends JpaRepository<CatalogUpdateJob, UUID> {
}

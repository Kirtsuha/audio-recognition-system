package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.domain.*;
import org.hse.musicrecognition.dto.*;
import org.hse.musicrecognition.exception.BadRequestException;
import org.hse.musicrecognition.exception.NotFoundException;
import org.hse.musicrecognition.repository.CatalogUpdateJobRepository;
import org.hse.musicrecognition.repository.UserRepository;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.web.multipart.MultipartFile;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

@Service
@RequiredArgsConstructor
public class CatalogAdminService {

    private final FingerprintClient fingerprintClient;
    private final MlPipelineClient mlPipelineClient;
    private final MlRerankerClient mlRerankerClient;
    private final CatalogUpdateJobRepository jobRepository;
    private final UserRepository userRepository;

    @Value("${catalog.s3.bucket:tracks}")
    private String catalogBucket;

    @Value("${catalog.s3.prefix:}")
    private String catalogPrefix;

    public AdminTrackUploadResponse uploadTrack(
            String username,
            MultipartFile file,
            String title,
            String artist,
            String album
    ) {
        if (file == null || file.isEmpty()) {
            throw new BadRequestException("Track file is empty");
        }

        if (title == null || title.isBlank()) {
            throw new BadRequestException("title is required");
        }

        if (artist == null || artist.isBlank()) {
            throw new BadRequestException("artist is required");
        }

        FingerprintTrackUploadResponse response = fingerprintClient.uploadTrack(
                file,
                title,
                artist,
                album
        );

        return new AdminTrackUploadResponse(
                response.trackId(),
                response.title(),
                response.artist(),
                response.album(),
                response.status() == null ? "CREATED" : response.status()
        );
    }

    @Transactional
    public AdminBulkUploadResponse uploadBulk(
            String username,
            MultipartFile file,
            MultipartFile manifest,
            String bucket,
            String prefix,
            Integer maxFiles
    ) {
        if (file == null || file.isEmpty()) {
            throw new BadRequestException("Archive file is empty");
        }

        if (manifest == null || manifest.isEmpty()) {
            throw new BadRequestException("Manifest file is empty");
        }

        Map<String, Object> summary = fingerprintClient.uploadArchive(
                file,
                manifest,
                bucket,
                prefix,
                maxFiles
        );

        return new AdminBulkUploadResponse("COMPLETED", summary);
    }

    @Transactional
    public CatalogSyncResponse runIncrementalSync(String username) {
        CatalogUpdateJob job = createJob(username, CatalogUpdateJobType.INCREMENTAL_SYNC);

        try {
            runStep(job, CatalogUpdateStepName.FINGERPRINT_INDEX, () ->
                    fingerprintClient.indexS3(catalogBucket, catalogPrefix)
            );

            runStep(job, CatalogUpdateStepName.ML_INCREMENTAL_SYNC, () -> {
                mlPipelineClient.startIncrementalSync(catalogBucket, catalogPrefix, true);
                mlPipelineClient.waitForJob("incremental-sync");
            });

            runStep(job, CatalogUpdateStepName.ML_RELOAD_RUNTIME, mlPipelineClient::reloadRuntime);

            runStep(job, CatalogUpdateStepName.RERANKER_REFERENCE_BUILD, () -> {
                mlPipelineClient.startBuildRerankerReferenceEmbeddings();
                mlPipelineClient.waitForJob("build-reranker-reference-embeddings");
            });

            runStep(job, CatalogUpdateStepName.RERANKER_RELOAD_REFERENCE_STORE, mlRerankerClient::reloadReferenceStore);

            completeJob(job);
        } catch (RuntimeException e) {
            failJob(job, e);
            throw e;
        }

        return new CatalogSyncResponse(
                job.getId(),
                job.getStatus().name(),
                toStepResponses(job.getSteps())
        );
    }

    @Transactional
    public CatalogJobStartedResponse startFullRebuild(String username) {
        CatalogUpdateJob job = createJob(username, CatalogUpdateJobType.FULL_REBUILD);

        CompletableFuture.runAsync(() -> runFullRebuild(job.getId()));

        return new CatalogJobStartedResponse(job.getId(), "STARTED");
    }

    @Transactional(readOnly = true)
    public CatalogJobResponse getJob(UUID jobId) {
        CatalogUpdateJob job = jobRepository.findById(jobId)
                .orElseThrow(() -> new NotFoundException("Catalog update job not found"));

        return toJobResponse(job);
    }

    @Transactional
    public void runFullRebuild(UUID jobId) {
        CatalogUpdateJob job = jobRepository.findById(jobId)
                .orElseThrow(() -> new NotFoundException("Catalog update job not found"));

        try {
            runStep(job, CatalogUpdateStepName.FINGERPRINT_INDEX, () ->
                    fingerprintClient.indexS3(catalogBucket, catalogPrefix)
            );

            runStep(job, CatalogUpdateStepName.ML_FULL_BUILD, () -> {
                mlPipelineClient.startFullRun(catalogBucket, catalogPrefix, true);
                mlPipelineClient.waitForJob("full-build");
            });

            runStep(job, CatalogUpdateStepName.ML_RELOAD_RUNTIME, mlPipelineClient::reloadRuntime);

            runStep(job, CatalogUpdateStepName.RERANKER_REFERENCE_BUILD, () -> {
                mlPipelineClient.startBuildRerankerReferenceEmbeddings();
                mlPipelineClient.waitForJob("build-reranker-reference-embeddings");
            });

            runStep(job, CatalogUpdateStepName.RERANKER_RELOAD_MODEL, mlRerankerClient::reloadModel);
            runStep(job, CatalogUpdateStepName.RERANKER_RELOAD_REFERENCE_STORE, mlRerankerClient::reloadReferenceStore);

            completeJob(job);
        } catch (RuntimeException e) {
            failJob(job, e);
        }
    }

    private CatalogUpdateJob createJob(String username, CatalogUpdateJobType type) {
        Long userId = userRepository.findByUsername(username)
                .map(user -> user.getId())
                .orElse(null);

        CatalogUpdateJob job = CatalogUpdateJob.builder()
                .id(UUID.randomUUID())
                .type(type)
                .status(CatalogUpdateJobStatus.RUNNING)
                .createdByUserId(userId)
                .startedAt(Instant.now())
                .build();

        return jobRepository.save(job);
    }

    private void runStep(CatalogUpdateJob job, CatalogUpdateStepName name, Runnable action) {
        CatalogUpdateJobStep step = CatalogUpdateJobStep.builder()
                .name(name)
                .status(CatalogUpdateJobStatus.RUNNING)
                .startedAt(Instant.now())
                .build();

        job.addStep(step);
        jobRepository.save(job);

        try {
            action.run();
            step.setStatus(CatalogUpdateJobStatus.COMPLETED);
            step.setFinishedAt(Instant.now());
            jobRepository.save(job);
        } catch (RuntimeException e) {
            step.setStatus(CatalogUpdateJobStatus.FAILED);
            step.setFinishedAt(Instant.now());
            step.setErrorMessage(e.getMessage());
            jobRepository.save(job);
            throw e;
        }
    }

    private void completeJob(CatalogUpdateJob job) {
        job.setStatus(CatalogUpdateJobStatus.COMPLETED);
        job.setFinishedAt(Instant.now());
        jobRepository.save(job);
    }

    private void failJob(CatalogUpdateJob job, RuntimeException e) {
        job.setStatus(CatalogUpdateJobStatus.FAILED);
        job.setFinishedAt(Instant.now());
        job.setErrorMessage(e.getMessage());
        jobRepository.save(job);
    }

    private CatalogJobResponse toJobResponse(CatalogUpdateJob job) {
        return new CatalogJobResponse(
                job.getId(),
                job.getType().name(),
                job.getStatus().name(),
                job.getStartedAt(),
                job.getFinishedAt(),
                job.getErrorMessage(),
                toStepResponses(job.getSteps())
        );
    }

    private List<CatalogJobStepResponse> toStepResponses(List<CatalogUpdateJobStep> steps) {
        return steps.stream()
                .map(step -> new CatalogJobStepResponse(
                        step.getName().name(),
                        step.getStatus().name(),
                        step.getErrorMessage()
                ))
                .toList();
    }
}

package org.hse.musicrecognition.service;

import lombok.RequiredArgsConstructor;
import org.hse.musicrecognition.exception.ExternalServiceUnavailableException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;
import org.springframework.web.client.RestClientException;

import java.util.Map;

@Service
@RequiredArgsConstructor
public class MlPipelineClient {

    private final RestClient restClient;

    @Value("${ml.service.url}")
    private String mlServiceUrl;

    @Value("${catalog.active-artifacts-dir:/app/artifacts/active}")
    private String activeArtifactsDir;

    public void startIncrementalSync(String bucket, String prefix, boolean promote) {
        post(
                "/pipeline/incremental-sync",
                Map.of(
                        "bucket", bucket,
                        "prefix", prefix == null ? "" : prefix,
                        "promote", promote,
                        "allow_incremental_when_retrain_recommended", true,
                        "index_windows_override", 96
                )
        );
    }

    public void startFullRun(String bucket, String prefix, boolean promote) {
        post(
                "/pipeline/full-run",
                Map.of(
                        "bucket", bucket,
                        "prefix", prefix == null ? "" : prefix,
                        "promote", promote
                )
        );
    }

    public void startBuildRerankerReferenceEmbeddings() {
        post(
                "/pipeline/build-reranker-reference-embeddings",
                Map.of(
                        "model_path", activeArtifactsDir + "/model.pt",
                        "run_dir", activeArtifactsDir,
                        "index_all_prepared", true
                )
        );
    }

    public void reloadRuntime() {
        post("/admin/reload-runtime", Map.of());
    }

    public Map getStatus() {
        try {
            return restClient.get()
                    .uri(mlServiceUrl + "/pipeline/status")
                    .retrieve()
                    .body(Map.class);
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("ML pipeline service is unavailable", e);
        }
    }

    public void waitForJob(String expectedJob) {
        long deadline = System.currentTimeMillis() + 60L * 60L * 1000L;
        boolean observedExpectedJob = false;

        while (System.currentTimeMillis() < deadline) {
            Map status = getStatus();
            Map jobStatus = status == null ? null : (Map) status.get("job_status");
            Object currentJob = jobStatus == null ? null : jobStatus.get("current_job");
            Object phase = jobStatus == null ? null : jobStatus.get("phase");
            Object error = jobStatus == null ? null : jobStatus.get("last_error");

            if (expectedJob.equals(String.valueOf(currentJob))) {
                observedExpectedJob = true;
            }

            Object inProgress = status == null ? null : status.get("pipeline_in_progress");

            if (observedExpectedJob && !Boolean.TRUE.equals(inProgress)) {
                if ("failed".equals(String.valueOf(phase))) {
                    throw new ExternalServiceUnavailableException("ML pipeline failed: " + error);
                }

                return;
            }

            sleep();
        }

        throw new ExternalServiceUnavailableException("ML pipeline did not finish in time");
    }

    private void post(String path, Map<String, Object> body) {
        try {
            restClient.post()
                    .uri(mlServiceUrl + path)
                    .body(body)
                    .retrieve()
                    .toBodilessEntity();
        } catch (RestClientException e) {
            throw new ExternalServiceUnavailableException("ML pipeline service is unavailable", e);
        }
    }

    private void sleep() {
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new ExternalServiceUnavailableException("Interrupted while waiting for ML pipeline", e);
        }
    }
}
